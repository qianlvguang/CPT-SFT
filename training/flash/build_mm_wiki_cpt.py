#!/usr/bin/env python3
"""Multimodal Wiki + Logo/Flag + Qwen-Image 2.1 + Flash-Next RAG CPT Builder.

Validates raw/synthesized CPT records, enforces causal image-prefix rules and
document-family zero-leakage train/val splits, constructs token-balanced Stage 1
(Stable) and Stage 2 (WSD Cooldown with unresolved long-tail oversampling) JSONL
shards for MS-SWIFT `megatron pt` / `swift pt`, and records SHA256 manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CANONICAL_CHANNELS: tuple[str, ...] = (
    # Active v-CPT production 6 channels:
    "ch_text_doc",
    "ch_mm_doc",
    "ch_rel_jump",
    "ch2_std_logo_flag",
    "ch5_flash_rag_raft",
    "ch6_general_replay",
    # v2.3 sch iso channels (merged from Phase A Lite):
    "ch_ip_grounding",
    "ch_ip_negative",
    # v2 / v0.4 unified Mid-Training channels:
    "ch_v2_multi_img_doc",
    "ch_sports_expanded",
    "ch_v2_cf_pair",
    "ch_v1_recycled_neg",
    "ch_qa_knowledge_prose",
    "ch_direct_anti_collapse",
    "wiki_rag_qa_cot",
    "wiki_rag_qa_direct",
    # Compatible / next-iteration Qwen-Image & stub channels:
    "ch1_wiki_text_stub",
    "ch3_qwen_img_synth",
    "ch4_audit_crawl_cf",
)

# Unified Single-Stage Instruct-Aware Mid-Training 4-bucket target token mix:
# 30% domain KB & v2 multi-img + 35% real photo replay + 15% CF/distractor + 20% sanitized QA prose/anti-collapse
MIDTRAIN_BUCKET_WEIGHTS: dict[str, float] = {
    "domain_kb_v2": 0.30,
    "real_replay": 0.35,
    "cf_distractor": 0.15,
    "qa_prose_anti_collapse": 0.20,
}

BUCKET_BY_CHANNEL: dict[str, str] = {
    "ch_qa_knowledge_prose": "qa_prose_anti_collapse",
    "ch_direct_anti_collapse": "qa_prose_anti_collapse",
    "wiki_rag_qa_cot": "qa_prose_anti_collapse",
    "wiki_rag_qa_direct": "qa_prose_anti_collapse",
    "ch6_general_replay": "real_replay",
    "ch4_audit_crawl_cf": "cf_distractor",
    "ch_ip_negative": "cf_distractor",
    "ch_v2_cf_pair": "cf_distractor",
    "ch_v1_recycled_neg": "cf_distractor",
    "ch3_qwen_img_synth": "cf_distractor",
    "ch5_flash_rag_raft": "cf_distractor",
    "ch_text_doc": "domain_kb_v2",
    "ch1_wiki_text_stub": "domain_kb_v2",
    "ch_mm_doc": "domain_kb_v2",
    "ch_v2_multi_img_doc": "domain_kb_v2",
    "ch_sports_expanded": "domain_kb_v2",
    "ch_rel_jump": "domain_kb_v2",
    "ch2_std_logo_flag": "domain_kb_v2",
    "ch_ip_grounding": "domain_kb_v2",
}

SYNTHETIC_CHANNELS = frozenset(
    {"ch3_qwen_img_synth", "ch_v2_multi_img_doc", "ch_sports_expanded", "ch_v2_cf_pair", "ch_v1_recycled_neg"}
)

STAGE1_TOKEN_WEIGHTS: dict[str, float] = {
    "ch_text_doc": 0.25,
    "ch6_general_replay": 0.22,
    "ch2_std_logo_flag": 0.16,
    "ch_mm_doc": 0.15,
    "ch5_flash_rag_raft": 0.08,
    "ch_rel_jump": 0.02,
    "ch_ip_grounding": 0.06,
    "ch_ip_negative": 0.06,
    "ch1_wiki_text_stub": 0.25,
    "ch3_qwen_img_synth": 0.15,
    "ch4_audit_crawl_cf": 0.15,
}

STAGE2_TOKEN_WEIGHTS: dict[str, float] = {
    "ch_text_doc": 0.18,
    "ch6_general_replay": 0.18,
    "ch5_flash_rag_raft": 0.20,
    "ch_mm_doc": 0.14,
    "ch_rel_jump": 0.14,
    "ch2_std_logo_flag": 0.08,
    "ch_ip_grounding": 0.04,
    "ch_ip_negative": 0.04,
    "ch1_wiki_text_stub": 0.15,
    "ch3_qwen_img_synth": 0.15,
    "ch4_audit_crawl_cf": 0.20,
}

TEXT_DOC_CHANNELS = frozenset({"ch_text_doc", "ch1_wiki_text_stub"})

IMAGE_PLACEHOLDER = "<image>"
ALLOWED_IMAGE_PREFIX_RE = re.compile(
    r"^(?:(?:图\s*\d+|待审图|商品图|参考图|原图|对比图|Image\s*\d+)\s*[：:]?\s*)*$"
)


def sha256_file(path: Path) -> str:
    """Computes SHA256 digest of a file in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Loads non-empty JSON objects from a UTF-8 JSONL file."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Writes JSONL rows using UTF-8 without ASCII escaping."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def bucket_for_channel(channel: str) -> str:
    return BUCKET_BY_CHANNEL.get(channel, "domain_kb_v2")


def _record_text_content(record: dict[str, Any]) -> str:
    messages = record.get("messages") or []
    return "\n".join(str(m.get("content") or "") for m in messages if isinstance(m, dict))


def estimate_record_tokens(
    content: str,
    num_images: int,
    tokens_per_image: int = 512,
) -> int:
    """Estimates non-padding tokens for token-level mixture allocation."""
    zh_chars = sum(1 for ch in content if "\u4e00" <= ch <= "\u9fff")
    other_chars = max(0, len(content) - zh_chars)
    text_tokens = int(round(zh_chars / 1.6 + other_chars / 4.0))
    return max(1, text_tokens + num_images * tokens_per_image)


def validate_midtrain_record(
    record: dict[str, Any],
    *,
    allow_custom_channels: bool = False,
) -> list[str]:
    """Validates both single-assistant CPT rows and ChatML CoT replay rows for Mid-Training."""
    messages = record.get("messages")
    if isinstance(messages, list) and len(messages) == 1:
        return validate_cpt_record(record, allow_custom_channels=allow_custom_channels)

    errors: list[str] = []
    channel = record.get("channel")
    if not isinstance(channel, str) or not channel.strip():
        errors.append("missing_channel")
    elif not allow_custom_channels and channel not in CANONICAL_CHANNELS:
        errors.append(f"unknown_channel:{channel}")

    doc_family_id = record.get("doc_family_id")
    if not isinstance(doc_family_id, str) or not doc_family_id.strip():
        errors.append("missing_doc_family_id")

    if not isinstance(messages, list) or len(messages) < 2:
        errors.append("midtrain_chatml_requires_at_least_2_messages")
        return errors

    roles = [m.get("role") if isinstance(m, dict) else None for m in messages]
    if "user" not in roles or roles[-1] != "assistant":
        errors.append("midtrain_chatml_requires_user_and_trailing_assistant")
    for m in messages:
        if not isinstance(m, dict) or not str(m.get("content") or "").strip():
            errors.append("empty_chatml_message")
            break

    images = record.get("images") or []
    if not isinstance(images, list) or any(not isinstance(i, str) or not i.strip() for i in images):
        errors.append("invalid_images_list")
        return errors

    combined = _record_text_content(record)
    placeholder_count = combined.count(IMAGE_PLACEHOLDER)
    if placeholder_count != len(images):
        errors.append(
            f"image_placeholder_mismatch:placeholders={placeholder_count},images={len(images)}"
        )
    return errors


def validate_cpt_record(
    record: dict[str, Any],
    *,
    allow_custom_channels: bool = False,
) -> list[str]:
    """Validates schema, image placeholder parity, and causal image-prefix ordering."""
    errors: list[str] = []

    channel = record.get("channel")
    if not isinstance(channel, str) or not channel.strip():
        errors.append("missing_channel")
    elif not allow_custom_channels and channel not in CANONICAL_CHANNELS:
        errors.append(f"unknown_channel:{channel}")

    doc_family_id = record.get("doc_family_id")
    if not isinstance(doc_family_id, str) or not doc_family_id.strip():
        errors.append("missing_doc_family_id")

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        errors.append("cpt_requires_single_assistant_message")
        return errors

    msg = messages[0]
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        errors.append("cpt_message_role_must_be_assistant")
        return errors

    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        errors.append("empty_assistant_content")
        return errors

    images = record.get("images")
    if images is None:
        images = []
    if not isinstance(images, list) or any(
        not isinstance(img, str) or not img.strip() for img in images
    ):
        errors.append("invalid_images_list")
        return errors

    placeholder_count = content.count(IMAGE_PLACEHOLDER)
    if placeholder_count != len(images):
        errors.append(
            f"image_placeholder_mismatch:placeholders={placeholder_count},images={len(images)}"
        )

    if len(images) > 0 and placeholder_count > 0:
        first_idx = content.find(IMAGE_PLACEHOLDER)
        prefix = content[:first_idx].strip()
        if prefix and not ALLOWED_IMAGE_PREFIX_RE.fullmatch(prefix):
            errors.append("causal_prefix_violation:text_before_first_image")

    return errors


def split_by_doc_family(
    records: list[dict[str, Any]],
    *,
    val_ratio: float = 0.02,
    seed: int = 42,
    min_val_families: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Splits records into train/val with zero overlap across `doc_family_id`."""
    family_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        family_buckets[str(rec["doc_family_id"])].append(rec)

    families = sorted(family_buckets.keys())
    if len(families) < 2 or val_ratio <= 0.0:
        return list(records), []

    rng = random.Random(seed)
    shuffled_families = list(families)
    rng.shuffle(shuffled_families)

    val_count = max(min_val_families, int(round(len(families) * val_ratio)))
    val_count = min(val_count, len(families) - 1)
    val_families = set(shuffled_families[:val_count])

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    for fam in families:
        target = val_records if fam in val_families else train_records
        target.extend(family_buckets[fam])

    train_fam_set = {str(r["doc_family_id"]) for r in train_records}
    val_fam_set = {str(r["doc_family_id"]) for r in val_records}
    overlap = train_fam_set & val_fam_set
    if overlap:
        raise AssertionError(f"doc_family_id leakage detected: {sorted(overlap)}")

    return train_records, val_records


def to_swift_pt_row(record: dict[str, Any]) -> dict[str, Any]:
    """Converts a validated record into a clean MS-SWIFT `pt` JSONL row."""
    row: dict[str, Any] = {
        "messages": [
            {
                "role": "assistant",
                "content": record["messages"][0]["content"],
            }
        ],
        "channel": record["channel"],
        "doc_family_id": record["doc_family_id"],
    }
    images = record.get("images") or []
    if images:
        row["images"] = list(images)
    for optional_key in ("sample_id", "source_kind", "is_unresolved_tail", "example_only"):
        if optional_key in record:
            row[optional_key] = record[optional_key]
    return row


def to_midtrain_row(record: dict[str, Any]) -> dict[str, Any]:
    """Formats a CPT or CoT replay record for unified Instruct-Aware Mid-Training."""
    messages = record.get("messages") or []
    if len(messages) == 1:
        row = to_swift_pt_row(record)
    else:
        row = {
            "messages": [
                {"role": str(m["role"]), "content": str(m["content"])}
                for m in messages
            ],
            "channel": record["channel"],
            "doc_family_id": record["doc_family_id"],
        }
        images = record.get("images") or []
        if images:
            row["images"] = list(images)
    ch = str(record["channel"])
    row["bucket"] = str(record.get("bucket") or bucket_for_channel(ch))
    for opt_key in (
        "sample_id",
        "source_kind",
        "is_unresolved_tail",
        "example_only",
        "entity_id",
        "domain",
        "enable_thinking",
        "task_type",
        "cot_mode",
        "is_synthetic_image",
        "synthetic_image_count",
        "doc_text_repeat",
    ):
        if opt_key in record:
            row[opt_key] = record[opt_key]
    return row


def apply_length_guard(
    records: list[dict[str, Any]],
    *,
    max_tokens: int | None = 16384,
    tokens_per_image: int = 512,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filters records exceeding `max_tokens` and logs per-channel drops for audit."""
    if max_tokens is None or max_tokens <= 0:
        return list(records), {
            "enabled": False,
            "max_tokens": max_tokens,
            "input_rows": len(records),
            "kept_rows": len(records),
            "dropped_rows": 0,
            "dropped_by_channel": {},
        }

    kept: list[dict[str, Any]] = []
    dropped_by_channel: Counter[str] = Counter()
    for rec in records:
        content = _record_text_content(rec)
        n_img = len(rec.get("images") or [])
        est_tok = estimate_record_tokens(content, n_img, tokens_per_image=tokens_per_image)
        if est_tok > max_tokens:
            dropped_by_channel[str(rec["channel"])] += 1
            continue
        kept.append(rec)

    report = {
        "enabled": True,
        "max_tokens": max_tokens,
        "input_rows": len(records),
        "kept_rows": len(kept),
        "dropped_rows": len(records) - len(kept),
        "dropped_by_channel": dict(sorted(dropped_by_channel.items())),
    }
    return kept, report


def _eligible_channel_rows(
    rows: list[dict[str, Any]],
    *,
    stage: str,
    channel: str,
) -> list[dict[str, Any]]:
    """Filters channel pool to unresolved tail items during Stage 2 text-doc cooldown."""
    if stage == "stage2_cooldown" and channel in TEXT_DOC_CHANNELS:
        unresolved = [r for r in rows if r.get("is_unresolved_tail", False)]
        if unresolved:
            return unresolved
    return rows


def _allocate_token_balanced_counts(
    records_by_channel: dict[str, list[dict[str, Any]]],
    target_weights: dict[str, float],
    total_rows: int,
    tokens_per_image: int,
    *,
    max_row_repeats: int | None = None,
) -> dict[str, int]:
    """Computes per-channel row counts so expected token shares match `target_weights`."""
    present_channels = [ch for ch, rows in records_by_channel.items() if rows]
    if not present_channels:
        return {}

    avg_tokens: dict[str, float] = {}
    for ch in present_channels:
        tok_list = [
            estimate_record_tokens(
                _record_text_content(r),
                len(r.get("images") or []),
                tokens_per_image=tokens_per_image,
            )
            for r in records_by_channel[ch]
        ]
        avg_tokens[ch] = sum(tok_list) / len(tok_list)

    raw_row_scores: dict[str, float] = {}
    for ch in present_channels:
        weight = target_weights.get(ch, 1.0 / len(present_channels))
        raw_row_scores[ch] = weight / max(1.0, avg_tokens[ch])

    score_sum = sum(raw_row_scores.values()) or 1.0
    counts: dict[str, int] = {
        ch: max(1, int(round(total_rows * (raw_row_scores[ch] / score_sum))))
        for ch in present_channels
    }

    if max_row_repeats is not None and max_row_repeats > 0:
        caps = {ch: len(records_by_channel[ch]) * max_row_repeats for ch in present_channels}
        excess = 0
        uncapped: list[str] = []
        for ch in present_channels:
            if counts[ch] > caps[ch]:
                excess += counts[ch] - caps[ch]
                counts[ch] = caps[ch]
            elif counts[ch] < caps[ch]:
                uncapped.append(ch)
        while excess > 0 and uncapped:
            uncapped_scores = sum(raw_row_scores[ch] for ch in uncapped) or 1.0
            distributed = 0
            next_uncapped: list[str] = []
            for ch in uncapped:
                share = max(1, int(round(excess * (raw_row_scores[ch] / uncapped_scores))))
                room = caps[ch] - counts[ch]
                add = min(room, share, excess - distributed)
                counts[ch] += add
                distributed += add
                if counts[ch] < caps[ch]:
                    next_uncapped.append(ch)
            if distributed == 0:
                break
            excess -= distributed
            uncapped = next_uncapped

    return counts


def _sample_channel_records(
    rows: list[dict[str, Any]],
    count: int,
    *,
    stage: str,
    tail_oversample: int,
    max_row_repeats: int | None,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Samples `count` records within an eligible channel pool, bounding per-row repeats when set."""
    if not rows or count <= 0:
        return []

    if max_row_repeats is not None and max_row_repeats > 0:
        # Stratified bounded expansion guarantees no row appears > max_row_repeats times.
        prioritized: list[dict[str, Any]] = []
        remainder: list[dict[str, Any]] = []
        for r in rows:
            target_copies = (
                tail_oversample
                if (stage == "stage2_cooldown" and r.get("is_unresolved_tail", False))
                else 1
            )
            primary = min(max_row_repeats, max(1, target_copies))
            prioritized.extend([r] * primary)
            if max_row_repeats > primary:
                remainder.extend([r] * (max_row_repeats - primary))
        rng.shuffle(prioritized)
        rng.shuffle(remainder)
        bounded_pool = (prioritized + remainder)[:count]
        rng.shuffle(bounded_pool)
        return bounded_pool

    weights = [
        float(tail_oversample)
        if (stage == "stage2_cooldown" and r.get("is_unresolved_tail", False))
        else 1.0
        for r in rows
    ]

    if count <= len(rows) and stage != "stage2_cooldown":
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        return [rows[i] for i in indices[:count]]

    sampled: list[dict[str, Any]] = []
    if count >= len(rows) and all(w == 1.0 for w in weights):
        sampled.extend(rows)
        remaining = count - len(rows)
        if remaining > 0:
            sampled.extend(rng.choices(rows, weights=weights, k=remaining))
        return sampled

    return rng.choices(rows, weights=weights, k=count)


def build_stage_mixture(
    train_records: list[dict[str, Any]],
    *,
    stage: str,
    target_weights: dict[str, float],
    target_rows: int | None = None,
    tail_oversample: int = 2,
    max_row_repeats: int | None = None,
    tokens_per_image: int = 512,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Builds a shuffled stage dataset aligned to token-level channel weights."""
    rng = random.Random(seed)
    by_channel_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in train_records:
        by_channel_raw[str(rec["channel"])].append(rec)

    by_channel: dict[str, list[dict[str, Any]]] = {
        ch: _eligible_channel_rows(rows, stage=stage, channel=ch)
        for ch, rows in by_channel_raw.items()
    }

    if target_rows is None:
        selected: list[dict[str, Any]] = []
        for ch, pool in sorted(by_channel.items()):
            for r in pool:
                repeat = (
                    tail_oversample
                    if (stage == "stage2_cooldown" and r.get("is_unresolved_tail", False))
                    else 1
                )
                if max_row_repeats is not None and max_row_repeats > 0:
                    repeat = min(repeat, max_row_repeats)
                selected.extend([r] * repeat)
        rng.shuffle(selected)
        return [to_swift_pt_row(r) for r in selected]

    counts = _allocate_token_balanced_counts(
        by_channel,
        target_weights=target_weights,
        total_rows=target_rows,
        tokens_per_image=tokens_per_image,
        max_row_repeats=max_row_repeats,
    )
    selected = []
    for ch, count in sorted(counts.items()):
        selected.extend(
            _sample_channel_records(
                by_channel[ch],
                count,
                stage=stage,
                tail_oversample=tail_oversample,
                max_row_repeats=max_row_repeats,
                rng=rng,
            )
        )
    rng.shuffle(selected)
    return [to_swift_pt_row(r) for r in selected]


def _row_identity_key(row: dict[str, Any]) -> str:
    sid = row.get("sample_id")
    if isinstance(sid, str) and sid.strip():
        return f"sid:{sid}"
    return hashlib.sha256(
        json.dumps(
            {"messages": row.get("messages"), "images": row.get("images", [])},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    tokens_per_image: int = 512,
) -> dict[str, Any]:
    """Summarizes row counts, unique counts, repeat factors, and token shares per channel."""
    ch_rows: Counter[str] = Counter()
    ch_tokens: Counter[str] = Counter()
    ch_images: Counter[str] = Counter()
    ch_unique: dict[str, set[str]] = defaultdict(set)
    unresolved_count = 0

    for r in rows:
        ch = str(r["channel"])
        content = _record_text_content(r)
        n_img = len(r.get("images") or [])
        tok = estimate_record_tokens(content, n_img, tokens_per_image=tokens_per_image)
        ch_rows[ch] += 1
        ch_tokens[ch] += tok
        ch_images[ch] += n_img
        ch_unique[ch].add(_row_identity_key(r))
        if r.get("is_unresolved_tail"):
            unresolved_count += 1

    total_tokens = sum(ch_tokens.values()) or 1
    channel_stats = {
        ch: {
            "rows": ch_rows[ch],
            "unique_rows": len(ch_unique[ch]),
            "effective_repeat_factor": round(ch_rows[ch] / max(1, len(ch_unique[ch])), 2),
            "images": ch_images[ch],
            "estimated_tokens": ch_tokens[ch],
            "token_share": round(ch_tokens[ch] / total_tokens, 4),
        }
        for ch in sorted(ch_rows.keys())
    }
    return {
        "total_rows": len(rows),
        "total_images": sum(ch_images.values()),
        "total_estimated_tokens": sum(ch_tokens.values()),
        "unresolved_tail_rows": unresolved_count,
        "channels": channel_stats,
    }


def build_demo_records() -> list[dict[str, Any]]:
    """Generates a 12-family synthetic 6-channel dataset for local smoke/validation."""
    examples_path = Path(__file__).resolve().parent / "examples" / "cpt_mm_6ch_example.jsonl"
    base_records = load_jsonl(examples_path)
    records: list[dict[str, Any]] = []
    for rec in base_records:
        records.append(rec)
        sibling = json.loads(json.dumps(rec, ensure_ascii=False))
        sibling["sample_id"] = f"{rec['sample_id']}_twin"
        sibling["doc_family_id"] = f"{rec['doc_family_id']}_twin"
        if rec["channel"] in TEXT_DOC_CHANNELS:
            sibling["is_unresolved_tail"] = False
            sibling["messages"][0]["content"] = (
                "[特征与属性线索]：示例常规词条（已掌握头部锚点），蓝白相间十字旗。\n"
                "[对应标准实体]：教学示例头部旗帜B（Demo Head Flag B）。"
            )
        records.append(sibling)
    return records


def _collect_repeat_warnings(
    stage_summaries: dict[str, dict[str, Any]],
    threshold: float = 3.0,
) -> list[str]:
    warnings: list[str] = []
    for stage_name, summary in stage_summaries.items():
        for ch, stats in summary.get("channels", {}).items():
            repeat_factor = stats.get("effective_repeat_factor", 1.0)
            if repeat_factor > threshold:
                warnings.append(
                    f"{stage_name}.{ch}: effective_repeat_factor={repeat_factor} > {threshold} "
                    f"(rows={stats['rows']}, unique_rows={stats['unique_rows']}); consider shrinking stage2 size or setting --max-row-repeats."
                )
    return warnings


def build_cpt_bundle(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    val_ratio: float = 0.02,
    stage1_rows: int | None = None,
    stage2_rows: int | None = None,
    tail_oversample: int = 2,
    max_tokens: int | None = 16384,
    max_row_repeats: int | None = None,
    tokens_per_image: int = 512,
    seed: int = 42,
    allow_custom_channels: bool = False,
) -> dict[str, Any]:
    """Validates records, enforces length guard, performs zero-leakage split, builds WSD stages, and writes manifest."""
    if not records:
        raise ValueError("input records must not be empty")

    for idx, rec in enumerate(records):
        errs = validate_cpt_record(rec, allow_custom_channels=allow_custom_channels)
        if errs:
            sid = rec.get("sample_id", f"index_{idx}")
            raise ValueError(f"Record {sid} failed validation: {errs}")

    guarded_records, length_guard_report = apply_length_guard(
        records,
        max_tokens=max_tokens,
        tokens_per_image=tokens_per_image,
    )
    if not guarded_records:
        raise ValueError("All records exceeded --max-tokens length guard.")

    train_records, val_records = split_by_doc_family(
        guarded_records,
        val_ratio=val_ratio,
        seed=seed,
    )

    stage1_out = build_stage_mixture(
        train_records,
        stage="stage1_stable",
        target_weights=STAGE1_TOKEN_WEIGHTS,
        target_rows=stage1_rows,
        tail_oversample=1,
        max_row_repeats=max_row_repeats,
        tokens_per_image=tokens_per_image,
        seed=seed,
    )
    stage2_out = build_stage_mixture(
        train_records,
        stage="stage2_cooldown",
        target_weights=STAGE2_TOKEN_WEIGHTS,
        target_rows=stage2_rows,
        tail_oversample=tail_oversample,
        max_row_repeats=max_row_repeats,
        tokens_per_image=tokens_per_image,
        seed=seed + 1,
    )
    val_out = [to_swift_pt_row(r) for r in val_records]

    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = output_dir / "stage1_train.jsonl"
    stage2_path = output_dir / "stage2_cooldown.jsonl"
    val_path = output_dir / "val.jsonl"
    manifest_path = output_dir / "manifest.json"

    write_jsonl(stage1_path, stage1_out)
    write_jsonl(stage2_path, stage2_out)
    write_jsonl(val_path, val_out)

    s1_summary = summarize_rows(stage1_out, tokens_per_image=tokens_per_image)
    s2_summary = summarize_rows(stage2_out, tokens_per_image=tokens_per_image)
    val_summary = summarize_rows(val_out, tokens_per_image=tokens_per_image)
    warnings = _collect_repeat_warnings({"stage1_train": s1_summary, "stage2_cooldown": s2_summary})

    manifest = {
        "schema": "ms_swift_pt_mm_6ch_v2",
        "seed": seed,
        "val_ratio": val_ratio,
        "tail_oversample_stage2": tail_oversample,
        "max_row_repeats": max_row_repeats,
        "tokens_per_image_estimate": tokens_per_image,
        "length_guard": length_guard_report,
        "doc_family_overlap": len(
            {str(r["doc_family_id"]) for r in train_records}
            & {str(r["doc_family_id"]) for r in val_records}
        ),
        "warnings": warnings,
        "artifacts": {
            "stage1_train": {
                "path": str(stage1_path),
                "sha256": sha256_file(stage1_path),
                "summary": s1_summary,
            },
            "stage2_cooldown": {
                "path": str(stage2_path),
                "sha256": sha256_file(stage2_path),
                "summary": s2_summary,
            },
            "val": {
                "path": str(val_path),
                "sha256": sha256_file(val_path),
                "summary": val_summary,
            },
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def sanitize_qa_for_cpt(record: dict[str, Any]) -> dict[str, Any]:
    """Converts multi-turn ChatML / `<think>` QA into single-assistant CPT prose (zero role/tag pollution)."""
    messages = record.get("messages") or []
    user_txt = ""
    assistant_txt = ""
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = str(m.get("content") or "")
        if role == "user":
            user_txt = content
        elif role == "assistant":
            assistant_txt = content

    question = ""
    if "问题：" in user_txt:
        question = user_txt.split("问题：", 1)[1].strip()
    elif user_txt:
        question = re.sub(r"【本次审核给定规则】[\s\S]*?(?=\n\n|$)", "", user_txt).strip()

    cleaned_ans = re.sub(r"</?think>", "", assistant_txt)
    cleaned_ans = re.sub(r"\[(?:S|R)\d+\]", "", cleaned_ans).strip()

    parts: list[str] = []
    if question:
        parts.append(f"[领域知识命题] {question}")
    if cleaned_ans:
        parts.append(f"[推演与规范结论] {cleaned_ans}")
    prose = "\n".join(parts) or cleaned_ans or user_txt or "领域知识条目"

    out = dict(record)
    out["channel"] = "ch_qa_knowledge_prose"
    out["bucket"] = "qa_prose_anti_collapse"
    out["is_qa_prose"] = True
    out.pop("enable_thinking", None)
    out["messages"] = [{"role": "assistant", "content": prose}]
    return out


def _row_synth_image_count(row: dict[str, Any]) -> int:
    imgs = row.get("images") or []
    if not imgs:
        return 0
    if "synthetic_image_count" in row:
        return max(0, int(row["synthetic_image_count"]))
    if row.get("is_synthetic_image") or row.get("channel") in SYNTHETIC_CHANNELS:
        return len(imgs)
    return 0


def compute_synthetic_image_share(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_images = sum(len(r.get("images") or []) for r in rows)
    synth_images = sum(_row_synth_image_count(r) for r in rows)
    share = round(synth_images / total_images, 4) if total_images > 0 else 0.0
    return {
        "synth_images": synth_images,
        "total_images": total_images,
        "real_images": max(0, total_images - synth_images),
        "synth_image_share": share,
    }


def _is_qa_prose_row(row: dict[str, Any]) -> bool:
    return bool(row.get("is_qa_prose")) or str(row.get("channel")) in {
        "ch_qa_knowledge_prose",
        "wiki_rag_qa_cot",
        "wiki_rag_qa_direct",
    }


def extract_families_from_paths(paths: list[Path]) -> set[str]:
    families: set[str] = set()
    for p in paths:
        if not p.exists():
            continue
        for row in load_jsonl(p):
            fam = str(row.get("doc_family_id") or "").strip()
            if fam:
                families.add(fam)
    return families


def _summarize_buckets(rows: list[dict[str, Any]], *, tokens_per_image: int) -> dict[str, Any]:
    b_rows: Counter[str] = Counter()
    b_tokens: Counter[str] = Counter()
    b_images: Counter[str] = Counter()
    for r in rows:
        b = str(r.get("bucket") or bucket_for_channel(str(r["channel"])))
        tok = estimate_record_tokens(
            _record_text_content(r),
            len(r.get("images") or []),
            tokens_per_image=tokens_per_image,
        )
        b_rows[b] += 1
        b_tokens[b] += tok
        b_images[b] += len(r.get("images") or [])
    tot_tok = sum(b_tokens.values()) or 1
    return {
        b: {
            "rows": b_rows[b],
            "images": b_images[b],
            "estimated_tokens": b_tokens[b],
            "token_share": round(b_tokens[b] / tot_tok, 4),
        }
        for b in sorted(b_rows.keys())
    }


def build_midtrain_bundle(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    val_ratio: float = 0.05,
    total_rows: int | None = None,
    max_tokens: int = 8192,
    max_row_repeats: int = 2,
    max_synth_image_share: float = 0.25,
    max_qa_prose_share: float = 0.15,
    allow_raw_chatml_in_cpt: bool = False,
    excluded_eval_families: set[str] | None = None,
    tokens_per_image: int = 512,
    seed: int = 42,
    allow_custom_channels: bool = False,
) -> dict[str, Any]:
    """Builds a single-stage Instruct-Aware Mid-Training bundle with anti-degradation & QA-risk guards."""
    if not records:
        raise ValueError("input records must not be empty")

    excluded_set = excluded_eval_families or set()
    sanitized_count = 0
    eval_dropped = 0
    cleaned_records: list[dict[str, Any]] = []

    for idx, rec in enumerate(records):
        errs = validate_midtrain_record(rec, allow_custom_channels=allow_custom_channels)
        if errs:
            sid = rec.get("sample_id", f"index_{idx}")
            raise ValueError(f"Record {sid} failed midtrain validation: {errs}")

        item = dict(rec)
        needs_sanitize = (
            len(item.get("messages") or []) > 1
            or item.get("channel") in {"wiki_rag_qa_cot", "wiki_rag_qa_direct"}
            or "<think>" in _record_text_content(item).lower()
        )
        if needs_sanitize and not allow_raw_chatml_in_cpt:
            item = sanitize_qa_for_cpt(item)
            sanitized_count += 1

        if _is_qa_prose_row(item) and str(item.get("doc_family_id")) in excluded_set:
            eval_dropped += 1
            continue

        item["bucket"] = bucket_for_channel(str(item["channel"]))
        cleaned_records.append(item)

    if not cleaned_records:
        raise ValueError("No records remained after QA-in-CPT eval exclusion / sanitization.")

    guarded_records, length_guard_report = apply_length_guard(
        cleaned_records,
        max_tokens=max_tokens,
        tokens_per_image=tokens_per_image,
    )
    if not guarded_records:
        raise ValueError("All records exceeded --max-tokens length guard.")

    train_records, val_records = split_by_doc_family(
        guarded_records,
        val_ratio=val_ratio,
        seed=seed,
    )

    rng = random.Random(seed)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in train_records:
        by_bucket[bucket_for_channel(str(r["channel"]))].append(r)

    present_buckets = [b for b, pool in by_bucket.items() if pool]
    # Allocate counts across present buckets capping QA prose share to <= max_qa_prose_share
    eff_weights: dict[str, float] = {}
    for b in present_buckets:
        w = MIDTRAIN_BUCKET_WEIGHTS.get(b, 0.25)
        if b == "qa_prose_anti_collapse":
            w = min(w, max_qa_prose_share)
        eff_weights[b] = w

    target_total = total_rows or max(len(train_records), len(present_buckets) * 4)
    avg_bucket_tok = {
        b: (
            sum(
                estimate_record_tokens(
                    _record_text_content(r),
                    len(r.get("images") or []),
                    tokens_per_image=tokens_per_image,
                )
                for r in by_bucket[b]
            )
            / len(by_bucket[b])
        )
        for b in present_buckets
    }
    raw_scores = {b: eff_weights[b] / max(1.0, avg_bucket_tok[b]) for b in present_buckets}
    score_sum = sum(raw_scores.values()) or 1.0

    selected: list[dict[str, Any]] = []
    copy_counts: Counter[str] = Counter()

    for b in sorted(present_buckets):
        pool = list(by_bucket[b])
        rng.shuffle(pool)
        desired = max(1, int(round(target_total * (raw_scores[b] / score_sum))))
        cap = len(pool) * max(1, max_row_repeats)
        desired = min(desired, cap)
        # Prioritize non-synthetic & tail/domain diversity when filling up to max_row_repeats
        expanded: list[dict[str, Any]] = []
        for rep in range(max(1, max_row_repeats)):
            for r in pool:
                if len(expanded) >= desired:
                    break
                expanded.append(r)
        for r in expanded:
            selected.append(r)
            copy_counts[_row_identity_key(r)] += 1

    # 1. Enforce QA prose token share <= max_qa_prose_share
    qa_cap_applied = False
    def _calc_qa_share(rows_list: list[dict[str, Any]]) -> float:
        qa_tok = sum(
            estimate_record_tokens(_record_text_content(r), len(r.get("images") or []), tokens_per_image=tokens_per_image)
            for r in rows_list
            if _is_qa_prose_row(r)
        )
        tot_tok = sum(
            estimate_record_tokens(_record_text_content(r), len(r.get("images") or []), tokens_per_image=tokens_per_image)
            for r in rows_list
        )
        return (qa_tok / tot_tok) if tot_tok > 0 else 0.0

    non_qa_pool = [r for r in train_records if not _is_qa_prose_row(r)]
    rng.shuffle(non_qa_pool)
    while _calc_qa_share(selected) > max_qa_prose_share:
        added_real = False
        for cand in non_qa_pool:
            cid = _row_identity_key(cand)
            if copy_counts[cid] < max_row_repeats:
                selected.append(cand)
                copy_counts[cid] += 1
                added_real = True
                break
        if _calc_qa_share(selected) <= max_qa_prose_share:
            break
        if not added_real:
            qa_indices = [
                i for i, r in enumerate(selected) if _is_qa_prose_row(r)
            ]
            if len(qa_indices) <= 0:
                break
            # Drop largest token QA prose item first
            drop_idx = max(
                qa_indices,
                key=lambda idx: estimate_record_tokens(
                    _record_text_content(selected[idx]), len(selected[idx].get("images") or []), tokens_per_image=tokens_per_image
                ),
            )
            popped = selected.pop(drop_idx)
            copy_counts[_row_identity_key(popped)] -= 1
            qa_cap_applied = True

    # 2. Enforce Synthetic Image Share <= max_synth_image_share
    synth_cap_applied = False
    real_img_pool = [
        r for r in train_records if (len(r.get("images") or []) > 0 and _row_synth_image_count(r) == 0)
    ]
    rng.shuffle(real_img_pool)
    while True:
        img_stats = compute_synthetic_image_share(selected)
        if img_stats["total_images"] == 0 or img_stats["synth_image_share"] <= max_synth_image_share:
            break
        # First try adding real images up to max_row_repeats before dropping synthetic rows
        added_real_img = False
        for cand in real_img_pool:
            cid = _row_identity_key(cand)
            if copy_counts[cid] < max_row_repeats:
                selected.append(cand)
                copy_counts[cid] += 1
                added_real_img = True
                break
        if added_real_img:
            continue

        synth_indices = [i for i, r in enumerate(selected) if _row_synth_image_count(r) > 0]
        if not synth_indices:
            break
        bucket_counts_now = Counter(bucket_for_channel(str(r["channel"])) for r in selected)
        droppable = [
            i for i in synth_indices if bucket_counts_now[bucket_for_channel(str(selected[i]["channel"]))] > 1
        ] or synth_indices
        drop_idx = max(droppable, key=lambda i: _row_synth_image_count(selected[i]))
        popped = selected.pop(drop_idx)
        copy_counts[_row_identity_key(popped)] -= 1
        synth_cap_applied = True

    rng.shuffle(selected)
    train_out = [to_midtrain_row(r) for r in selected]
    val_out = [to_midtrain_row(r) for r in val_records]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "midtrain_train.jsonl"
    val_path = output_dir / "midtrain_val.jsonl"
    manifest_path = output_dir / "midtrain_manifest.json"
    write_jsonl(train_path, train_out)
    write_jsonl(val_path, val_out)

    final_img_audit = compute_synthetic_image_share(train_out)
    final_img_audit["cap_applied"] = synth_cap_applied
    final_img_audit["max_synth_image_share"] = max_synth_image_share

    raw_chatml_count = sum(
        1
        for r in train_out
        if len(r.get("messages") or []) > 1 or "<think>" in _record_text_content(r).lower()
    )
    qa_prose_eval_overlap = len(
        {str(r["doc_family_id"]) for r in (train_out + val_out) if _is_qa_prose_row(r)}
        & excluded_set
    )
    doc_text_repeats = [
        int(r.get("doc_text_repeat", 1))
        for r in train_out
        if r.get("channel") in {"ch_v2_multi_img_doc", "ch_sports_expanded"}
    ]

    qa_cpt_risk_audit = {
        "allow_raw_chatml_in_cpt": allow_raw_chatml_in_cpt,
        "raw_chatml_sanitized_to_prose": sanitized_count,
        "raw_chatml_in_cpt_count": raw_chatml_count,
        "qa_prose_token_share": round(_calc_qa_share(train_out), 4),
        "max_qa_prose_share": max_qa_prose_share,
        "qa_prose_cap_applied": qa_cap_applied,
        "eval_family_dropped": eval_dropped,
        "eval_family_overlap": qa_prose_eval_overlap,
        "max_doc_text_repeat": max(doc_text_repeats, default=1),
    }

    train_summary = summarize_rows(train_out, tokens_per_image=tokens_per_image)
    val_summary = summarize_rows(val_out, tokens_per_image=tokens_per_image)
    bucket_summary = _summarize_buckets(train_out, tokens_per_image=tokens_per_image)

    manifest = {
        "schema": "ms_swift_midtrain_v2",
        "seed": seed,
        "val_ratio": val_ratio,
        "max_tokens": max_tokens,
        "max_row_repeats": max_row_repeats,
        "tokens_per_image_estimate": tokens_per_image,
        "length_guard": length_guard_report,
        "doc_family_overlap": len(
            {str(r["doc_family_id"]) for r in train_out}
            & {str(r["doc_family_id"]) for r in val_out}
        ),
        "bucket_summary": bucket_summary,
        "synthetic_image_audit": final_img_audit,
        "qa_cpt_risk_audit": qa_cpt_risk_audit,
        "artifacts": {
            "midtrain_train": {
                "path": str(train_path),
                "sha256": sha256_file(train_path),
                "summary": train_summary,
            },
            "midtrain_val": {
                "path": str(val_path),
                "sha256": sha256_file(val_path),
                "summary": val_summary,
            },
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        default=[],
        help="Input JSONL files conforming to CPT / Mid-Training schema.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Build a local synthetic demo bundle.",
    )
    parser.add_argument(
        "--mode",
        choices=["cpt_2stage", "midtrain"],
        default="cpt_2stage",
        help="`cpt_2stage` (legacy stage1+stage2) or `midtrain` (single-stage 4-bucket bundle).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for JSONL bundle and manifest JSON.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--stage1-rows", type=int, default=None)
    parser.add_argument("--stage2-rows", type=int, default=None)
    parser.add_argument("--total-rows", type=int, default=None)
    parser.add_argument("--tail-oversample", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-row-repeats", type=int, default=None)
    parser.add_argument("--max-synth-image-share", type=float, default=0.25)
    parser.add_argument("--max-qa-prose-share", type=float, default=0.15)
    parser.add_argument("--exclude-families-from", nargs="*", type=Path, default=[])
    parser.add_argument("--allow-raw-chatml-in-cpt", action="store_true")
    parser.add_argument("--tokens-per-image", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-custom-channels", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.demo:
        records = build_demo_records()
        val_ratio = args.val_ratio if args.val_ratio >= 0.15 else 0.17
        stage1_rows = args.stage1_rows if args.stage1_rows is not None else 24
        stage2_rows = args.stage2_rows if args.stage2_rows is not None else 16
    elif args.inputs:
        records = []
        for path in args.inputs:
            if path.exists():
                records.extend(load_jsonl(path))
        val_ratio = args.val_ratio
        stage1_rows = args.stage1_rows
        stage2_rows = args.stage2_rows
    else:
        print("Specify --demo or at least one file via --inputs.", file=sys.stderr)
        return 2

    if args.mode == "midtrain":
        if args.demo:
            records.extend(
                [
                    {
                        "sample_id": "demo_qa_prose_01",
                        "doc_family_id": "fam_demo_prose_01",
                        "channel": "ch_qa_knowledge_prose",
                        "is_qa_prose": True,
                        "is_unresolved_tail": False,
                        "images": [],
                        "messages": [
                            {
                                "role": "assistant",
                                "content": "[领域知识命题] 示例电鼠P面部识别要素有哪些？\n[稳定结论锚点] 黄色短毛躯干搭配红色圆形电气囊、黑色耳尖与闪电状锯齿折尾。",
                            }
                        ],
                    },
                    {
                        "sample_id": "demo_qa_prose_02",
                        "doc_family_id": "fam_demo_prose_02",
                        "channel": "ch_qa_knowledge_prose",
                        "is_qa_prose": True,
                        "is_unresolved_tail": True,
                        "images": [],
                        "messages": [
                            {
                                "role": "assistant",
                                "content": "[领域知识命题] 无标高奢B款马鞍包设计要素侵权边界为何？\n[稳定结论锚点] 需同时具备不对称弧面盖体、马蹄形五金锁扣与双轨斜向工程缝线。",
                            }
                        ],
                    },
                ]
            )
        excluded_eval = extract_families_from_paths(args.exclude_families_from)
        manifest = build_midtrain_bundle(
            records,
            args.output_dir,
            val_ratio=val_ratio,
            total_rows=args.total_rows,
            max_tokens=8192 if args.max_tokens == 16384 and not args.demo else args.max_tokens,
            max_row_repeats=args.max_row_repeats if args.max_row_repeats is not None else 2,
            max_synth_image_share=args.max_synth_image_share,
            max_qa_prose_share=args.max_qa_prose_share,
            allow_raw_chatml_in_cpt=args.allow_raw_chatml_in_cpt,
            excluded_eval_families=excluded_eval,
            tokens_per_image=args.tokens_per_image,
            seed=args.seed,
            allow_custom_channels=args.allow_custom_channels,
        )
    else:
        manifest = build_cpt_bundle(
            records,
            args.output_dir,
            val_ratio=val_ratio,
            stage1_rows=stage1_rows,
            stage2_rows=stage2_rows,
            tail_oversample=args.tail_oversample,
            max_tokens=args.max_tokens,
            max_row_repeats=args.max_row_repeats,
            tokens_per_image=args.tokens_per_image,
            seed=args.seed,
            allow_custom_channels=args.allow_custom_channels,
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
