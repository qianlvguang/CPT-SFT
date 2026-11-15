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
    "ch1_wiki_text_stub",
    "ch2_std_logo_flag",
    "ch3_qwen_img_synth",
    "ch4_audit_crawl_cf",
    "ch5_flash_rag_raft",
    "ch6_general_replay",
)

STAGE1_TOKEN_WEIGHTS: dict[str, float] = {
    "ch1_wiki_text_stub": 0.25,
    "ch2_std_logo_flag": 0.15,
    "ch3_qwen_img_synth": 0.15,
    "ch4_audit_crawl_cf": 0.15,
    "ch5_flash_rag_raft": 0.15,
    "ch6_general_replay": 0.15,
}

STAGE2_TOKEN_WEIGHTS: dict[str, float] = {
    "ch1_wiki_text_stub": 0.15,
    "ch2_std_logo_flag": 0.10,
    "ch3_qwen_img_synth": 0.15,
    "ch4_audit_crawl_cf": 0.20,
    "ch5_flash_rag_raft": 0.20,
    "ch6_general_replay": 0.20,
}

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


def _allocate_token_balanced_counts(
    records_by_channel: dict[str, list[dict[str, Any]]],
    target_weights: dict[str, float],
    total_rows: int,
    tokens_per_image: int,
) -> dict[str, int]:
    """Computes per-channel row counts so expected token shares match `target_weights`."""
    present_channels = [ch for ch, rows in records_by_channel.items() if rows]
    if not present_channels:
        return {}

    avg_tokens: dict[str, float] = {}
    for ch in present_channels:
        tok_list = [
            estimate_record_tokens(
                r["messages"][0]["content"],
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
    counts: dict[str, int] = {}
    for ch in present_channels:
        counts[ch] = max(1, int(round(total_rows * (raw_row_scores[ch] / score_sum))))
    return counts


def _sample_channel_records(
    rows: list[dict[str, Any]],
    count: int,
    *,
    stage: str,
    channel: str,
    tail_oversample: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Samples `count` records within one channel using Stage 2 tail weighting."""
    if stage == "stage2_cooldown" and channel == "ch1_wiki_text_stub":
        unresolved = [r for r in rows if r.get("is_unresolved_tail", False)]
        if unresolved:
            rows = unresolved

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
    tokens_per_image: int = 512,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Builds a shuffled stage dataset aligned to token-level channel weights."""
    rng = random.Random(seed)
    by_channel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in train_records:
        by_channel[str(rec["channel"])].append(rec)

    if target_rows is None:
        selected: list[dict[str, Any]] = []
        for ch, rows in sorted(by_channel.items()):
            if stage == "stage2_cooldown" and ch == "ch1_wiki_text_stub":
                unresolved = [r for r in rows if r.get("is_unresolved_tail", False)]
                pool = unresolved if unresolved else rows
            else:
                pool = rows
            for r in pool:
                repeat = (
                    tail_oversample
                    if (stage == "stage2_cooldown" and r.get("is_unresolved_tail", False))
                    else 1
                )
                selected.extend([r] * repeat)
        rng.shuffle(selected)
        return [to_swift_pt_row(r) for r in selected]

    counts = _allocate_token_balanced_counts(
        by_channel,
        target_weights=target_weights,
        total_rows=target_rows,
        tokens_per_image=tokens_per_image,
    )
    selected = []
    for ch, count in sorted(counts.items()):
        selected.extend(
            _sample_channel_records(
                by_channel[ch],
                count,
                stage=stage,
                channel=ch,
                tail_oversample=tail_oversample,
                rng=rng,
            )
        )
    rng.shuffle(selected)
    return [to_swift_pt_row(r) for r in selected]


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    tokens_per_image: int = 512,
) -> dict[str, Any]:
    """Summarizes row counts, image counts, and estimated token share per channel."""
    ch_rows: Counter[str] = Counter()
    ch_tokens: Counter[str] = Counter()
    ch_images: Counter[str] = Counter()
    unresolved_count = 0

    for r in rows:
        ch = str(r["channel"])
        content = r["messages"][0]["content"]
        n_img = len(r.get("images") or [])
        tok = estimate_record_tokens(content, n_img, tokens_per_image=tokens_per_image)
        ch_rows[ch] += 1
        ch_tokens[ch] += tok
        ch_images[ch] += n_img
        if r.get("is_unresolved_tail"):
            unresolved_count += 1

    total_tokens = sum(ch_tokens.values()) or 1
    channel_stats = {
        ch: {
            "rows": ch_rows[ch],
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
        if rec["channel"] == "ch1_wiki_text_stub":
            sibling["is_unresolved_tail"] = False
            sibling["messages"][0]["content"] = (
                "[特征与属性线索]：示例常规词条（已掌握头部锚点），蓝白相间十字旗。\n"
                "[对应标准实体]：教学示例头部旗帜B（Demo Head Flag B）。"
            )
        records.append(sibling)
    return records


def build_cpt_bundle(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    val_ratio: float = 0.02,
    stage1_rows: int | None = None,
    stage2_rows: int | None = None,
    tail_oversample: int = 2,
    tokens_per_image: int = 512,
    seed: int = 42,
    allow_custom_channels: bool = False,
) -> dict[str, Any]:
    """Validates records, performs zero-leakage split, builds WSD stages, and writes manifest."""
    if not records:
        raise ValueError("input records must not be empty")

    for idx, rec in enumerate(records):
        errs = validate_cpt_record(rec, allow_custom_channels=allow_custom_channels)
        if errs:
            sid = rec.get("sample_id", f"index_{idx}")
            raise ValueError(f"Record {sid} failed validation: {errs}")

    train_records, val_records = split_by_doc_family(
        records,
        val_ratio=val_ratio,
        seed=seed,
    )

    stage1_out = build_stage_mixture(
        train_records,
        stage="stage1_stable",
        target_weights=STAGE1_TOKEN_WEIGHTS,
        target_rows=stage1_rows,
        tail_oversample=1,
        tokens_per_image=tokens_per_image,
        seed=seed,
    )
    stage2_out = build_stage_mixture(
        train_records,
        stage="stage2_cooldown",
        target_weights=STAGE2_TOKEN_WEIGHTS,
        target_rows=stage2_rows,
        tail_oversample=tail_oversample,
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

    manifest = {
        "schema": "ms_swift_pt_mm_6ch_v1",
        "seed": seed,
        "val_ratio": val_ratio,
        "tail_oversample_stage2": tail_oversample,
        "tokens_per_image_estimate": tokens_per_image,
        "doc_family_overlap": len(
            {str(r["doc_family_id"]) for r in train_records}
            & {str(r["doc_family_id"]) for r in val_records}
        ),
        "artifacts": {
            "stage1_train": {
                "path": str(stage1_path),
                "sha256": sha256_file(stage1_path),
                "summary": summarize_rows(stage1_out, tokens_per_image=tokens_per_image),
            },
            "stage2_cooldown": {
                "path": str(stage2_path),
                "sha256": sha256_file(stage2_path),
                "summary": summarize_rows(stage2_out, tokens_per_image=tokens_per_image),
            },
            "val": {
                "path": str(val_path),
                "sha256": sha256_file(val_path),
                "summary": summarize_rows(val_out, tokens_per_image=tokens_per_image),
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
        help="Input JSONL files conforming to the 6-channel CPT schema.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Build a local synthetic 6-channel demo bundle from examples/cpt_mm_6ch_example.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for stage1_train.jsonl, stage2_cooldown.jsonl, val.jsonl, and manifest.json.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--stage1-rows", type=int, default=None)
    parser.add_argument("--stage2-rows", type=int, default=None)
    parser.add_argument("--tail-oversample", type=int, default=2)
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
            records.extend(load_jsonl(path))
        val_ratio = args.val_ratio
        stage1_rows = args.stage1_rows
        stage2_rows = args.stage2_rows
    else:
        print("Specify --demo or at least one file via --inputs.", file=sys.stderr)
        return 2

    manifest = build_cpt_bundle(
        records,
        args.output_dir,
        val_ratio=val_ratio,
        stage1_rows=stage1_rows,
        stage2_rows=stage2_rows,
        tail_oversample=args.tail_oversample,
        tokens_per_image=args.tokens_per_image,
        seed=args.seed,
        allow_custom_channels=args.allow_custom_channels,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
