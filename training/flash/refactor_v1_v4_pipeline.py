#!/usr/bin/env python3
"""Refactored Unified V1-V4 -> CPT & QA-SFT Pipeline Router.

Bridges legacy/existing V1-V4 (and optional V5/V6) multimodal SFT pools into
the new B300 Multimodal Wiki+RAG CPT and Unified QA-SFT architecture:
  1. Enforces strict ternary-key uniqueness `(variant, sample_id, cate_id, images)`
     so multi-image V4 records are never dropped, and blocks mini/medium/final
     cross-pool contamination + V4 final `enable_thinking != false`.
  2. Routes `V1` / `V3` / `V4` (excluding `final` pools by default, and excluding
     `V2` ordinary captions by default to prevent Cambrian-1 dilution) into
     single-assistant causal-image-prefix CPT records (`ch6_general_replay` or
     `ch5_flash_rag_raft` when RAG evidence is attached).
  3. Rebalances V1-V4 + new Wiki/RAG/Montessori QA into a token-balanced Unified
     SFT bundle (preserving `channel=V1..V4` + `wiki_rag_qa` for `enable_channel_loss`)
     with zero `doc_family_id` leakage and excluding `V5` replay from `val`.
  4. Exports hard-cluster image seed candidates from V1/V3 for Qwen-Image-Edit
     background variation & counterfactual erasure (`ch4_audit_crawl_cf`).
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

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import build_mm_wiki_cpt as cpt_builder  # noqa: E402

DEFAULT_SFT_TOKEN_WEIGHTS: dict[str, float] = {
    "V1": 0.30,
    "V2": 0.05,
    "V3": 0.15,
    "V4": 0.20,
    "V5": 0.05,
    "V6": 0.05,
    "wiki_rag_qa": 0.20,
    "wiki_rag_qa_cot": 0.16,
    "wiki_rag_qa_direct": 0.04,
}

VALID_SFT_ARMS = ("unified", "ood_only", "ood_plus_qa_direct", "ood_plus_qa_cot")
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

DEFAULT_HARD_EDIT_CATES: tuple[str, ...] = (
    "brand_logo",
    "flag_emblem",
    "design_element",
    "mascot_ip",
    "misleading",
    "flag-hard",
    "apparel",
    "ip-brand",
)

IMAGE_TAG_RE = re.compile(r"<image>\s*")


def load_records_auto(path: Path) -> list[dict[str, Any]]:
    """Loads JSONL natively, or Parquet if `pyarrow` is installed on the host."""
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                f"Reading Parquet file {path} requires pyarrow in the target environment."
            ) from exc
        table = pq.read_table(path)
        return table.to_pylist()
    return cpt_builder.load_jsonl(path)


def _infer_pool(record: dict[str, Any], source_path: Path | None = None) -> str:
    pool = record.get("pool")
    if isinstance(pool, str) and pool.strip():
        return pool.strip().lower()
    if source_path is not None:
        name = source_path.name.lower()
        for candidate in ("mini", "medium", "final"):
            if candidate in name:
                return candidate
    return "unspecified"


def _derive_doc_family_id(record: dict[str, Any]) -> str:
    fam = record.get("doc_family_id")
    if isinstance(fam, str) and fam.strip():
        return fam.strip()
    source_docs = record.get("source_doc_ids")
    if isinstance(source_docs, list) and source_docs and str(source_docs[0]).strip():
        return f"src_{source_docs[0]}"
    variant = record.get("variant")
    sid = record.get("sample_id")
    if variant is not None and sid is not None:
        return f"v{variant}_{sid}"
    if sid is not None and str(sid).strip():
        return str(sid).strip()
    digest = hashlib.sha256(
        json.dumps(record.get("messages", []), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"auto_fam_{digest}"


def validate_and_dedup_v1_v4(
    records_with_paths: list[tuple[dict[str, Any], Path | None]],
    *,
    dedup_same_pool: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validates schema/V4 thinking rules and enforces ternary-key & cross-pool isolation."""
    seen_ternary: dict[tuple[Any, ...], str] = {}
    cleaned: list[dict[str, Any]] = []
    skipped_same_pool_dups = 0
    variant_counts: Counter[str] = Counter()

    for idx, (raw, src_path) in enumerate(records_with_paths):
        rec = dict(raw)
        variant = rec.get("variant")
        sample_id = rec.get("sample_id")
        cate_id = rec.get("cate_id")
        messages = rec.get("messages")
        images = rec.get("images")

        if variant not in (1, 2, 3, 4, 5, 6):
            raise ValueError(f"Record #{idx}: invalid variant={variant!r}")
        if sample_id is None or not str(sample_id).strip():
            raise ValueError(f"Record #{idx}: missing sample_id")
        if cate_id is None or not str(cate_id).strip():
            raise ValueError(f"Record #{idx}: missing cate_id")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Record {sample_id}: messages must be a non-empty list")
        if messages[-1].get("role") != "assistant" or not str(messages[-1].get("content", "")).strip():
            raise ValueError(f"Record {sample_id}: final message must be non-empty assistant response")
        if images is None:
            images = []
        if not isinstance(images, list):
            raise ValueError(f"Record {sample_id}: images must be a list")

        pool = _infer_pool(rec, src_path)
        rec["pool"] = pool
        rec["doc_family_id"] = _derive_doc_family_id(rec)

        if variant == 4 and pool == "final":
            if "enable_thinking" not in rec or rec["enable_thinking"] is not False:
                raise ValueError(
                    f"V4 final record {sample_id} must explicitly set enable_thinking=false"
                )

        ternary_key = (int(variant), str(sample_id), str(cate_id), tuple(str(x) for x in images))
        prior_pool = seen_ternary.get(ternary_key)
        if prior_pool is not None:
            if prior_pool != pool:
                raise ValueError(
                    f"Cross-pool overlap detected for V{variant} key={ternary_key}: {prior_pool} vs {pool}"
                )
            if dedup_same_pool:
                skipped_same_pool_dups += 1
                continue
            raise ValueError(f"Duplicate ternary key in pool={pool}: {ternary_key}")

        seen_ternary[ternary_key] = pool
        rec["images"] = [str(x) for x in images]
        cleaned.append(rec)
        variant_counts[f"V{variant}"] += 1

    report = {
        "input_rows": len(records_with_paths),
        "validated_rows": len(cleaned),
        "skipped_same_pool_dups": skipped_same_pool_dups,
        "variant_counts": dict(sorted(variant_counts.items())),
    }
    return cleaned, report


def _format_image_prefix(num_images: int) -> str:
    if num_images <= 0:
        return ""
    if num_images == 1:
        return "<image>\n"
    return "\n".join(f"图{i}：<image>" for i in range(1, num_images + 1)) + "\n"


def _strip_image_tags(text: str) -> str:
    return IMAGE_TAG_RE.sub("", text).strip()


def convert_record_to_cpt(
    record: dict[str, Any],
    *,
    rag_evidence_map: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Converts a V1/V3/V4 SFT dialog record into a causal-prefix-valid CPT `pt` row."""
    variant = int(record["variant"])
    images = list(record.get("images") or [])
    messages = record["messages"]

    user_texts = [
        _strip_image_tags(str(m.get("content", "")))
        for m in messages
        if m.get("role") == "user"
    ]
    assistant_texts = [
        _strip_image_tags(str(m.get("content", "")))
        for m in messages
        if m.get("role") == "assistant"
    ]
    core_answer = "\n".join(t for t in assistant_texts if t).strip()
    user_cue = user_texts[0].splitlines()[0].strip() if user_texts and user_texts[0] else ""

    prefix = _format_image_prefix(len(images))
    sid = str(record["sample_id"])
    raw_evidence = record.get("rag_evidence")
    if not raw_evidence and rag_evidence_map is not None:
        raw_evidence = rag_evidence_map.get(sid)

    if variant == 4 and raw_evidence:
        ev_list = [str(raw_evidence)] if isinstance(raw_evidence, str) else [str(x) for x in raw_evidence]
        ev_block = "\n".join(ev_list)
        content = (
            f"{prefix}"
            f"[视觉观察与检索提名]：{user_cue or '结合图元外观特征检索知识依据'}\n"
            f"[检索证据]\n{ev_block}\n"
            f"[综合知识描述]：{core_answer}"
        )
        channel = "ch5_flash_rag_raft"
        source_kind = "v4_upgraded_rag_raft"
    else:
        body = f"[任务观察]：{core_answer}" if variant == 1 else core_answer
        content = f"{prefix}{body}"
        channel = "ch6_general_replay"
        source_kind = f"v{variant}_cpt_replay"

    cpt_row: dict[str, Any] = {
        "sample_id": f"cpt_{sid}_{record['cate_id']}",
        "doc_family_id": record["doc_family_id"],
        "channel": channel,
        "source_kind": source_kind,
        "is_unresolved_tail": False,
        "images": images,
        "messages": [{"role": "assistant", "content": content}],
    }
    if "example_only" in record:
        cpt_row["example_only"] = record["example_only"]

    errors = cpt_builder.validate_cpt_record(cpt_row)
    if errors:
        raise ValueError(f"Converted CPT row {cpt_row['sample_id']} invalid: {errors}")
    return cpt_row


def route_v1_v4_to_cpt(
    cleaned_records: list[dict[str, Any]],
    *,
    include_final_in_cpt: bool = False,
    include_v2_in_cpt: bool = False,
    rag_evidence_map: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Routes eligible V1-V4 records into CPT replay/RAFT shards while skipping V2/final."""
    routed: list[dict[str, Any]] = []
    skipped_reasons: Counter[str] = Counter()

    for rec in cleaned_records:
        variant = int(rec["variant"])
        pool = rec.get("pool", "unspecified")
        if not include_final_in_cpt and pool == "final":
            skipped_reasons["reserved_final_for_sft_eval"] += 1
            continue
        if variant == 2 and not include_v2_in_cpt:
            skipped_reasons["filtered_v2_caption_dilution"] += 1
            continue
        routed.append(convert_record_to_cpt(rec, rag_evidence_map=rag_evidence_map))

    summary = {
        "cpt_routed_rows": len(routed),
        "skipped_reasons": dict(sorted(skipped_reasons.items())),
        "channels": dict(sorted(Counter(r["channel"] for r in routed).items())),
    }
    return routed, summary


def estimate_assistant_supervised_tokens(record: dict[str, Any]) -> int:
    """Estimates supervised assistant tokens (excluding prompt/images) for SFT balancing."""
    assistant_text = "\n".join(
        str(m.get("content", ""))
        for m in record.get("messages", [])
        if m.get("role") == "assistant"
    )
    return cpt_builder.estimate_record_tokens(assistant_text, num_images=0)


def _normalize_sft_record(record: dict[str, Any], default_channel: str) -> dict[str, Any]:
    variant = record.get("variant")
    channel = record.get("channel")
    if not isinstance(channel, str) or not channel.strip():
        channel = f"V{variant}" if variant is not None else default_channel

    out: dict[str, Any] = {
        "messages": list(record["messages"]),
        "channel": channel,
        "doc_family_id": _derive_doc_family_id(record),
    }
    images = record.get("images") or []
    if images:
        out["images"] = list(images)
    for key in ("sample_id", "cate_id", "variant", "pool", "enable_thinking", "example_only"):
        if key in record:
            out[key] = record[key]
    return out


def _strip_think_block(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for msg in messages:
        copied = dict(msg)
        if copied.get("role") == "assistant" and isinstance(copied.get("content"), str):
            copied["content"] = THINK_BLOCK_RE.sub("", copied["content"]).strip()
        cleaned.append(copied)
    return cleaned


def _prepare_qa_for_sft_arm(
    new_qa_records: list[dict[str, Any]],
    *,
    sft_arm: str,
    cot_direct_ratio: float,
    seed: int,
) -> list[dict[str, Any]]:
    if sft_arm not in VALID_SFT_ARMS:
        raise ValueError(f"Unsupported sft_arm={sft_arm!r}; expected one of {VALID_SFT_ARMS}")
    if sft_arm == "ood_only":
        return []
    if sft_arm == "unified":
        return [_normalize_sft_record(r, default_channel="wiki_rag_qa") for r in new_qa_records]
    if sft_arm == "ood_plus_qa_direct":
        out: list[dict[str, Any]] = []
        for r in new_qa_records:
            rec = _normalize_sft_record(r, default_channel="wiki_rag_qa_direct")
            rec["channel"] = "wiki_rag_qa_direct"
            rec["enable_thinking"] = False
            rec["messages"] = _strip_think_block(rec["messages"])
            out.append(rec)
        return out

    # sft_arm == "ood_plus_qa_cot" (8:2 CoT:Direct mixture by default)
    cot_rows: list[dict[str, Any]] = []
    direct_rows: list[dict[str, Any]] = []
    for r in new_qa_records:
        rec = _normalize_sft_record(r, default_channel="wiki_rag_qa_cot")
        has_think = any(
            m.get("role") == "assistant" and "<think>" in str(m.get("content", ""))
            for m in rec["messages"]
        )
        is_cot = (rec.get("enable_thinking") is True) or has_think
        if is_cot:
            rec["channel"] = "wiki_rag_qa_cot"
            rec["enable_thinking"] = True
            cot_rows.append(rec)
        else:
            rec["channel"] = "wiki_rag_qa_direct"
            rec["enable_thinking"] = False
            rec["messages"] = _strip_think_block(rec["messages"])
            direct_rows.append(rec)

    if cot_rows and direct_rows and 0.0 < cot_direct_ratio < 1.0:
        rng = random.Random(seed)
        desired_direct = max(1, int(round(len(cot_rows) * (1.0 - cot_direct_ratio) / cot_direct_ratio)))
        if len(direct_rows) > desired_direct:
            rng.shuffle(direct_rows)
            direct_rows = direct_rows[:desired_direct]
    return cot_rows + direct_rows


def build_unified_sft_bundle(
    v1_v4_records: list[dict[str, Any]],
    new_qa_records: list[dict[str, Any]],
    *,
    target_rows: int | None = None,
    sft_token_weights: dict[str, float] | None = None,
    sft_arm: str = "unified",
    cot_direct_ratio: float = 0.8,
    val_ratio: float = 0.02,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Merges V1-V4 and new Wiki/RAG/CoT QA into token-balanced train/val SFT sets."""
    weights = dict(sft_token_weights or DEFAULT_SFT_TOKEN_WEIGHTS)
    if sft_arm == "ood_plus_qa_cot":
        qa_total_w = weights.get("wiki_rag_qa", 0.20)
        weights["wiki_rag_qa_cot"] = qa_total_w * cot_direct_ratio
        weights["wiki_rag_qa_direct"] = qa_total_w * (1.0 - cot_direct_ratio)
    elif sft_arm == "ood_plus_qa_direct":
        weights["wiki_rag_qa_direct"] = weights.get("wiki_rag_qa", 0.20)

    qa_prepared = _prepare_qa_for_sft_arm(
        new_qa_records,
        sft_arm=sft_arm,
        cot_direct_ratio=cot_direct_ratio,
        seed=seed,
    )
    normalized: list[dict[str, Any]] = [
        _normalize_sft_record(r, default_channel=f"V{r['variant']}") for r in v1_v4_records
    ] + qa_prepared

    # Never carve validation rows from V5 rule replay (mirrors legacy S2/S3 invariant).
    splittable = [r for r in normalized if r.get("variant") != 5 and r.get("channel") != "V5"]
    replay_only = [r for r in normalized if r.get("variant") == 5 or r.get("channel") == "V5"]

    train_splittable, val_rows = cpt_builder.split_by_doc_family(
        splittable,
        val_ratio=val_ratio,
        seed=seed,
        min_val_families=1 if len({r["doc_family_id"] for r in splittable}) >= 2 else 0,
    )
    train_pool = train_splittable + replay_only

    val_fam_set = {r["doc_family_id"] for r in val_rows}
    train_fam_set = {r["doc_family_id"] for r in train_pool}
    if val_fam_set & train_fam_set:
        raise AssertionError("doc_family_id overlap between SFT train and val")

    by_channel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in train_pool:
        by_channel[str(rec["channel"])].append(rec)

    rng = random.Random(seed)
    if target_rows is None:
        train_out = list(train_pool)
        rng.shuffle(train_out)
    else:
        avg_sup_toks = {
            ch: sum(estimate_assistant_supervised_tokens(r) for r in rows) / len(rows)
            for ch, rows in by_channel.items()
        }
        raw_scores = {
            ch: weights.get(ch, 0.10) / max(1.0, avg_sup_toks[ch])
            for ch in by_channel
        }
        score_sum = sum(raw_scores.values()) or 1.0
        train_out = []
        for ch in sorted(by_channel.keys()):
            pool = by_channel[ch]
            cnt = max(1, int(round(target_rows * (raw_scores[ch] / score_sum))))
            if cnt <= len(pool):
                picks = list(pool)
                rng.shuffle(picks)
                train_out.extend(picks[:cnt])
            else:
                train_out.extend(pool)
                train_out.extend(rng.choices(pool, k=cnt - len(pool)))
        rng.shuffle(train_out)

    tok_totals: Counter[str] = Counter()
    row_totals: Counter[str] = Counter()
    for r in train_out:
        ch = str(r["channel"])
        row_totals[ch] += 1
        tok_totals[ch] += estimate_assistant_supervised_tokens(r)
    grand_toks = sum(tok_totals.values()) or 1

    summary = {
        "sft_arm": sft_arm,
        "train_rows": len(train_out),
        "val_rows": len(val_rows),
        "doc_family_overlap": len(val_fam_set & train_fam_set),
        "channels": {
            ch: {
                "rows": row_totals[ch],
                "assistant_tokens": tok_totals[ch],
                "assistant_token_share": round(tok_totals[ch] / grand_toks, 4),
            }
            for ch in sorted(row_totals.keys())
        },
    }
    return train_out, val_rows, summary


def extract_qwen_image_edit_seeds(
    cleaned_records: list[dict[str, Any]],
    *,
    hard_cates: tuple[str, ...] = DEFAULT_HARD_EDIT_CATES,
    cate_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Extracts V1/V3 hard-cluster images to feed Qwen-Image-Edit counterfactual augmentation."""
    seeds: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    cate_set = set(hard_cates)
    for rec in cleaned_records:
        if rec.get("variant") not in (1, 3):
            continue
        cate_key = str(rec.get("cate_id"))
        if cate_key not in cate_set and not (cate_map and cate_key in cate_map):
            continue
        for img in rec.get("images") or []:
            if img in seen_images:
                continue
            seen_images.add(img)
            seed_item: dict[str, Any] = {
                "seed_id": f"edit_seed_{rec['sample_id']}_{len(seeds)}",
                "source_sample_id": rec["sample_id"],
                "source_variant": rec["variant"],
                "cate_id": rec["cate_id"],
                "doc_family_id": rec["doc_family_id"],
                "image": img,
                "recommended_ops": ["background_relight_protect", "counterfactual_erasure"],
            }
            if cate_map and cate_key in cate_map:
                seed_item["cate_name"] = cate_map[cate_key]
            seeds.append(seed_item)
    return seeds


def run_refactor_pipeline(
    v1_v4_paths: list[Path],
    output_dir: Path,
    *,
    new_qa_paths: list[Path] | None = None,
    rag_evidence_map_path: Path | None = None,
    cate_map_path: Path | None = None,
    include_final_in_cpt: bool = False,
    include_v2_in_cpt: bool = False,
    dedup_same_pool: bool = False,
    sft_target_rows: int | None = None,
    sft_arm: str = "unified",
    cot_direct_ratio: float = 0.8,
    val_ratio: float = 0.02,
    seed: int = 42,
) -> dict[str, Any]:
    """Executes end-to-end validation, CPT routing, Unified SFT balancing, and seed export."""
    records_with_paths: list[tuple[dict[str, Any], Path | None]] = []
    for p in v1_v4_paths:
        for row in load_records_auto(p):
            records_with_paths.append((row, p))

    new_qa_records: list[dict[str, Any]] = []
    for p in new_qa_paths or []:
        new_qa_records.extend(load_records_auto(p))

    rag_evidence_map: dict[str, list[str]] | None = None
    if rag_evidence_map_path is not None:
        rag_evidence_map = json.loads(rag_evidence_map_path.read_text(encoding="utf-8"))

    cate_map: dict[str, str] | None = None
    if cate_map_path is not None:
        raw_map = json.loads(cate_map_path.read_text(encoding="utf-8"))
        cate_map = {str(k): str(v) for k, v in raw_map.items()}

    cleaned, dedup_report = validate_and_dedup_v1_v4(
        records_with_paths,
        dedup_same_pool=dedup_same_pool,
    )
    cpt_rows, cpt_summary = route_v1_v4_to_cpt(
        cleaned,
        include_final_in_cpt=include_final_in_cpt,
        include_v2_in_cpt=include_v2_in_cpt,
        rag_evidence_map=rag_evidence_map,
    )
    sft_train, sft_val, sft_summary = build_unified_sft_bundle(
        cleaned,
        new_qa_records,
        target_rows=sft_target_rows,
        sft_arm=sft_arm,
        cot_direct_ratio=cot_direct_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    edit_seeds = extract_qwen_image_edit_seeds(cleaned, cate_map=cate_map)

    output_dir.mkdir(parents=True, exist_ok=True)
    cpt_path = output_dir / "cpt_v1_v4_replay.jsonl"
    sft_train_path = output_dir / "sft_unified_train.jsonl"
    sft_val_path = output_dir / "sft_unified_val.jsonl"
    seeds_path = output_dir / "qwen_image_edit_seeds.jsonl"
    manifest_path = output_dir / "refactor_manifest.json"

    cpt_builder.write_jsonl(cpt_path, cpt_rows)
    cpt_builder.write_jsonl(sft_train_path, sft_train)
    cpt_builder.write_jsonl(sft_val_path, sft_val)
    cpt_builder.write_jsonl(seeds_path, edit_seeds)

    manifest = {
        "schema": "unified_v1_v4_cpt_qa_v1",
        "seed": seed,
        "dedup_validation": dedup_report,
        "cpt_routing": cpt_summary,
        "unified_sft": sft_summary,
        "qwen_image_edit_seeds_count": len(edit_seeds),
        "artifacts": {
            "cpt_v1_v4_replay": {
                "path": str(cpt_path),
                "sha256": cpt_builder.sha256_file(cpt_path),
            },
            "sft_unified_train": {
                "path": str(sft_train_path),
                "sha256": cpt_builder.sha256_file(sft_train_path),
            },
            "sft_unified_val": {
                "path": str(sft_val_path),
                "sha256": cpt_builder.sha256_file(sft_val_path),
            },
            "qwen_image_edit_seeds": {
                "path": str(seeds_path),
                "sha256": cpt_builder.sha256_file(seeds_path),
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
    parser.add_argument("--v1-v4-inputs", nargs="*", type=Path, default=[])
    parser.add_argument("--new-qa-inputs", nargs="*", type=Path, default=[])
    parser.add_argument("--rag-evidence-map", type=Path, default=None)
    parser.add_argument("--cate-map", type=Path, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-final-in-cpt", action="store_true")
    parser.add_argument("--include-v2-in-cpt", action="store_true")
    parser.add_argument("--dedup-same-pool", action="store_true")
    parser.add_argument("--sft-target-rows", type=int, default=None)
    parser.add_argument(
        "--sft-arm",
        choices=list(VALID_SFT_ARMS),
        default="unified",
        help="Experimental arm for SFT bundle: unified | ood_only | ood_plus_qa_direct | ood_plus_qa_cot",
    )
    parser.add_argument(
        "--cot-direct-ratio",
        type=float,
        default=0.8,
        help="Target CoT share within QA pool when --sft-arm=ood_plus_qa_cot (default 0.8 = 8:2)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.demo:
        v1_v4_inputs = [THIS_DIR / "examples" / "v1_v4_legacy_sample.jsonl"]
        new_qa_inputs = [THIS_DIR / "examples" / "style_sft_example.jsonl"]
        val_ratio = args.val_ratio if args.val_ratio >= 0.15 else 0.20
        sft_target_rows = args.sft_target_rows if args.sft_target_rows is not None else 20
    elif args.v1_v4_inputs:
        v1_v4_inputs = args.v1_v4_inputs
        new_qa_inputs = args.new_qa_inputs
        val_ratio = args.val_ratio
        sft_target_rows = args.sft_target_rows
    else:
        print("Specify --demo or --v1-v4-inputs.", file=sys.stderr)
        return 2

    manifest = run_refactor_pipeline(
        v1_v4_inputs,
        args.output_dir,
        new_qa_paths=new_qa_inputs,
        rag_evidence_map_path=args.rag_evidence_map,
        cate_map_path=args.cate_map,
        include_final_in_cpt=args.include_final_in_cpt,
        include_v2_in_cpt=args.include_v2_in_cpt,
        dedup_same_pool=args.dedup_same_pool,
        sft_target_rows=sft_target_rows,
        sft_arm=args.sft_arm,
        cot_direct_ratio=args.cot_direct_ratio,
        val_ratio=val_ratio,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
