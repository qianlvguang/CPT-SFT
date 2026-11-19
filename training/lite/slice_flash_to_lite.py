#!/usr/bin/env python3
"""Proportional Stratified Flash->Lite Slicer (Phase A Gain Control, val0).

Replaces the old 95%-sch iso Lite dataset that collapsed to 7 B300 steps under
`GBS=32 * 16384`. Slices whole `doc_family_id` groups proportionally from Flash's
sanitized 4-bucket Mid-Training pool (`--ratio`, default 0.15) and computes
calibrated B300 Lite step projections at `MBS=1, GBS=4, MAX_LENGTH=8192`
(`32,768 tok/step`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
FLASH_DIR = THIS_DIR.parent / "flash"

import sys

if str(FLASH_DIR) not in sys.path:
    sys.path.insert(0, str(FLASH_DIR))

from build_mm_wiki_cpt import (  # noqa: E402
    _record_text_content,
    _summarize_buckets,
    bucket_for_channel,
    compute_synthetic_image_share,
    estimate_record_tokens,
    load_jsonl,
    sha256_file,
    write_jsonl,
)


def _is_qa_prose(row: dict[str, Any]) -> bool:
    return bool(row.get("is_qa_prose")) or str(row.get("channel")) in {
        "ch_qa_knowledge_prose",
        "wiki_rag_qa_cot",
        "wiki_rag_qa_direct",
    }


def _demo_flash_midtrain_pool() -> list[dict[str, Any]]:
    """Builds a small realistic 4-bucket demo pool when `--demo` is passed directly."""
    records: list[dict[str, Any]] = []
    channels = [
        ("ch_text_doc", "domain_kb_v2", 0, False),
        ("ch_mm_doc", "domain_kb_v2", 1, False),
        ("ch_v2_multi_img_doc", "domain_kb_v2", 2, True),
        ("ch6_general_replay", "real_replay", 1, False),
        ("ch4_audit_crawl_cf", "cf_distractor", 2, False),
        ("ch_qa_knowledge_prose", "qa_prose_anti_collapse", 0, False),
    ]
    idx = 0
    for ch, bucket, n_img, is_synth in channels:
        for f in range(4):
            idx += 1
            imgs = [f"/mock/img_{idx}_{i}.jpg" for i in range(n_img)]
            prefix = "\n".join("<image>" for _ in imgs)
            text = (
                f"{prefix}\n[{ch}] 示例实体家族 fam_{ch}_{f} 标准属性描述与设计要素比对。"
                if prefix
                else f"[{ch}] 示例领域知识家族 fam_{ch}_{f}：核心视觉锚点、合规约束与易混淆边界。"
            )
            records.append(
                {
                    "sample_id": f"lite_demo_{idx}",
                    "doc_family_id": f"fam_{ch}_{f}",
                    "channel": ch,
                    "bucket": bucket,
                    "images": imgs,
                    "is_synthetic_image": is_synth,
                    "synthetic_image_count": 1 if is_synth else 0,
                    "is_qa_prose": ch == "ch_qa_knowledge_prose",
                    "messages": [{"role": "assistant", "content": text}],
                }
            )
    return records


def slice_flash_to_lite(
    flash_rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    ratio: float = 0.15,
    epochs: int = 2,
    lite_mbs: int = 1,
    lite_gbs: int = 4,
    lite_max_length: int = 8192,
    tokens_per_image: int = 256,
    seed: int = 42,
) -> dict[str, Any]:
    """Performs stratified whole-family proportional slicing from Flash -> Lite (val0)."""
    if not flash_rows:
        raise ValueError("flash_rows must not be empty")
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")

    rng = random.Random(seed)

    # Group by doc_family_id globally so no family is ever split across partial slices
    fam_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in flash_rows:
        fam_id = str(row.get("doc_family_id") or "").strip()
        if not fam_id:
            fam_id = "anon_" + hashlib.sha256(
                _record_text_content(row).encode("utf-8")
            ).hexdigest()[:12]
        item = dict(row)
        item["doc_family_id"] = fam_id
        item["bucket"] = str(item.get("bucket") or bucket_for_channel(str(item["channel"])))
        fam_groups[fam_id].append(item)

    # Assign each family to its primary bucket (most frequent bucket among its rows)
    families_by_bucket: dict[str, list[str]] = defaultdict(list)
    for fam_id, rows in fam_groups.items():
        b_counts = Counter(str(r["bucket"]) for r in rows)
        primary_bucket = sorted(b_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        families_by_bucket[primary_bucket].append(fam_id)

    selected_families: list[str] = []
    bucket_family_quota: dict[str, dict[str, int]] = {}
    for b in sorted(families_by_bucket.keys()):
        fams = sorted(families_by_bucket[b])
        rng.shuffle(fams)
        take = max(1, int(round(len(fams) * ratio)))
        take = min(take, len(fams))
        selected_families.extend(fams[:take])
        bucket_family_quota[b] = {"flash_families": len(fams), "lite_families": take}

    selected_set = set(selected_families)
    lite_rows: list[dict[str, Any]] = []
    for fam_id in sorted(selected_set):
        lite_rows.extend(fam_groups[fam_id])
    rng.shuffle(lite_rows)

    # Token & step projection for calibrated B300 Lite (MBS=1, GBS=4, 8192)
    est_tokens = sum(
        estimate_record_tokens(
            _record_text_content(r),
            len(r.get("images") or []),
            tokens_per_image=tokens_per_image,
        )
        for r in lite_rows
    )
    tok_per_step = max(1, lite_gbs * lite_max_length)
    projected_steps = round((est_tokens * max(1, epochs)) / tok_per_step, 2)
    low_step_warning = projected_steps < 60.0

    # Check audits preserved from Flash
    img_audit = compute_synthetic_image_share(lite_rows)
    qa_tok = sum(
        estimate_record_tokens(
            _record_text_content(r),
            len(r.get("images") or []),
            tokens_per_image=tokens_per_image,
        )
        for r in lite_rows
        if _is_qa_prose(r)
    )
    qa_prose_share = round(qa_tok / max(1, est_tokens), 4)
    partial_family_split = 0
    for fam_id in selected_set:
        src_cnt = len(fam_groups[fam_id])
        dst_cnt = sum(1 for r in lite_rows if r["doc_family_id"] == fam_id)
        if dst_cnt != src_cnt:
            partial_family_split += 1

    bundle_lite_dir = output_dir / "bundle_lite"
    compat_dir = output_dir / "bundle_lite_compat"
    bundle_lite_dir.mkdir(parents=True, exist_ok=True)
    compat_dir.mkdir(parents=True, exist_ok=True)

    lite_train_path = bundle_lite_dir / "lite_train.jsonl"
    manifest_path = bundle_lite_dir / "lite_slice_manifest.json"
    compat_stage1_path = compat_dir / "stage1_train.jsonl"

    write_jsonl(lite_train_path, lite_rows)
    if compat_stage1_path.exists() or compat_stage1_path.is_symlink():
        compat_stage1_path.unlink()
    try:
        os.symlink("../bundle_lite/lite_train.jsonl", compat_stage1_path)
    except OSError:
        shutil.copy2(lite_train_path, compat_stage1_path)

    bucket_summary = _summarize_buckets(lite_rows, tokens_per_image=tokens_per_image)
    manifest = {
        "schema": "lite_proportional_flash_slice_v2",
        "seed": seed,
        "ratio": ratio,
        "val_dataset": None,
        "val_ratio": 0.0,
        "source_flash_rows": len(flash_rows),
        "source_flash_families": len(fam_groups),
        "lite_rows": len(lite_rows),
        "lite_families": len(selected_set),
        "partial_family_splits": partial_family_split,
        "bucket_family_quota": bucket_family_quota,
        "bucket_summary": bucket_summary,
        "synthetic_image_audit": img_audit,
        "qa_prose_token_share": qa_prose_share,
        "b300_step_projection": {
            "micro_batch_size": lite_mbs,
            "global_batch_size": lite_gbs,
            "max_length": lite_max_length,
            "tokens_per_step": tok_per_step,
            "epochs": epochs,
            "estimated_total_tokens_per_epoch": est_tokens,
            "projected_optimizer_steps": projected_steps,
            "recommended_min_steps": 60,
            "low_step_warning": low_step_warning,
            "recommended_min_rows_for_100_steps": math.ceil((100 * tok_per_step) / max(1, epochs * max(1, est_tokens // max(1, len(lite_rows))))),
        },
        "artifacts": {
            "lite_train": str(lite_train_path),
            "lite_train_sha256": sha256_file(lite_train_path),
            "compat_stage1_train": str(compat_stage1_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="Path to Flash midtrain_train.jsonl")
    parser.add_argument("--demo", action="store_true", help="Slice a built-in synthetic 4-bucket demo pool")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lite-mbs", type=int, default=1)
    parser.add_argument("--lite-gbs", type=int, default=4)
    parser.add_argument("--lite-max-length", type=int, default=8192)
    parser.add_argument("--tokens-per-image", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.demo:
        rows = _demo_flash_midtrain_pool()
    elif args.input:
        rows = load_jsonl(args.input)
    else:
        raise SystemExit("Specify --demo or --input <midtrain_train.jsonl>.")

    manifest = slice_flash_to_lite(
        rows,
        args.output_dir,
        ratio=args.ratio,
        epochs=args.epochs,
        lite_mbs=args.lite_mbs,
        lite_gbs=args.lite_gbs,
        lite_max_length=args.lite_max_length,
        tokens_per_image=args.tokens_per_image,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
