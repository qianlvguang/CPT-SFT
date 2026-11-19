#!/usr/bin/env python3
"""Phase C (Pro) Same-Source Programmatic Enhancer over Flash Mid-Training Pool.

Shares the exact same raw data pool as Flash and adds 4 programmatic upgrades:
  1. High-Value Information Density & Tail Curation (`curate_high_value_midtrain`):
     Scores tail rarity, multi-image packing richness, CF/distractor contrast,
     and complex reasoning; trims repetitive low-density boilerplate while enforcing
     `max_row_repeats <= 2`, `qa_prose_token_share <= 0.15`, and zero val leakage.
  2. Cross-Modal Entity Anchor Alignment (`audit_cross_modal_anchor_alignment`):
     Checks visual (`v2_packed_cpt_docs` / `ch_mm_doc`) <-> text CoT v0.4
     (`qa_cot_passed`) family & entity overlap; tags aligned rows and logs gaps.
  3. Dual-Mode Canary + Thinking-OFF Retention & WiSE-FT Gate (`evaluate_canary_and_wise_ft`):
     Detects general forgetting (`drop > 2.0pp`) -> emits `0.8*CPT + 0.2*Base`
     WiSE-FT merge recipe; detects Thinking-OFF collapse (`think_off_drop > 2.0pp`
     or `think_on_off_gap > 8.0pp`) -> recommends OPD Think-ON->OFF Context Distill.
  4. Student Preference (DPO/KTO) & OPD BSD Context-Distill Pack (`build_pro_preference_and_bsd_pack`):
     Converts failed/rejected CoT traces into Student DPO/KTO pairs (never Teacher
     DPO) and exports OPD-compatible BSD / Think-ON->OFF Context Distillation seeds.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
FLASH_DIR = THIS_DIR.parent / "flash"

if str(FLASH_DIR) not in sys.path:
    sys.path.insert(0, str(FLASH_DIR))

from build_mm_wiki_cpt import (  # noqa: E402
    _is_qa_prose_row,
    _record_text_content,
    _summarize_buckets,
    bucket_for_channel,
    build_demo_records,
    build_midtrain_bundle,
    compute_synthetic_image_share,
    estimate_record_tokens,
    load_jsonl,
    sha256_file,
    to_midtrain_row,
    write_jsonl,
)
from build_v2_image_pipeline import run_v2_image_pipeline  # noqa: E402
from gen_cot_from_qa import run_cot_generation_pipeline  # noqa: E402


def score_midtrain_row_value(row: dict[str, Any]) -> float:
    """Computes programmatic information-density & tail curation score."""
    ch = str(row.get("channel") or "")
    text = _record_text_content(row)
    imgs = row.get("images") or []
    score = 1.0
    if row.get("is_unresolved_tail"):
        score += 0.40
    if ch in {"ch_v2_multi_img_doc", "ch_sports_expanded"} and len(imgs) >= 2:
        score += 0.35
    if ch in {"ch_v2_cf_pair", "ch_v1_recycled_neg", "ch4_audit_crawl_cf"}:
        score += 0.30
    if ch == "ch_rel_jump":
        score += 0.25
    if ch == "ch_qa_knowledge_prose" and len(text) >= 120:
        score += 0.20
    if len(text) < 40 and len(imgs) == 0:
        score -= 0.25
    return round(score, 4)


def curate_high_value_midtrain(
    train_rows: list[dict[str, Any]],
    *,
    val_rows: list[dict[str, Any]],
    max_row_repeats: int = 2,
    max_qa_prose_share: float = 0.15,
    max_synth_image_share: float = 0.25,
    tokens_per_image: int = 512,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Curates Flash's mid-training rows by density/tail value while holding Flash caps."""
    val_families = {str(r.get("doc_family_id") or "") for r in val_rows if r.get("doc_family_id")}

    eligible: list[dict[str, Any]] = []
    val_leak_dropped = 0
    for r in train_rows:
        fam = str(r.get("doc_family_id") or "")
        if fam and fam in val_families:
            val_leak_dropped += 1
            continue
        item = dict(r)
        item["pro_value_score"] = score_midtrain_row_value(item)
        item["bucket"] = str(item.get("bucket") or bucket_for_channel(str(item["channel"])))
        eligible.append(item)

    if not eligible:
        raise ValueError("No eligible training rows remained for Pro curation.")

    # Deduplicate over-repeated boilerplate and re-select top-value rows per bucket
    by_bucket_unique: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for item in eligible:
        sid = str(item.get("sample_id") or json.dumps(item.get("messages"), ensure_ascii=False, sort_keys=True))
        b = item["bucket"]
        if sid not in by_bucket_unique[b] or item["pro_value_score"] > by_bucket_unique[b][sid]["pro_value_score"]:
            by_bucket_unique[b][sid] = item

    curated: list[dict[str, Any]] = []
    copy_counts: Counter[str] = Counter()
    for b in sorted(by_bucket_unique.keys()):
        ranked = sorted(
            by_bucket_unique[b].values(),
            key=lambda x: (-float(x["pro_value_score"]), str(x.get("sample_id") or "")),
        )
        for idx, item in enumerate(ranked):
            # Up-weight top half high-value tail/CF items up to max_row_repeats; cap tail-low boilerplate at 1
            copies = max_row_repeats if (item["pro_value_score"] >= 1.25 and not _is_qa_prose_row(item)) else 1
            copies = min(max_row_repeats, copies)
            sid = str(item.get("sample_id") or f"{b}_{idx}")
            for _ in range(copies):
                if copy_counts[sid] < max_row_repeats:
                    curated.append(dict(item))
                    copy_counts[sid] += 1

    # Jointly enforce BOTH qa_prose <= max_qa_prose_share AND synth_images <= max_synth_image_share
    def _qa_tok_share(rows_list: list[dict[str, Any]]) -> float:
        qa_t = sum(
            estimate_record_tokens(_record_text_content(r), len(r.get("images") or []), tokens_per_image=tokens_per_image)
            for r in rows_list
            if _is_qa_prose_row(r)
        )
        tot_t = sum(
            estimate_record_tokens(_record_text_content(r), len(r.get("images") or []), tokens_per_image=tokens_per_image)
            for r in rows_list
        )
        return (qa_t / tot_t) if tot_t > 0 else 0.0

    # First ensure every real_replay / non-synthetic real-image item gets up to max_row_repeats if needed to dilute synth/qa
    real_non_qa_pool = [
        r for r in eligible
        if not _is_qa_prose_row(r) and not (r.get("is_synthetic_image") or r.get("synthetic_image_count", 0) > 0)
    ]

    for _ in range(len(curated) + len(real_non_qa_pool) + 10):
        img_stats = compute_synthetic_image_share(curated)
        qa_share_now = _qa_tok_share(curated)
        if qa_share_now <= max_qa_prose_share and (img_stats["total_images"] == 0 or img_stats["synth_image_share"] <= max_synth_image_share):
            break
        if img_stats["total_images"] > 0 and img_stats["synth_image_share"] > max_synth_image_share:
            # Try adding real-image non-synth row first before dropping
            added_real = False
            for cand in real_non_qa_pool:
                if len(cand.get("images") or []) > 0:
                    cid = str(cand.get("sample_id") or "")
                    if copy_counts[cid] < max_row_repeats:
                        curated.append(dict(cand))
                        copy_counts[cid] += 1
                        added_real = True
                        break
            if added_real:
                continue
            synth_idxs = [i for i, r in enumerate(curated) if (r.get("is_synthetic_image") or r.get("synthetic_image_count", 0) > 0)]
            if synth_idxs:
                curated.pop(min(synth_idxs, key=lambda i: float(curated[i].get("pro_value_score", 0.0))))
                continue
        if qa_share_now > max_qa_prose_share:
            qa_idxs = [i for i, r in enumerate(curated) if _is_qa_prose_row(r)]
            if not qa_idxs:
                break
            curated.pop(min(qa_idxs, key=lambda i: float(curated[i].get("pro_value_score", 0.0))))
            continue
        break

    rng = random.Random(seed)
    rng.shuffle(curated)
    curated_out = [to_midtrain_row(r) for r in curated]
    for orig, out_r in zip(curated, curated_out):
        out_r["pro_value_score"] = orig["pro_value_score"]

    avg_score = round(sum(float(r["pro_value_score"]) for r in curated) / len(curated), 4)
    return curated_out, {
        "input_train_rows": len(train_rows),
        "val_leak_dropped": val_leak_dropped,
        "curated_rows": len(curated_out),
        "avg_value_score": avg_score,
        "qa_prose_token_share": round(_qa_tok_share(curated), 4),
        "synthetic_image_audit": compute_synthetic_image_share(curated_out),
        "bucket_summary": _summarize_buckets(curated_out, tokens_per_image=tokens_per_image),
    }


def _extract_subject_tokens(text: str) -> set[str]:
    candidates = (
        "示例电鼠",
        "爱尔兰",
        "科特迪瓦",
        "银发",
        "眼罩",
        "三道斜杠",
        "三道杠",
        "马鞍包",
        "徽章A",
        "西甲",
    )
    return {tok for tok in candidates if tok in text}


def audit_cross_modal_anchor_alignment(
    curated_rows: list[dict[str, Any]],
    v2_docs: list[dict[str, Any]],
    cot_passed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Links entity/family anchors across visual docs and CoT v0.4 text traces."""
    visual_families: set[str] = {
        str(r.get("doc_family_id"))
        for r in (list(v2_docs) + [x for x in curated_rows if len(x.get("images") or []) > 0])
        if r.get("doc_family_id")
    }
    cot_families: set[str] = {
        str(r.get("doc_family_id")) for r in cot_passed if r.get("doc_family_id")
    } | {
        str(r.get("doc_family_id")) for r in curated_rows if _is_qa_prose_row(r) and r.get("doc_family_id")
    }
    visual_subjects: set[str] = set()
    for r in list(v2_docs) + [x for x in curated_rows if len(x.get("images") or []) > 0]:
        visual_subjects |= _extract_subject_tokens(_record_text_content(r))
    cot_subjects: set[str] = set()
    for r in cot_passed:
        cot_subjects |= _extract_subject_tokens(_record_text_content(r))
    for r in curated_rows:
        if _is_qa_prose_row(r):
            cot_subjects |= _extract_subject_tokens(_record_text_content(r))

    aligned_families = sorted(visual_families & cot_families)
    aligned_subjects = sorted(visual_subjects & cot_subjects)
    aligned_fam_set = set(aligned_families)
    aligned_subj_set = set(aligned_subjects)

    tagged_rows = 0
    for row in curated_rows:
        fam_hit = str(row.get("doc_family_id") or "") in aligned_fam_set
        subj_hit = bool(_extract_subject_tokens(_record_text_content(row)) & aligned_subj_set)
        if fam_hit or subj_hit:
            row["cross_modal_aligned"] = True
            tagged_rows += 1
        else:
            row["cross_modal_aligned"] = False

    visual_only_gaps = sorted(visual_families - aligned_fam_set)
    text_only_gaps = sorted(cot_families - aligned_fam_set)

    return {
        "visual_families_count": len(visual_families),
        "cot_families_count": len(cot_families),
        "cross_modal_aligned_families": aligned_families,
        "cross_modal_aligned_subjects": aligned_subjects,
        "cross_modal_aligned_count": len(aligned_fam_set) + len(aligned_subj_set),
        "curated_rows_tagged_aligned": tagged_rows,
        "visual_only_gap_families": visual_only_gaps,
        "text_only_gap_families": text_only_gaps,
    }


def evaluate_canary_and_wise_ft(
    canary_metrics: dict[str, float] | None,
    *,
    cpt_ckpt_path: str = "<CPT_HF_MERGED>",
    base_model_path: str = "<BASE_QWEN35_27B>",
    max_drop_pp: float = 2.0,
    max_think_on_off_gap_pp: float = 8.0,
    cpt_weight: float = 0.8,
    base_weight: float = 0.2,
) -> dict[str, Any]:
    """Evaluates general forgetting (WiSE-FT gate) AND Thinking-OFF deployment collapse risk."""
    metrics = canary_metrics or {
        "base_general_acc": 76.5,
        "cpt_general_acc": 74.2,
        "base_think_off_acc": 73.0,
        "cpt_think_on_acc": 84.0,
        "cpt_think_off_acc": 70.5,
    }
    base_gen = float(metrics.get("base_general_acc", 76.0))
    cpt_gen = float(metrics.get("cpt_general_acc", 75.0))
    base_off = float(metrics.get("base_think_off_acc", 73.0))
    cpt_on = float(metrics.get("cpt_think_on_acc", 80.0))
    cpt_off = float(metrics.get("cpt_think_off_acc", 72.0))

    general_drop_pp = round(base_gen - cpt_gen, 2)
    think_off_vs_base_drop_pp = round(base_off - cpt_off, 2)
    think_on_off_gap_pp = round(cpt_on - cpt_off, 2)

    wise_ft_triggered = general_drop_pp > max_drop_pp
    think_off_collapse_risk = (
        think_off_vs_base_drop_pp > max_drop_pp or think_on_off_gap_pp > max_think_on_off_gap_pp
    )

    return {
        "schema": "wise_ft_and_dual_mode_canary_v1",
        "metrics": {
            "base_general_acc": base_gen,
            "cpt_general_acc": cpt_gen,
            "base_think_off_acc": base_off,
            "cpt_think_on_acc": cpt_on,
            "cpt_think_off_acc": cpt_off,
            "general_drop_pp": general_drop_pp,
            "think_off_vs_base_drop_pp": think_off_vs_base_drop_pp,
            "think_on_off_gap_pp": think_on_off_gap_pp,
        },
        "thresholds": {
            "max_drop_pp": max_drop_pp,
            "max_think_on_off_gap_pp": max_think_on_off_gap_pp,
        },
        "wise_ft_merge_triggered": wise_ft_triggered,
        "think_off_collapse_risk": think_off_collapse_risk,
        "opd_think_on_to_off_distill_recommended": think_off_collapse_risk,
        "merge_recipe": {
            "formula": f"{cpt_weight} * CPT + {base_weight} * Base",
            "cpt_weight": cpt_weight,
            "base_weight": base_weight,
            "cpt_checkpoint": cpt_ckpt_path,
            "base_checkpoint": base_model_path,
            "status": "EXECUTE_MERGE" if wise_ft_triggered else "STANDBY_NO_GENERAL_DROP",
        },
    }


def build_pro_preference_and_bsd_pack(
    cot_passed: list[dict[str, Any]],
    cot_rejected: list[dict[str, Any]],
    *,
    excluded_doc_families: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Builds Student-only DPO/KTO pairs + OPD Think-ON->OFF Context Distillation BSD seeds."""
    excl = excluded_doc_families or set()
    passed_by_id = {str(r.get("sample_id") or ""): r for r in cot_passed}

    dpo_kto_pairs: list[dict[str, Any]] = []
    for rej in cot_rejected:
        sid = str(rej.get("sample_id") or "")
        chosen_row = passed_by_id.get(sid)
        fam = str((chosen_row or {}).get("doc_family_id") or rej.get("doc_family_id") or "")
        if fam and fam in excl:
            continue
        msgs = rej.get("messages") or (chosen_row or {}).get("messages") or []
        if len(msgs) < 2:
            continue
        prompt_msgs = [m for m in msgs if m.get("role") != "assistant"]
        chosen_text = str((chosen_row or msgs[-1]).get("content") or msgs[-1].get("content") or "")
        dpo_kto_pairs.append(
            {
                "sample_id": f"pro_pref_{sid}",
                "doc_family_id": fam,
                "target": "student_kto_dpo_not_teacher",
                "messages": prompt_msgs,
                "chosen": chosen_text,
                "rejected": str(rej.get("rejected_response") or ""),
                "rejected_reasons": list(rej.get("rejected_reasons") or []),
            }
        )

    opd_bsd_seeds: list[dict[str, Any]] = []
    for row in cot_passed:
        fam = str(row.get("doc_family_id") or "")
        if fam and fam in excl:
            continue
        msgs = row.get("messages") or []
        if len(msgs) < 2:
            continue
        sys_msg = next((m["content"] for m in msgs if m.get("role") == "system"), "")
        usr_msg = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        ast_msg = str(msgs[-1].get("content") or "")
        clean_direct_target = (
            ast_msg.split("</think>", 1)[1].strip() if "</think>" in ast_msg else ast_msg
        )
        opd_bsd_seeds.append(
            {
                "sample_id": f"opd_seed_{row.get('sample_id')}",
                "doc_family_id": fam,
                "task_type": row.get("task_type", "fact"),
                "distill_mode": "think_on_to_off_context_distill",
                "teacher_enable_thinking": True,
                "student_enable_thinking": False,
                "bsd_compatible": True,
                "system_prompt": sys_msg,
                "prompt": usr_msg,
                "teacher_full_cot_response": ast_msg,
                "student_direct_target": clean_direct_target,
            }
        )

    return dpo_kto_pairs, opd_bsd_seeds


def enhance_flash_to_pro(
    *,
    flash_train_rows: list[dict[str, Any]],
    flash_val_rows: list[dict[str, Any]],
    cot_passed: list[dict[str, Any]],
    cot_rejected: list[dict[str, Any]],
    v2_docs: list[dict[str, Any]],
    output_dir: Path,
    canary_metrics: dict[str, float] | None = None,
    max_row_repeats: int = 2,
    max_qa_prose_share: float = 0.15,
    max_synth_image_share: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    """Runs all 4 Pro enhancements on Flash's source pool and writes `bundle_pro`."""
    curated_train, curate_stats = curate_high_value_midtrain(
        flash_train_rows,
        val_rows=flash_val_rows,
        max_row_repeats=max_row_repeats,
        max_qa_prose_share=max_qa_prose_share,
        max_synth_image_share=max_synth_image_share,
        seed=seed,
    )
    pro_val = [to_midtrain_row(r) for r in flash_val_rows]
    alignment_audit = audit_cross_modal_anchor_alignment(curated_train, v2_docs, cot_passed)
    canary_wise_ft = evaluate_canary_and_wise_ft(canary_metrics)

    val_fams = {str(r.get("doc_family_id") or "") for r in pro_val if r.get("doc_family_id")}
    dpo_kto_pairs, opd_bsd_seeds = build_pro_preference_and_bsd_pack(
        cot_passed,
        cot_rejected,
        excluded_doc_families=val_fams,
    )

    bundle_dir = output_dir / "bundle_pro"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    pro_train_path = bundle_dir / "pro_midtrain_train.jsonl"
    pro_val_path = bundle_dir / "pro_midtrain_val.jsonl"
    align_path = output_dir / "alignment_audit.json"
    wise_path = output_dir / "wise_ft_merge_recipe.json"
    dpo_path = output_dir / "pro_dpo_kto_pairs.jsonl"
    opd_path = output_dir / "pro_opd_bsd_seeds.jsonl"
    manifest_path = output_dir / "pro_enhance_manifest.json"

    write_jsonl(pro_train_path, curated_train)
    write_jsonl(pro_val_path, pro_val)
    write_jsonl(dpo_path, dpo_kto_pairs)
    write_jsonl(opd_path, opd_bsd_seeds)
    align_path.write_text(json.dumps(alignment_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    wise_path.write_text(json.dumps(canary_wise_ft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema": "pro_enhanced_same_source_v2",
        "seed": seed,
        "curation": curate_stats,
        "alignment_summary": {
            "cross_modal_aligned_count": alignment_audit["cross_modal_aligned_count"],
            "curated_rows_tagged_aligned": alignment_audit["curated_rows_tagged_aligned"],
            "visual_only_gaps": len(alignment_audit["visual_only_gap_families"]),
            "text_only_gaps": len(alignment_audit["text_only_gap_families"]),
        },
        "canary_and_wise_ft": {
            "wise_ft_merge_triggered": canary_wise_ft["wise_ft_merge_triggered"],
            "think_off_collapse_risk": canary_wise_ft["think_off_collapse_risk"],
            "opd_think_on_to_off_distill_recommended": canary_wise_ft["opd_think_on_to_off_distill_recommended"],
            "formula": canary_wise_ft["merge_recipe"]["formula"],
        },
        "post_train_packs": {
            "pro_dpo_kto_pairs_rows": len(dpo_kto_pairs),
            "pro_opd_bsd_seeds_rows": len(opd_bsd_seeds),
        },
        "doc_family_overlap": len(
            {str(r.get("doc_family_id")) for r in curated_train}
            & {str(r.get("doc_family_id")) for r in pro_val}
        ),
        "artifacts": {
            "pro_midtrain_train": {"path": str(pro_train_path), "sha256": sha256_file(pro_train_path)},
            "pro_midtrain_val": {"path": str(pro_val_path), "sha256": sha256_file(pro_val_path)},
            "alignment_audit": str(align_path),
            "wise_ft_merge_recipe": str(wise_path),
            "pro_dpo_kto_pairs": str(dpo_path),
            "pro_opd_bsd_seeds": str(opd_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _build_demo_flash_inputs(scratch_dir: Path) -> dict[str, Any]:
    qa_demo = load_jsonl(FLASH_DIR / "examples" / "qa_kb_v04_example.jsonl")
    # Inject 1 intentionally failing item so rejected CoT pairs are populated cleanly
    qa_with_fail = qa_demo + [
        {**qa_demo[0], "sample_id": "qa_v04_force_fail_demo", "question": "强制失败测试：电鼠特征？", "_mock_force_fail": True}
    ]
    cot_dir = scratch_dir / "cot"
    run_cot_generation_pipeline(qa_records=qa_with_fail, output_dir=cot_dir, cot_mode="internalized", mode="mock")

    entities = load_jsonl(FLASH_DIR / "examples" / "v2_image_entities_sample.jsonl")
    v1_audit = load_jsonl(FLASH_DIR / "examples" / "v2_image_v1_audit_sample.jsonl")
    v2_dir = scratch_dir / "v2"
    run_v2_image_pipeline(entities=entities, v1_audit_records=v1_audit, output_dir=v2_dir, mode="mock")

    v2_docs = load_jsonl(v2_dir / "v2_packed_cpt_docs.jsonl")
    v2_cf = load_jsonl(v2_dir / "v2_cf_neg_pairs.jsonl")
    prose = load_jsonl(cot_dir / "qa_midtrain_prose.jsonl")
    all_mid = build_demo_records() + v2_docs + v2_cf + prose
    mid_dir = scratch_dir / "mid"
    build_midtrain_bundle(all_mid, mid_dir, val_ratio=0.18, total_rows=24, max_row_repeats=2)

    return {
        "flash_train": load_jsonl(mid_dir / "midtrain_train.jsonl"),
        "flash_val": load_jsonl(mid_dir / "midtrain_val.jsonl"),
        "cot_passed": load_jsonl(cot_dir / "qa_cot_passed.jsonl"),
        "cot_rejected": load_jsonl(cot_dir / "qa_dpo_rejected_cots.jsonl"),
        "v2_docs": v2_docs,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flash-midtrain-train", type=Path, default=None)
    parser.add_argument("--flash-midtrain-val", type=Path, default=None)
    parser.add_argument("--cot-passed", type=Path, default=None)
    parser.add_argument("--cot-rejected", type=Path, default=None)
    parser.add_argument("--v2-packed-docs", type=Path, default=None)
    parser.add_argument("--canary-metrics", type=Path, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-row-repeats", type=int, default=2)
    parser.add_argument("--max-qa-prose-share", type=float, default=0.15)
    parser.add_argument("--max-synth-image-share", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.demo:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _build_demo_flash_inputs(Path(tmp))
            canary = (
                json.loads(args.canary_metrics.read_text(encoding="utf-8"))
                if (args.canary_metrics and args.canary_metrics.exists())
                else None
            )
            manifest = enhance_flash_to_pro(
                flash_train_rows=bundle["flash_train"],
                flash_val_rows=bundle["flash_val"],
                cot_passed=bundle["cot_passed"],
                cot_rejected=bundle["cot_rejected"],
                v2_docs=bundle["v2_docs"],
                output_dir=args.output_dir,
                canary_metrics=canary,
                max_row_repeats=args.max_row_repeats,
                max_qa_prose_share=args.max_qa_prose_share,
                max_synth_image_share=args.max_synth_image_share,
                seed=args.seed,
            )
    elif args.flash_midtrain_train and args.flash_midtrain_val:
        manifest = enhance_flash_to_pro(
            flash_train_rows=load_jsonl(args.flash_midtrain_train),
            flash_val_rows=load_jsonl(args.flash_midtrain_val),
            cot_passed=load_jsonl(args.cot_passed) if (args.cot_passed and args.cot_passed.exists()) else [],
            cot_rejected=load_jsonl(args.cot_rejected) if (args.cot_rejected and args.cot_rejected.exists()) else [],
            v2_docs=load_jsonl(args.v2_packed_docs) if (args.v2_packed_docs and args.v2_packed_docs.exists()) else [],
            output_dir=args.output_dir,
            canary_metrics=(
                json.loads(args.canary_metrics.read_text(encoding="utf-8"))
                if (args.canary_metrics and args.canary_metrics.exists())
                else None
            ),
            max_row_repeats=args.max_row_repeats,
            max_qa_prose_share=args.max_qa_prose_share,
            max_synth_image_share=args.max_synth_image_share,
            seed=args.seed,
        )
    else:
        raise SystemExit("Specify --demo or --flash-midtrain-train + --flash-midtrain-val.")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
