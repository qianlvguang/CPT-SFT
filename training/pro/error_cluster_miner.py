#!/usr/bin/env python3
"""Phase C (Pro) Error-Cluster Miner, Fixed-Weight Gatekeeper & 50/30/20 Incremental Builder.

Implements `docs/training_plans/online_ip_template_cpt.md` (§Pro) and
`docs/training_plans/pro/pro_closed_loop_plan.md`:
  1. Classifies evaluation/production error logs into 4 gap modes:
       - `missing_index_evidence`: missing reference image, alias, or rule -> emits
         index/retrieval refill prescription (`requires_weight_update=False`).
       - `style_collision_false_positive`: same-style original confused with derivative IP ->
         constructs contrastive CPT pairs (`requires_weight_update=True`, target=CPT).
       - `insufficient_forced_judgment`: forced verdict when evidence is missing/occluded ->
         constructs paired `evidence_present vs evidence_stripped` rows (domain -> CPT,
         uncertainty expression -> cross-template SFT).
       - `disposition_mapping_drift`: recognition right but disposition/route wrong ->
         emits template routing fix (`requires_weight_update=False`).
  2. Builds a 50/30/20 incremental CPT package (`pro_train.jsonl` & `pro_val.jsonl`):
       - 50% newly verified gap cases (`verified=True`)
       - 30% accepted online template replay (`accepted_templates`)
       - 20% independent V-series historical replay (`v_replay`)
  3. Enforces strict `sft_overlap == 0` cross-template SFT isolation across
     `template_family_id` and `doc_family_id`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

IMAGE_PLACEHOLDER = "<image>"
ALLOWED_IMAGE_PREFIX_RE = re.compile(
    r"^(?:(?:图\s*\d+|待审图|商品图|参考图|原图|对比图|Image\s*\d+)\s*[：:]?\s*)*$"
)

TARGET_PRO_CPT_SHARES = {
    "new_verified_gap": 0.50,
    "accepted_template_replay": 0.30,
    "v_series_replay": 0.20,
}

GAP_CATEGORIES = (
    "missing_index_evidence",
    "style_collision_false_positive",
    "insufficient_forced_judgment",
    "disposition_mapping_drift",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _format_image_prefix(num_images: int) -> str:
    if num_images <= 0:
        return ""
    if num_images == 1:
        return "<image>\n"
    return "\n".join(f"图{i}：<image>" for i in range(1, num_images + 1)) + "\n"


def _validate_cpt_row(rec: dict[str, Any]) -> None:
    msgs = rec.get("messages") or []
    if len(msgs) != 1 or msgs[0].get("role") != "assistant":
        raise ValueError(f"Invalid Pro CPT row {rec.get('sample_id')}: must have 1 assistant message")
    content = str(msgs[0].get("content") or "")
    images = rec.get("images") or []
    if content.count(IMAGE_PLACEHOLDER) != len(images):
        raise ValueError(f"Image placeholder mismatch in {rec.get('sample_id')}")
    if images:
        first_idx = content.find(IMAGE_PLACEHOLDER)
        prefix = content[:first_idx].strip()
        if prefix and not ALLOWED_IMAGE_PREFIX_RE.fullmatch(prefix):
            raise ValueError(f"Causal prefix violation in {rec.get('sample_id')}")


def classify_error_gap(record: dict[str, Any]) -> str:
    """Maps a raw error record into one of the 4 canonical Pro gap clusters."""
    err_type = str(record.get("error_type") or record.get("gap_type") or "").strip().lower()
    if err_type in GAP_CATEGORIES:
        return err_type
    if any(k in err_type for k in ("missing", "index", "alias", "recall", "retrieval")):
        return "missing_index_evidence"
    if any(k in err_type for k in ("style", "collision", "false_positive", "original")):
        return "style_collision_false_positive"
    if any(k in err_type for k in ("insufficient", "forced", "uncertain", "occlud")):
        return "insufficient_forced_judgment"
    if any(k in err_type for k in ("disposition", "mapping", "route", "action", "version")):
        return "disposition_mapping_drift"
    return "missing_index_evidence"


def _template_family_of(rec: dict[str, Any]) -> str:
    return str(
        rec.get("template_family_id")
        or rec.get("template_id")
        or rec.get("doc_family_id")
        or "unknown_family"
    )


def _doc_family_of(rec: dict[str, Any]) -> str:
    return str(rec.get("doc_family_id") or _template_family_of(rec))


def mine_error_clusters(
    error_logs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Triages error logs into fixed-weight prescriptions vs verified CPT/SFT repairs."""
    cluster_counts: Counter[str] = Counter()
    index_prescriptions: list[dict[str, Any]] = []
    cpt_gap_rows: list[dict[str, Any]] = []
    sft_uncertainty_rows: list[dict[str, Any]] = []
    unverified_skipped = 0

    for idx, rec in enumerate(error_logs):
        gap = classify_error_gap(rec)
        cluster_counts[gap] += 1
        sid = str(rec.get("sample_id") or f"err_{idx}")
        tpl_fam = _template_family_of(rec)
        doc_fam = _doc_family_of(rec)

        if gap == "missing_index_evidence":
            index_prescriptions.append(
                {
                    "sample_id": sid,
                    "gap_category": gap,
                    "requires_weight_update": False,
                    "action": "top_k_index_and_alias_refill",
                    "template_id": rec.get("template_id"),
                    "template_family_id": tpl_fam,
                    "query_entity": rec.get("query_entity", ""),
                    "candidate_aliases": rec.get("candidate_aliases", []),
                    "missing_rule_version": rec.get("missing_rule_version", ""),
                    "next_step": "Verify fixed-weight retrieval recall after index patch before launching CPT.",
                }
            )
            continue

        if gap == "disposition_mapping_drift":
            index_prescriptions.append(
                {
                    "sample_id": sid,
                    "gap_category": gap,
                    "requires_weight_update": False,
                    "action": "template_disposition_mapping_fix",
                    "template_id": rec.get("template_id"),
                    "template_family_id": tpl_fam,
                    "recognized_relation": rec.get("recognized_relation", ""),
                    "predicted_action": rec.get("predicted_action", ""),
                    "expected_action": rec.get("expected_action", ""),
                    "template_version": rec.get("template_version", ""),
                    "next_step": "Align template output mapping / version config and re-eval with frozen weights.",
                }
            )
            continue

        if not rec.get("verified", False):
            unverified_skipped += 1
            continue

        images = [str(x) for x in (rec.get("images") or []) if str(x).strip()]
        prefix = _format_image_prefix(len(images))

        if gap == "style_collision_false_positive":
            shared = str(rec.get("shared_features") or "局部风格或常见元素相近").strip()
            clues = str(rec.get("distinguishing_clues") or "缺少关键排他性标识组合").strip()
            relation = str(rec.get("gold_relation") or "已确认原创，并非二创").strip()
            content = (
                f"{prefix}[风格碰撞正负对照]\n"
                f"共同表层线索：{shared}。\n"
                f"核心排他差异：{clues}。\n"
                f"核验关系结论：{relation}。仅共享画风、常见服饰或通用几何轮廓不足以建立侵权关联。"
            )
            row = {
                "sample_id": f"pro_gap_{sid}",
                "doc_family_id": doc_fam,
                "template_id": rec.get("template_id"),
                "template_family_id": tpl_fam,
                "channel": "pro_new_verified_gap",
                "pro_bucket": "new_verified_gap",
                "gap_category": gap,
                "images": images,
                "messages": [{"role": "assistant", "content": content}],
            }
            _validate_cpt_row(row)
            cpt_gap_rows.append(row)

        elif gap == "insufficient_forced_judgment":
            ev_present = str(
                rec.get("evidence_present_conclusion") or "具备完整辨识特征组合，依据有效来源确认命中。"
            ).strip()
            ev_stripped = str(
                rec.get("evidence_stripped_conclusion") or "关键辨识区域缺失或未检索到归属依据，现有证据不足以判定。"
            ).strip()
            missing_slot = str(rec.get("missing_slot") or "关键识别部位或授权状态").strip()
            content = (
                f"{prefix}[证据存在与抹除边界对照]\n"
                f"证据充分情形：{ev_present}\n"
                f"证据抹除/缺失情形（缺失槽位：{missing_slot}）：{ev_stripped}\n"
                f"边界纪律：当决定性视觉线索或检索条文缺失时，禁止强判二创或侵权，必须保持证据不足。"
            )
            row = {
                "sample_id": f"pro_gap_{sid}",
                "doc_family_id": doc_fam,
                "template_id": rec.get("template_id"),
                "template_family_id": tpl_fam,
                "channel": "pro_new_verified_gap",
                "pro_bucket": "new_verified_gap",
                "gap_category": gap,
                "images": images,
                "messages": [{"role": "assistant", "content": content}],
            }
            _validate_cpt_row(row)
            cpt_gap_rows.append(row)

            # Paired uncertainty expression sample marked for cross-template SFT isolation
            sft_uncertainty_rows.append(
                {
                    "sample_id": f"pro_sft_uncert_{sid}",
                    "doc_family_id": f"sft_uncert_{doc_fam}",
                    "template_id": "cross_template_uncertainty_expression",
                    "template_family_id": "cross_template_uncertainty_expression",
                    "channel": "cross_template_uncertainty_sft",
                    "enable_thinking": True,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"待核验对象存在「{missing_slot}」缺失且无有效补充条文时，应如何规范输出裁决边界？",
                        },
                        {
                            "role": "assistant",
                            "content": (
                                f"现有证据不足以闭合判定。由于决定性字段（{missing_slot}）未披露或处于遮挡状态，"
                                f"无法确认其是否满足规则命中条件；若后续补齐该证据再按对应规则复核，当前应保持不确定并明确列出缺失槽位。"
                            ),
                        },
                    ],
                }
            )

    total_errors = len(error_logs) or 1
    weight_update_needed = len(cpt_gap_rows)
    return {
        "cluster_counts": dict(sorted(cluster_counts.items())),
        "index_prescriptions": index_prescriptions,
        "cpt_gap_rows": cpt_gap_rows,
        "sft_uncertainty_rows": sft_uncertainty_rows,
        "unverified_skipped": unverified_skipped,
        "weight_update_required_ratio": round(weight_update_needed / total_errors, 4),
    }


def _split_train_val_zero_leak(
    records: list[dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        buckets[_doc_family_of(r)].append(r)
    families = sorted(buckets.keys())
    if len(families) < 2 or val_ratio <= 0.0:
        return list(records), []
    rng = random.Random(seed)
    shuffled = list(families)
    rng.shuffle(shuffled)
    val_cnt = max(1, min(len(families) - 1, int(round(len(families) * val_ratio))))
    val_fams = set(shuffled[:val_cnt])
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for fam in families:
        (val_rows if fam in val_fams else train_rows).extend(buckets[fam])
    assert not ({_doc_family_of(r) for r in train_rows} & {_doc_family_of(r) for r in val_rows})
    return train_rows, val_rows


def _allocate_50_30_20_mixture(
    new_gap_rows: list[dict[str, Any]],
    accepted_rows: list[dict[str, Any]],
    v_replay_rows: list[dict[str, Any]],
    *,
    total_rows: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pools = {
        "new_verified_gap": [dict(r, pro_bucket="new_verified_gap") for r in new_gap_rows],
        "accepted_template_replay": [
            dict(r, pro_bucket="accepted_template_replay") for r in accepted_rows
        ],
        "v_series_replay": [dict(r, pro_bucket="v_series_replay") for r in v_replay_rows],
    }
    present = [k for k, v in pools.items() if v]
    if not present:
        return [], {}

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    if total_rows is None:
        # Infer largest integer batch size matching 50/30/20 without excessive oversampling
        base_size = max(len(pools[k]) for k in present) * len(present)
        total_rows = max(base_size, 10)

    weight_sum = sum(TARGET_PRO_CPT_SHARES[k] for k in present) or 1.0
    counts: dict[str, int] = {}
    for k in present:
        cnt = max(1, int(round(total_rows * (TARGET_PRO_CPT_SHARES[k] / weight_sum))))
        counts[k] = cnt
        pool = pools[k]
        if cnt <= len(pool):
            picks = list(pool)
            rng.shuffle(picks)
            selected.extend(picks[:cnt])
        else:
            selected.extend(pool)
            selected.extend(rng.choices(pool, k=cnt - len(pool)))

    rng.shuffle(selected)
    total_sel = len(selected) or 1
    actual_shares = {k: round(counts[k] / total_sel, 4) for k in sorted(counts.keys())}
    return selected, actual_shares


def run_pro_pipeline(
    *,
    error_logs: list[dict[str, Any]],
    accepted_templates: list[dict[str, Any]],
    v_replay: list[dict[str, Any]],
    sft_candidates: list[dict[str, Any]],
    output_dir: Path,
    total_cpt_rows: int | None = None,
    val_ratio: float = 0.2,
    round_id: str = "round_01",
    parent_ckpt: str = "",
    prev_ledger: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Runs Pro gap mining, 50/30/20 incremental CPT bundling, and monotonic `sft_overlap=0` SFT isolation."""
    prev_tpl_fams: set[str] = set()
    prev_doc_fams: set[str] = set()
    prev_rounds: list[dict[str, Any]] = []
    if prev_ledger and prev_ledger.exists():
        ledger_data = json.loads(prev_ledger.read_text(encoding="utf-8"))
        prev_tpl_fams = set(ledger_data.get("cumulative_excluded_template_families") or [])
        prev_doc_fams = set(ledger_data.get("cumulative_excluded_doc_families") or [])
        prev_rounds = list(ledger_data.get("rounds") or [])

    triaged = mine_error_clusters(error_logs)
    cpt_gap_rows = triaged["cpt_gap_rows"]

    for r in accepted_templates:
        _validate_cpt_row(r)
    for r in v_replay:
        _validate_cpt_row(r)

    all_cpt_sources = cpt_gap_rows + accepted_templates + v_replay
    train_pool, val_rows = _split_train_val_zero_leak(
        all_cpt_sources,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_gaps = [r for r in train_pool if r.get("pro_bucket") == "new_verified_gap"]
    train_acc = [r for r in train_pool if r not in train_gaps and r in accepted_templates]
    train_vrep = [r for r in train_pool if r not in train_gaps and r in v_replay]

    pro_train, actual_shares = _allocate_50_30_20_mixture(
        train_gaps,
        train_acc,
        train_vrep,
        total_rows=total_cpt_rows,
        seed=seed,
    )

    # Strict monotonic SFT isolation: exclude ANY template_family_id or doc_family_id from current or prior rounds
    round_tpl_fams = {_template_family_of(r) for r in all_cpt_sources} | {
        _template_family_of(r) for r in error_logs
    }
    round_doc_fams = {_doc_family_of(r) for r in all_cpt_sources}
    cpt_excluded_tpl_fams = prev_tpl_fams | round_tpl_fams
    cpt_excluded_doc_fams = prev_doc_fams | round_doc_fams

    sft_isolated: list[dict[str, Any]] = []
    sft_excluded_count = 0
    for cand in list(sft_candidates) + list(triaged["sft_uncertainty_rows"]):
        tfam = _template_family_of(cand)
        dfam = _doc_family_of(cand)
        if tfam in cpt_excluded_tpl_fams or dfam in cpt_excluded_doc_fams:
            sft_excluded_count += 1
            continue
        sft_isolated.append(cand)

    sft_tpl_fams = {_template_family_of(r) for r in sft_isolated}
    sft_doc_fams = {_doc_family_of(r) for r in sft_isolated}
    sft_overlap = len(sft_tpl_fams & cpt_excluded_tpl_fams) + len(sft_doc_fams & cpt_excluded_doc_fams)
    assert sft_overlap == 0, (
        f"SFT isolation violation: tpl={sft_tpl_fams & cpt_excluded_tpl_fams}, "
        f"doc={sft_doc_fams & cpt_excluded_doc_fams}"
    )

    bundle_dir = output_dir / "bundle_pro_inc"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    prescriptions_path = output_dir / "index_prescriptions.jsonl"
    pro_train_path = bundle_dir / "pro_train.jsonl"
    pro_val_path = bundle_dir / "pro_val.jsonl"
    pro_sft_path = output_dir / "pro_sft_isolated.jsonl"
    manifest_path = output_dir / "pro_manifest.json"
    ledger_path = output_dir / "iteration_ledger.json"

    write_jsonl(prescriptions_path, triaged["index_prescriptions"])
    write_jsonl(pro_train_path, pro_train)
    write_jsonl(pro_val_path, val_rows)
    write_jsonl(pro_sft_path, sft_isolated)

    round_entry = {
        "round_id": round_id,
        "parent_ckpt": parent_ckpt,
        "pro_cpt_train_rows": len(pro_train),
        "pro_cpt_val_rows": len(val_rows),
        "round_new_tpl_families": len(round_tpl_fams - prev_tpl_fams),
        "round_new_doc_families": len(round_doc_fams - prev_doc_fams),
        "cumulative_excluded_tpl_families_count": len(cpt_excluded_tpl_fams),
        "cumulative_excluded_doc_families_count": len(cpt_excluded_doc_fams),
        "sft_isolated_rows": len(sft_isolated),
        "sft_overlap": sft_overlap,
    }
    ledger = {
        "schema": "pro_iteration_ledger_v1",
        "latest_round_id": round_id,
        "parent_ckpt": parent_ckpt,
        "cumulative_excluded_template_families": sorted(cpt_excluded_tpl_fams),
        "cumulative_excluded_doc_families": sorted(cpt_excluded_doc_fams),
        "rounds": prev_rounds + [round_entry],
    }
    ledger_path.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema": "cpt_27b_pro_incremental_v1",
        "round_id": round_id,
        "parent_ckpt": parent_ckpt,
        "seed": seed,
        "cluster_counts": triaged["cluster_counts"],
        "index_prescriptions_count": len(triaged["index_prescriptions"]),
        "weight_update_required_ratio": triaged["weight_update_required_ratio"],
        "unverified_skipped": triaged["unverified_skipped"],
        "pro_cpt_train_rows": len(pro_train),
        "pro_cpt_val_rows": len(val_rows),
        "target_cpt_mix_shares": TARGET_PRO_CPT_SHARES,
        "actual_cpt_mix_shares": actual_shares,
        "cpt_excluded_families_count": len(cpt_excluded_tpl_fams),
        "cumulative_excluded_doc_families_count": len(cpt_excluded_doc_fams),
        "sft_isolated_rows": len(sft_isolated),
        "sft_excluded_rows": sft_excluded_count,
        "sft_overlap": sft_overlap,
        "artifacts": {
            "index_prescriptions": {
                "path": str(prescriptions_path),
                "sha256": sha256_file(prescriptions_path),
            },
            "pro_train": {
                "path": str(pro_train_path),
                "sha256": sha256_file(pro_train_path),
            },
            "pro_val": {
                "path": str(pro_val_path),
                "sha256": sha256_file(pro_val_path),
            },
            "pro_sft_isolated": {
                "path": str(pro_sft_path),
                "sha256": sha256_file(pro_sft_path),
            },
            "iteration_ledger": {
                "path": str(ledger_path),
                "sha256": sha256_file(ledger_path),
            },
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_demo_records() -> dict[str, list[dict[str, Any]]]:
    demo_path = Path(__file__).resolve().parent / "examples" / "pro_error_clusters_sample.jsonl"
    raw = load_jsonl(demo_path)
    return {
        "error_logs": [r for r in raw if r.get("record_kind") == "error_log"],
        "accepted_templates": [r for r in raw if r.get("record_kind") == "accepted_template"],
        "v_replay": [r for r in raw if r.get("record_kind") == "v_replay"],
        "sft_candidates": [r for r in raw if r.get("record_kind") == "sft_candidate"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--error-logs", type=Path, default=None)
    parser.add_argument("--accepted-templates", type=Path, default=None)
    parser.add_argument("--v-replay", type=Path, default=None)
    parser.add_argument("--sft-candidates", type=Path, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-cpt-rows", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--round-id", default="round_01")
    parser.add_argument("--parent-ckpt", default="")
    parser.add_argument("--prev-ledger", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.demo:
        inputs = load_demo_records()
    elif args.error_logs:
        inputs = {
            "error_logs": load_jsonl(args.error_logs),
            "accepted_templates": load_jsonl(args.accepted_templates) if args.accepted_templates else [],
            "v_replay": load_jsonl(args.v_replay) if args.v_replay else [],
            "sft_candidates": load_jsonl(args.sft_candidates) if args.sft_candidates else [],
        }
    else:
        raise SystemExit("Specify --demo or --error-logs.")

    manifest = run_pro_pipeline(
        error_logs=inputs["error_logs"],
        accepted_templates=inputs["accepted_templates"],
        v_replay=inputs["v_replay"],
        sft_candidates=inputs["sft_candidates"],
        output_dir=args.output_dir,
        total_cpt_rows=args.total_cpt_rows,
        val_ratio=args.val_ratio,
        round_id=args.round_id,
        parent_ckpt=args.parent_ckpt,
        prev_ledger=args.prev_ledger,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
