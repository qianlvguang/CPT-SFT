#!/usr/bin/env python3
"""Stage 3: Teacher DPO Preference Pair Builder & Final SFT Exporter.

Groups scored probing candidates by `prompt_group_id` (or `source_chunk_ids`) and
constructs preference pairs (prompt, chosen x+, rejected x-) for MS-SWIFT Teacher
DPO training:
  - Positive candidate x+ must have `programmatic_valid == True` and `influence.score > 0`.
  - Negative candidate x- has `influence.score < 0` (or the lowest score in the group)
    with `score(x+) - score(x-) >= min_margin`.

Also exports high-influence positive candidates directly in MS-SWIFT SFT `messages`
format (plus provenance manifest) for Student SFT training.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def normalize_teacher_prompt(prompt: str) -> str:
    """Strips optional per-call task_type prefix so prompt is identical for x+ and x-."""
    lines = prompt.strip().splitlines()
    if lines and lines[0].startswith("[指定优先题型:") and lines[0].endswith("]"):
        return "\n".join(lines[1:]).strip()
    return prompt.strip()


def build_dpo_and_sft_datasets(
    scored_candidates_path: Path,
    dpo_output_path: Path,
    sft_output_path: Path,
    provenance_output_path: Path,
    min_margin: float = 1e-6,
    require_negative_rejected: bool = True,
    max_pairs_per_group: int = 2,
) -> dict[str, Any]:
    """Constructs MS-SWIFT DPO pairs for Teacher and filtered SFT rows for Student."""
    records = load_jsonl(scored_candidates_path)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        gid = str(rec.get("prompt_group_id") or rec.get("source_chunk_ids", ["default"])[0])
        groups[gid].append(rec)

    dpo_rows: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []

    for gid, items in groups.items():
        # Sort descending by influence score
        sorted_items = sorted(
            items,
            key=lambda x: float(x.get("influence", {}).get("score", 0.0)),
            reverse=True,
        )

        # Positive pool for Student SFT and DPO chosen (must pass programmatic checks & score > 0)
        positives = [
            it
            for it in sorted_items
            if it.get("checks", {}).get("programmatic_valid", False)
            and float(it.get("influence", {}).get("score", 0.0)) > 0.0
        ]

        # Export the best positive item per source group for Student SFT
        if positives:
            best_pos = positives[0]
            sft_messages = best_pos.get("sft_messages", [])
            if sft_messages:
                row_idx = len(sft_rows)
                sft_rows.append({"messages": sft_messages})
                provenance_rows.append(
                    {
                        "sample_id": best_pos["sample_id"],
                        "row_index": row_idx,
                        "prompt_group_id": gid,
                        "source_doc_ids": best_pos.get("source_doc_ids", []),
                        "source_chunk_ids": best_pos.get("source_chunk_ids", []),
                        "task_type": best_pos.get("task_type"),
                        "citations": best_pos.get("citations", []),
                        "evidence": best_pos.get("evidence", []),
                        "influence_score": float(best_pos["influence"]["score"]),
                        "split": best_pos.get("split", "train"),
                    }
                )

        # Negative pool for DPO rejected
        if require_negative_rejected:
            negatives = [
                it
                for it in reversed(sorted_items)
                if float(it.get("influence", {}).get("score", 0.0)) < 0.0
            ]
        else:
            negatives = list(reversed(sorted_items))

        pairs_created = 0
        for pos in positives:
            if pairs_created >= max_pairs_per_group:
                break
            pos_score = float(pos["influence"]["score"])
            for neg in negatives:
                if pos["sample_id"] == neg["sample_id"]:
                    continue
                neg_score = float(neg["influence"]["score"])
                margin = pos_score - neg_score
                if margin < min_margin:
                    continue
                if str(pos.get("teacher_response", "")).strip() == str(neg.get("teacher_response", "")).strip():
                    continue

                shared_prompt = normalize_teacher_prompt(str(pos.get("teacher_prompt", "")))
                dpo_rows.append(
                    {
                        "messages": [
                            {"role": "user", "content": shared_prompt},
                            {"role": "assistant", "content": str(pos["teacher_response"]).strip()},
                        ],
                        "rejected_response": str(neg["teacher_response"]).strip(),
                        "meta": {
                            "prompt_group_id": gid,
                            "chosen_sample_id": pos["sample_id"],
                            "rejected_sample_id": neg["sample_id"],
                            "chosen_influence": pos_score,
                            "rejected_influence": neg_score,
                            "influence_margin": round(margin, 8),
                        },
                    }
                )
                pairs_created += 1
                break

    for path in (dpo_output_path, sft_output_path, provenance_output_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    with dpo_output_path.open("w", encoding="utf-8") as f:
        for row in dpo_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with sft_output_path.open("w", encoding="utf-8") as f:
        for row in sft_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with provenance_output_path.open("w", encoding="utf-8") as f:
        for row in provenance_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "total_groups": len(groups),
        "dpo_pairs_count": len(dpo_rows),
        "student_sft_count": len(sft_rows),
        "dpo_output_path": str(dpo_output_path),
        "sft_output_path": str(sft_output_path),
        "provenance_output_path": str(provenance_output_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-candidates", type=Path, required=True, help="Scored candidates JSONL")
    parser.add_argument("--dpo-output", type=Path, required=True, help="Output MS-SWIFT Teacher DPO JSONL")
    parser.add_argument("--sft-output", type=Path, required=True, help="Output MS-SWIFT Student SFT JSONL")
    parser.add_argument("--provenance-output", type=Path, required=True, help="Output provenance manifest JSONL")
    parser.add_argument("--min-margin", type=float, default=1e-6)
    parser.add_argument(
        "--allow-non-negative-rejected",
        action="store_true",
        help="Allow rejected candidates with score >= 0 as long as margin >= min_margin",
    )
    parser.add_argument("--max-pairs-per-group", type=int, default=2)
    args = parser.parse_args(argv)

    summary = build_dpo_and_sft_datasets(
        scored_candidates_path=args.scored_candidates,
        dpo_output_path=args.dpo_output,
        sft_output_path=args.sft_output,
        provenance_output_path=args.provenance_output,
        min_margin=args.min_margin,
        require_negative_rejected=not args.allow_non_negative_rejected,
        max_pairs_per_group=args.max_pairs_per_group,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

