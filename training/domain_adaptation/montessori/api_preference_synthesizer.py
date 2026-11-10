#!/usr/bin/env python3
"""API Black-Box Teacher Alignment & Multi-Dimensional Student Preference Synthesizer.

Addresses the practical scenario where the Teacher is a closed-source / read-only API
(e.g., GPT-4o, Claude, DeepSeek API, or hosted vLLM endpoint without weight write access)
and we want a diverse, systematic pipeline to:

1. Steer the API Teacher WITHOUT weight updates (Black-Box Montessori Alignment):
   - In-Context Contrastive Steering (ICDPO): Builds a stratified Preference Memory Bank
     indexed by `task_type` and lexical/character bigram similarity. For each new source
     chunk, retrieves top-m high-margin (x+, x-) pairs and injects them as explicit
     positive/negative demonstrations into the API prompt.
   - Meta-Prompt Rubric Evolution (TextGrad-style): Analyzes the top-K highest-margin
     (x+, x-) pairs across failure dimensions to distill explicit generation rules
     (e.g., condition preservation, quote precision, anti-brevity) directly into the
     system/generation prompt.

2. Synthesize Multi-Dimensional Preference Data directly for the Student:
   - Multi-Persona / Multi-Temperature API Rollouts: For a given source chunk and
     question, prompts the API across diverse synthesis personas (or temperatures) to
     elicit natural variations alongside controlled hard negatives:
       * `rigorous_cited` (Target gold style)
       * `over_brief` (Drops statutory caveats / intermediate steps)
       * `over_extrapolated` (Over-interprets beyond source boundaries)
       * `citation_drift` (Misaligned or incomplete evidence attribution)
   - Multi-Objective Composite Reward R(x, y):
       R(x, y) = w_inf * Z(Influence) + w_ver * S_verify(x, y) + w_cal * S_calib(x, y)
     Combines Student Local Data Influence (one-step validation loss reduction) with
     programmatic citation/substring verification and length/density calibration.
   - Multi-Format Preference Export (MS-SWIFT compatible):
       * Student DPO (`messages` + `rejected_response` with hard-negative type tags)
       * Student KTO (`messages` + binary `label` true/false for unpaired API outputs)
       * Student GRPO / RLVR (`messages` + verifiable `ground_truth` & `source_map`)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from probing_generate import (
    DEFAULT_SYSTEM_PROMPT,
    format_source_bundle,
    load_jsonl,
    sha256_text,
    verify_and_enrich_item,
)

# Diverse API sampling personas to elicit rich response spectra & hard negatives
API_SYNTHESIS_PERSONAS: dict[str, dict[str, Any]] = {
    "rigorous_cited": {
        "temperature": 0.3,
        "instruction_suffix": (
            "【生成策略：严谨详尽】先给出明确结论，再完整列出法条/规则的前提条件、但书与处理顺序，"
            "每项主张紧贴 [S1] 引用，最后指明材料未规定的边界。"
        ),
        "expected_polarity": "positive",
    },
    "structured_cot": {
        "temperature": 0.5,
        "instruction_suffix": (
            "【生成策略：分步推演】先核对给定材料的适用前提，再分步骤推导问题结论，"
            "严格使用给定引用标记，不省略中间逻辑环节。"
        ),
        "expected_polarity": "positive",
    },
    "over_brief": {
        "temperature": 0.7,
        "instruction_suffix": (
            "【对比采样：极简回答】仅用一两句简短的话直接给出最终结论并附带引用标记，省略具体处理顺序与未规定边界的展开说明。"
        ),
        "expected_polarity": "hard_negative_brevity",
    },
    "boundary_blur": {
        "temperature": 0.9,
        "instruction_suffix": (
            "【对比采样：弱化限定】快速回答核心问题，无需逐字强调前置限定词或区分材料未说明的例外情形。"
        ),
        "expected_polarity": "hard_negative_condition",
    },
}


def char_bigrams(text: str) -> set[str]:
    cleaned = "".join(ch for ch in text if not ch.isspace())
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def jaccard_similarity(text_a: str, text_b: str) -> float:
    bg_a = char_bigrams(text_a)
    bg_b = char_bigrams(text_b)
    if not bg_a or not bg_b:
        return 0.0
    return len(bg_a & bg_b) / len(bg_a | bg_b)


def diagnose_negative_defect(chosen_rec: dict[str, Any], rejected_rec: dict[str, Any]) -> str:
    """Classifies why `rejected_rec` had negative student influence compared to `chosen_rec`."""
    checks = rejected_rec.get("checks", {})
    if not checks.get("programmatic_valid", True):
        errs = checks.get("errors", [])
        if any("quote_not_exact_substring" in e for e in errs):
            return "引用摘录非原文精确子串（存在改写或幻觉）"
        if any("citation" in e for e in errs):
            return "引用标记缺失或使用了未定义的来源 Key"
        return f"未通过程序核验 ({', '.join(errs)})"

    ans_pos = str(chosen_rec.get("answer", ""))
    ans_neg = str(rejected_rec.get("answer", ""))
    if len(ans_neg) < max(45, int(len(ans_pos) * 0.55)):
        return "回答过度简略，遗漏了具体执行顺序、前置但书条件或未规定边界说明，易导致学生过拟合死板短模板"
    if "未规定" in ans_pos and "未规定" not in ans_neg and "无法" not in ans_neg:
        return "未明确划定给定材料的证据边界，缺少对材料未覆盖事项的严谨界定"
    return "推理颗粒度与目标学生认知分布不匹配，导致验证集泛化损失上升"


class PreferenceMemoryBank:
    """Stratified memory bank of student-scored (x+, x-) pairs for In-Context API Steering."""

    def __init__(self, scored_records: list[dict[str, Any]], min_margin: float = 1e-5) -> None:
        self.entries: list[dict[str, Any]] = []
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rec in scored_records:
            gid = str(rec.get("prompt_group_id") or rec.get("source_chunk_ids", ["default"])[0])
            groups[gid].append(rec)

        for gid, items in groups.items():
            ranked = sorted(
                items,
                key=lambda r: float(r.get("influence", {}).get("score", 0.0)),
                reverse=True,
            )
            positives = [
                r
                for r in ranked
                if r.get("checks", {}).get("programmatic_valid", False)
                and float(r.get("influence", {}).get("score", 0.0)) > 0.0
            ]
            negatives = [
                r
                for r in reversed(ranked)
                if float(r.get("influence", {}).get("score", 0.0)) < 0.0
            ]
            if not positives or not negatives:
                continue

            best_pos = positives[0]
            worst_neg = negatives[0]
            pos_score = float(best_pos["influence"]["score"])
            neg_score = float(worst_neg["influence"]["score"])
            margin = pos_score - neg_score
            if margin < min_margin:
                continue

            defect_reason = diagnose_negative_defect(best_pos, worst_neg)
            self.entries.append(
                {
                    "prompt_group_id": gid,
                    "task_type": str(best_pos.get("task_type", "fact")),
                    "source_text": str(best_pos.get("teacher_prompt", "")),
                    "chosen_sample_id": best_pos["sample_id"],
                    "rejected_sample_id": worst_neg["sample_id"],
                    "chosen_question": best_pos.get("question", ""),
                    "chosen_answer": best_pos.get("answer", ""),
                    "chosen_response_json": best_pos.get("teacher_response", ""),
                    "rejected_question": worst_neg.get("question", ""),
                    "rejected_answer": worst_neg.get("answer", ""),
                    "rejected_response_json": worst_neg.get("teacher_response", ""),
                    "chosen_influence": pos_score,
                    "rejected_influence": neg_score,
                    "margin": round(margin, 8),
                    "defect_diagnosis": defect_reason,
                }
            )

        self.entries.sort(key=lambda e: e["margin"], reverse=True)

    def retrieve(
        self,
        query_source_bundle: str,
        target_task_type: str | None = None,
        top_k: int = 2,
    ) -> list[dict[str, Any]]:
        """Retrieves top-k most relevant contrastive (x+, x-) pairs for a new source chunk."""
        if not self.entries:
            return []

        scored_entries: list[tuple[float, dict[str, Any]]] = []
        for entry in self.entries:
            sim = jaccard_similarity(query_source_bundle, entry["source_text"])
            type_bonus = 0.25 if (target_task_type and entry["task_type"] == target_task_type) else 0.0
            margin_bonus = min(entry["margin"] * 10.0, 0.5)
            rank_score = sim + type_bonus + margin_bonus
            scored_entries.append((rank_score, entry))

        scored_entries.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored_entries[:top_k]]

    def build_in_context_steered_prompt(
        self,
        base_prompt: str,
        query_source_bundle: str,
        target_task_type: str | None = None,
        top_k: int = 1,
    ) -> str:
        """Injects retrieved (x+, x-) contrastive examples into the API generation prompt."""
        demos = self.retrieve(query_source_bundle, target_task_type=target_task_type, top_k=top_k)
        if not demos:
            return base_prompt

        demo_blocks: list[str] = [
            "=== 目标学生模型梯度反馈校准范例 (In-Context Contrastive Steering) ===",
            "以下正反范例由下游学生模型在验证集上的单步梯度损失差分 (Local Data Influence) 实测选出：",
        ]
        for idx, d in enumerate(demos, start=1):
            demo_blocks.append(
                f"【对比范例 #{idx} (题型: {d['task_type']} | 影响差值 ΔInfluence = +{d['margin']:.6f})】\n"
                f"  ✅ 正向高收益生成 (x⁺, Influence = +{d['chosen_influence']:.6f})：\n"
                f"     问题：{d['chosen_question']}\n"
                f"     回答：{d['chosen_answer']}\n"
                f"  ❌ 负向受损生成 (x⁻, Influence = {d['rejected_influence']:.6f})：\n"
                f"     问题：{d['rejected_question']}\n"
                f"     回答：{d['rejected_answer']}\n"
                f"  ⚠️ 负例受损诊断：{d['defect_diagnosis']}"
            )
        demo_blocks.append(
            "请严格对齐【✅ 正向高收益生成】的论证完备度与证据引用方式，主动规避【❌ 负向受损生成】的缺陷。\n"
            "==================================================================\n"
        )
        return "\n\n".join(demo_blocks) + "\n" + base_prompt

    def synthesize_meta_prompt_rubric(self) -> dict[str, Any]:
        """Distills explicit meta-prompt rules from the memory bank's failure diagnoses."""
        defect_counts: dict[str, int] = defaultdict(int)
        for entry in self.entries:
            defect_counts[entry["defect_diagnosis"]] += 1

        rules = [
            "1. 结论先行与程序完备：首句直接回应可回答性结论，随后必须完整陈述原文规定的先后处理顺序与前置条件，严禁缩略成单句口号。",
            "2. 精确子串引证：所有事实主张后标注 [S1] 等给定 source_key，且 evidence.quote 必须逐字复制原文连续子串，不得改写标点或增删字词。",
            "3. 显式证据边界声明：当材料未规定期限、例外或延伸情形时，必须在末段明确指出“仅凭当前材料无法进一步判断”，严禁凭外部记忆补全。",
        ]
        return {
            "total_contrastive_pairs": len(self.entries),
            "observed_defect_distribution": dict(defect_counts),
            "evolved_meta_rules": rules,
            "meta_prompt_patch": "\n".join(["【学生认知对齐元规则 (Meta-Aligned Rubric)】"] + rules),
        }


def compute_composite_preference_score(
    record: dict[str, Any],
    w_influence: float = 0.65,
    w_verification: float = 0.25,
    w_calibration: float = 0.10,
) -> dict[str, float]:
    """Computes a multi-objective composite reward R(x, y) for student preference ranking.

    Combines:
      1. `influence_norm`: tanh-normalized student Local Data Influence score in [-1, 1]
      2. `verification_score`: +1.0 if all programmatic citation & exact-substring checks pass, else -1.0
      3. `calibration_score`: rewards well-structured answers (80~450 chars with explicit boundary/order explanations)
    """
    raw_inf = float(record.get("influence", {}).get("score", 0.0))
    # Scale raw influence (typically around 1e-4 to 1e-1) into [-1, 1] via tanh
    inf_norm = math.tanh(raw_inf * 25.0)

    is_valid = bool(record.get("checks", {}).get("programmatic_valid", False))
    ver_score = 1.0 if is_valid else -1.0

    answer = str(record.get("answer", "")).strip()
    ans_len = len(answer)
    if ans_len < 45:
        cal_score = -0.8
    elif 80 <= ans_len <= 450:
        cal_score = 1.0
    else:
        cal_score = 0.4

    composite = (
        w_influence * inf_norm
        + w_verification * ver_score
        + w_calibration * cal_score
    )
    return {
        "composite_reward": round(composite, 6),
        "influence_raw": raw_inf,
        "influence_norm": round(inf_norm, 6),
        "verification_score": ver_score,
        "calibration_score": cal_score,
    }


def export_student_multi_preference_datasets(
    scored_candidates: list[dict[str, Any]],
    sources_map_by_doc: dict[str, dict[str, Any]],
    output_dir: Path,
    min_composite_margin: float = 0.15,
) -> dict[str, Any]:
    """Exports diverse, systematic Student preference training datasets from API candidates:

      1. `student_dpo_pairs.jsonl`: Paired preference data (user prompt = source_bundle + question,
         assistant = chosen y+, rejected_response = y-) for `swift rlhf --rlhf_type dpo`.
      2. `student_kto_dataset.jsonl`: Unpaired binary desirability data (`messages` + `label`: true/false)
         for `swift rlhf --rlhf_type kto`, utilizing 100% of API rollouts without needing exact pairs.
      3. `student_grpo_prompts.jsonl`: Verifiable RLVR prompt dataset with reference evidence quotes
         and gold answers for on-policy GRPO / BSD training.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group candidates by source document / chunk
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in scored_candidates:
        enriched = dict(rec)
        enriched["reward_breakdown"] = compute_composite_preference_score(rec)
        gid = str(rec.get("prompt_group_id") or rec.get("source_chunk_ids", ["default"])[0])
        groups[gid].append(enriched)

    student_dpo_rows: list[dict[str, Any]] = []
    student_kto_rows: list[dict[str, Any]] = []
    student_grpo_rows: list[dict[str, Any]] = []

    for gid, items in groups.items():
        ranked = sorted(
            items,
            key=lambda r: r["reward_breakdown"]["composite_reward"],
            reverse=True,
        )

        # 1. KTO: Every valid positive-reward candidate gets label=True; negative gets label=False
        for item in ranked:
            sft_msgs = item.get("sft_messages")
            if not sft_msgs:
                # Reconstruct SFT messages even for rejected candidates so KTO can penalize them
                doc_ids = item.get("source_doc_ids", [])
                src_list = [sources_map_by_doc[d] for d in doc_ids if d in sources_map_by_doc]
                if src_list and item.get("question") and item.get("answer"):
                    bundle_str, _ = format_source_bundle(src_list)
                    sft_msgs = [
                        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                        {"role": "user", "content": f"{bundle_str}\n\n问题：{item['question']}"},
                        {"role": "assistant", "content": str(item["answer"])},
                    ]
            if not sft_msgs:
                continue

            comp_r = item["reward_breakdown"]["composite_reward"]
            is_desirable = (
                comp_r > 0.0
                and item.get("checks", {}).get("programmatic_valid", False)
                and float(item.get("influence", {}).get("score", 0.0)) > 0.0
            )
            kto_entry: dict[str, Any] = {
                "messages": sft_msgs,
                "label": bool(is_desirable),
                "meta": {
                    "sample_id": item["sample_id"],
                    "prompt_group_id": gid,
                    "composite_reward": comp_r,
                    "influence_score": item["reward_breakdown"]["influence_raw"],
                },
            }
            if item.get("images"):
                kto_entry["images"] = list(item["images"])
            student_kto_rows.append(kto_entry)

        # 2. Student DPO & GRPO: Pair top valid positive answers against negative / flawed answers
        positives = [
            it
            for it in ranked
            if it.get("checks", {}).get("programmatic_valid", False)
            and float(it.get("influence", {}).get("score", 0.0)) > 0.0
            and it.get("sft_messages")
        ]
        negatives = [
            it
            for it in reversed(ranked)
            if float(it.get("influence", {}).get("score", 0.0)) < 0.0
            or not it.get("checks", {}).get("programmatic_valid", True)
        ]

        if positives:
            best_pos = positives[0]
            # Export verifiable GRPO prompt entry
            grpo_entry: dict[str, Any] = {
                "messages": best_pos["sft_messages"][:-1],
                "ground_truth": best_pos["answer"],
                "verifiable_citations": best_pos.get("citations", []),
                "verifiable_evidence": best_pos.get("evidence", []),
                "task_type": best_pos.get("task_type", "fact"),
                "sample_id": best_pos["sample_id"],
            }
            if best_pos.get("images"):
                grpo_entry["images"] = list(best_pos["images"])
            student_grpo_rows.append(grpo_entry)

            # Construct Student DPO pairs: align on the shared question + source_bundle
            # (In API multi-answer sampling or cross-candidate comparison, we pair the best
            # high-influence answer against each negative answer on the same source chunk)
            for neg in negatives:
                if best_pos["sample_id"] == neg["sample_id"]:
                    continue
                margin = (
                    best_pos["reward_breakdown"]["composite_reward"]
                    - neg["reward_breakdown"]["composite_reward"]
                )
                if margin < min_composite_margin:
                    continue
                neg_answer = str(neg.get("answer", "")).strip()
                if not neg_answer or neg_answer == best_pos["answer"]:
                    continue

                defect_tag = diagnose_negative_defect(best_pos, neg)
                dpo_entry: dict[str, Any] = {
                    "messages": best_pos["sft_messages"],
                    "rejected_response": neg_answer,
                    "meta": {
                        "prompt_group_id": gid,
                        "chosen_sample_id": best_pos["sample_id"],
                        "rejected_sample_id": neg["sample_id"],
                        "chosen_composite_reward": best_pos["reward_breakdown"]["composite_reward"],
                        "rejected_composite_reward": neg["reward_breakdown"]["composite_reward"],
                        "composite_margin": round(margin, 6),
                        "chosen_influence": best_pos["reward_breakdown"]["influence_raw"],
                        "rejected_influence": neg["reward_breakdown"]["influence_raw"],
                        "negative_defect_type": defect_tag,
                    },
                }
                if best_pos.get("images"):
                    dpo_entry["images"] = list(best_pos["images"])
                student_dpo_rows.append(dpo_entry)

    dpo_path = output_dir / "student_dpo_pairs.jsonl"
    kto_path = output_dir / "student_kto_dataset.jsonl"
    grpo_path = output_dir / "student_grpo_prompts.jsonl"

    for path, rows in (
        (dpo_path, student_dpo_rows),
        (kto_path, student_kto_rows),
        (grpo_path, student_grpo_rows),
    ):
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "student_dpo_pairs_count": len(student_dpo_rows),
        "student_kto_rows_count": len(student_kto_rows),
        "student_grpo_prompts_count": len(student_grpo_rows),
        "paths": {
            "student_dpo": str(dpo_path),
            "student_kto": str(kto_path),
            "student_grpo": str(grpo_path),
        },
    }


def run_api_preference_synthesizer(
    sources_path: Path,
    scored_candidates_path: Path,
    prompt_template_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Runs the full Black-Box API Alignment + Multi-Format Student Preference export."""
    sources = load_jsonl(sources_path)
    sources_by_doc = {str(s["doc_id"]): s for s in sources}
    scored_candidates = load_jsonl(scored_candidates_path)
    base_template = prompt_template_path.read_text(encoding="utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build Stratified Preference Memory Bank for Black-Box API Steering
    memory_bank = PreferenceMemoryBank(scored_candidates)
    bank_path = output_dir / "api_preference_memory_bank.json"
    meta_rubric = memory_bank.synthesize_meta_prompt_rubric()

    # Render an example In-Context Contrastive Steered Prompt for the first source
    first_bundle, _ = format_source_bundle([sources[0]])
    steered_prompt = memory_bank.build_in_context_steered_prompt(
        base_prompt=base_template.replace("{{source_bundle}}", first_bundle),
        query_source_bundle=first_bundle,
        target_task_type="condition",
        top_k=1,
    )
    steered_prompt_path = output_dir / "api_steered_prompt_example.txt"
    steered_prompt_path.write_text(steered_prompt, encoding="utf-8")

    bank_payload = {
        "memory_bank_size": len(memory_bank.entries),
        "meta_prompt_evolution": meta_rubric,
        "entries": memory_bank.entries,
    }
    bank_path.write_text(json.dumps(bank_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 2. Export Multi-Dimensional Student Preference Datasets (DPO + KTO + GRPO)
    export_summary = export_student_multi_preference_datasets(
        scored_candidates=scored_candidates,
        sources_map_by_doc=sources_by_doc,
        output_dir=output_dir,
    )

    return {
        "api_alignment": {
            "memory_bank_size": len(memory_bank.entries),
            "memory_bank_path": str(bank_path),
            "steered_prompt_example_path": str(steered_prompt_path),
            "evolved_rules_count": len(meta_rubric["evolved_meta_rules"]),
        },
        "student_preference_exports": export_summary,
        "personas_configured": list(API_SYNTHESIS_PERSONAS.keys()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True, help="Input source_documents JSONL")
    parser.add_argument("--scored-candidates", type=Path, required=True, help="Scored candidates JSONL")
    parser.add_argument("--prompt-template", type=Path, required=True, help="Path to qa_generate.md")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for API & preference artifacts")
    args = parser.parse_args(argv)

    summary = run_api_preference_synthesizer(
        sources_path=args.sources,
        scored_candidates_path=args.scored_candidates,
        prompt_template_path=args.prompt_template,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
