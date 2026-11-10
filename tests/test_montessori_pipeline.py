#!/usr/bin/env python3
"""Unit and integration tests for the Montessori-Instruct Domain QA Synthesis pipeline."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MONTESSORI_DIR = REPO_ROOT / "training" / "domain_adaptation" / "montessori"
sys.path.insert(0, str(MONTESSORI_DIR))

import api_preference_synthesizer  # noqa: E402
import build_dpo_pairs  # noqa: E402
import influence_scorer  # noqa: E402
import probing_generate  # noqa: E402


class MontessoriPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sources_path = REPO_ROOT / "training" / "domain_adaptation" / "examples" / "source_documents_example.jsonl"
        self.prompt_path = REPO_ROOT / "training" / "domain_adaptation" / "prompts" / "qa_generate.md"
        self.ref_path = REPO_ROOT / "training" / "domain_adaptation" / "examples" / "style_sft_example.jsonl"

    def test_stage1_probing_generation_and_exact_quote_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "probing.jsonl"
            summary = probing_generate.run_probing_generation(
                sources_path=self.sources_path,
                prompt_template_path=self.prompt_path,
                output_path=out_path,
                num_candidates_per_source=4,
                mode="mock",
            )
            self.assertEqual(summary["total_sources"], 1)
            self.assertEqual(summary["total_generated"], 4)
            self.assertEqual(summary["valid_candidates"], 4)

            rows = probing_generate.load_jsonl(out_path)
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertTrue(row["checks"]["programmatic_valid"])
                self.assertEqual(len(row["sft_messages"]), 3)
                self.assertEqual(row["sft_messages"][-1]["role"], "assistant")
                for ev in row["evidence"]:
                    self.assertGreaterEqual(ev["start_char"], 0)
                    self.assertGreater(ev["end_char"], ev["start_char"])

    def test_fabricated_citation_or_non_substring_quote_rejected(self) -> None:
        sources = probing_generate.load_jsonl(self.sources_path)
        _, source_map = probing_generate.format_source_bundle(sources)
        bad_item = {
            "question": "测试问题？",
            "task_type": "fact",
            "answerable": True,
            "answer": "这是一个带有伪造引用的回答。[S99]",
            "citations": ["S99"],
            "evidence": [{"source_key": "S1", "quote": "原文中根本不存在的句子", "locator": "第二条"}],
        }
        is_valid, errors, _ = probing_generate.verify_and_enrich_item(bad_item, source_map)
        self.assertFalse(is_valid)
        self.assertIn("unknown_citation_key:S99", errors)
        self.assertIn("evidence_quote_not_exact_substring:S1", errors)

    def test_stage2_and_stage3_influence_to_dpo_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            probing_path = tmp_dir / "probing.jsonl"
            scored_path = tmp_dir / "scored.jsonl"
            dpo_path = tmp_dir / "dpo.jsonl"
            sft_path = tmp_dir / "sft.jsonl"
            prov_path = tmp_dir / "prov.jsonl"

            probing_generate.run_probing_generation(
                sources_path=self.sources_path,
                prompt_template_path=self.prompt_path,
                output_path=probing_path,
                num_candidates_per_source=4,
                mode="mock",
            )
            inf_summary = influence_scorer.run_influence_scoring(
                candidates_path=probing_path,
                ref_dataset_path=self.ref_path,
                output_path=scored_path,
                backend="mock",
                method="one_step_loss_diff",
                probe_lr=1e-4,
            )
            self.assertEqual(inf_summary["total_candidates"], 4)
            self.assertGreaterEqual(inf_summary["positive_influence_count"], 1)
            self.assertGreaterEqual(inf_summary["negative_influence_count"], 1)

            dpo_summary = build_dpo_pairs.build_dpo_and_sft_datasets(
                scored_candidates_path=scored_path,
                dpo_output_path=dpo_path,
                sft_output_path=sft_path,
                provenance_output_path=prov_path,
            )
            self.assertGreaterEqual(dpo_summary["dpo_pairs_count"], 1)
            self.assertEqual(dpo_summary["student_sft_count"], 1)

            dpo_rows = build_dpo_pairs.load_jsonl(dpo_path)
            for pair in dpo_rows:
                self.assertIn("messages", pair)
                self.assertIn("rejected_response", pair)
                self.assertGreater(pair["meta"]["chosen_influence"], 0.0)
                self.assertLess(pair["meta"]["rejected_influence"], 0.0)
                self.assertGreater(pair["meta"]["influence_margin"], 0.0)

    def test_end_to_end_shell_orchestrator_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin",
                "WORK_DIR": tmp,
                "DRY_RUN": "1",
            }
            proc = subprocess.run(
                ["bash", str(MONTESSORI_DIR / "run_montessori_pipeline.sh")],
                cwd=str(MONTESSORI_DIR),
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("swift rlhf --rlhf_type dpo", proc.stdout)
            self.assertIn("swift sft --model", proc.stdout)
            self.assertTrue((Path(tmp) / "teacher_dpo_pairs.jsonl").exists())
            self.assertTrue((Path(tmp) / "student_sft_messages.jsonl").exists())
            self.assertTrue((Path(tmp) / "api_preference" / "student_dpo_pairs.jsonl").exists())
            self.assertTrue((Path(tmp) / "api_preference" / "student_kto_dataset.jsonl").exists())
            self.assertTrue((Path(tmp) / "api_preference" / "student_grpo_prompts.jsonl").exists())
            self.assertTrue((Path(tmp) / "api_preference" / "api_steered_prompt_example.txt").exists())

    def test_api_blackbox_steering_and_multiformat_preference_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            probing_path = tmp_dir / "probing.jsonl"
            scored_path = tmp_dir / "scored.jsonl"
            api_out = tmp_dir / "api_out"

            probing_generate.run_probing_generation(
                sources_path=self.sources_path,
                prompt_template_path=self.prompt_path,
                output_path=probing_path,
                num_candidates_per_source=4,
                mode="mock",
            )
            influence_scorer.run_influence_scoring(
                candidates_path=probing_path,
                ref_dataset_path=self.ref_path,
                output_path=scored_path,
                backend="mock",
                method="one_step_loss_diff",
                probe_lr=1e-4,
            )
            summary = api_preference_synthesizer.run_api_preference_synthesizer(
                sources_path=self.sources_path,
                scored_candidates_path=scored_path,
                prompt_template_path=self.prompt_path,
                output_dir=api_out,
            )
            self.assertGreaterEqual(summary["api_alignment"]["memory_bank_size"], 1)
            exports = summary["student_preference_exports"]
            self.assertGreaterEqual(exports["student_dpo_pairs_count"], 1)
            self.assertEqual(exports["student_kto_rows_count"], 4)
            self.assertEqual(exports["student_grpo_prompts_count"], 1)

            steered_text = (api_out / "api_steered_prompt_example.txt").read_text(encoding="utf-8")
            self.assertIn("In-Context Contrastive Steering", steered_text)
            self.assertIn("正向高收益生成", steered_text)
            self.assertIn("负向受损生成", steered_text)

            kto_rows = probing_generate.load_jsonl(api_out / "student_kto_dataset.jsonl")
            labels = [r["label"] for r in kto_rows]
            self.assertIn(True, labels)
            self.assertIn(False, labels)


if __name__ == "__main__":
    unittest.main()

