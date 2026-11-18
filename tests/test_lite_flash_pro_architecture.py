#!/usr/bin/env python3
"""Unit and integration tests for the refactored Lite / Flash / Pro + QA-CoT v0.3 architecture."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LITE_DIR = REPO_ROOT / "training" / "lite"
FLASH_DIR = REPO_ROOT / "training" / "flash"
PRO_DIR = REPO_ROOT / "training" / "pro"
CPT27B_DIR = REPO_ROOT / "training" / "cpt_27b"

for p in (LITE_DIR, FLASH_DIR, PRO_DIR):
    sys.path.insert(0, str(p))

import build_cpt_from_sch as lite_builder  # noqa: E402
import error_cluster_miner as pro_miner  # noqa: E402
import gen_cot_from_qa as cot_gen  # noqa: E402


class LiteFlashProArchitectureTests(unittest.TestCase):
    def test_cpt_27b_wrapper_profiles_and_guards(self) -> None:
        wrapper = CPT27B_DIR / "run_cpt_27b_b300.sh"
        self.assertEqual(subprocess.run(["bash", "-n", str(wrapper)], capture_output=True).returncode, 0)

        # Missing --model must fail with exit code 2
        no_model = subprocess.run(
            ["bash", str(wrapper), "lite"],
            cwd=str(REPO_ROOT),
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            capture_output=True,
            text=True,
        )
        self.assertEqual(no_model.returncode, 2)

        model_path = "/data/autovision-cfs/autovision-notebook/algo4/models/Qwen3.5-27B"
        lite_proc = subprocess.run(
            ["bash", str(wrapper), "lite", "--model", model_path],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--gradient_accumulation_fusion false", lite_proc.stdout)
        self.assertIn("--micro_batch_size 4", lite_proc.stdout)
        self.assertIn("--global_batch_size 32", lite_proc.stdout)
        self.assertIn("--tensor_model_parallel_size 4", lite_proc.stdout)
        self.assertIn("--max_length 16384", lite_proc.stdout)
        self.assertIn("--num_train_epochs 2", lite_proc.stdout)
        self.assertIn("--lr_decay_style constant", lite_proc.stdout)
        self.assertIn("--lr_warmup_fraction 0.06", lite_proc.stdout)
        self.assertNotIn("--val_dataset", lite_proc.stdout)

        # Even if --launch is passed, explicit DRY_RUN=1 prevents execution
        safe_launch = subprocess.run(
            ["bash", str(wrapper), "stage2", "--model", model_path, "--stage2-decay-iters", "28", "--launch", "--dry-run"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--lr_decay_style WSD", safe_launch.stdout)
        self.assertIn("--lr_wsd_decay_style linear", safe_launch.stdout)
        self.assertIn("--lr_wsd_decay_iters 28", safe_launch.stdout)

        pro_proc = subprocess.run(
            ["bash", str(wrapper), "pro", "--model", model_path],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--lr 5e-6", pro_proc.stdout)
        self.assertIn("--lr_decay_style cosine", pro_proc.stdout)

    def test_lite_sch_iso_elements_override_and_val0(self) -> None:
        sch, kb, supp = lite_builder.build_demo_inputs()
        # Add an ft_mention KB item that must be excluded
        kb_with_ft = kb + [
            {
                "sample_id": "ft_bad_01",
                "doc_family_id": "fam_ft",
                "channel": "ch_text_doc",
                "source_kind": "ft_mention_noise",
                "level": "a_tier",
                "messages": [{"role": "assistant", "content": "should be excluded"}],
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            manifest = lite_builder.build_lite_bundle(
                sch_records=sch,
                kb_records=kb_with_ft,
                kb_supplement_records=supp,
                output_dir=out_dir,
            )
            self.assertEqual(manifest["val_rows"], 0)
            self.assertEqual(manifest["ft_mentions_excluded_count"], 1)
            self.assertEqual(manifest["zero_leakage_assert"], "passed")
            self.assertTrue((out_dir / "bundle_lite_v3_all" / "lite_val.jsonl").read_text(encoding="utf-8") == "")

            compat_link = out_dir / "bundle_lite_compat" / "stage1_train.jsonl"
            self.assertTrue(compat_link.is_symlink())
            rows = lite_builder.load_jsonl(compat_link)
            self.assertEqual(len(rows), manifest["rows"])

            by_sid = {r["sample_id"]: r for r in rows}
            # Critical Runbook rule: label=True but elements="无" -> ch_ip_negative
            neg_row = by_sid["demo_sch_label_mismatch_02"]
            self.assertEqual(neg_row["channel"], "ch_ip_negative")
            self.assertTrue(neg_row["audit_raw_label"])
            self.assertIn("元素：无。", neg_row["messages"][0]["content"])
            self.assertTrue(neg_row["messages"][0]["content"].startswith("<image>\n"))

            # Atomic desc + elements must be used instead of `polished_text`
            ground_row = by_sid["demo_sch_ground_01"]
            self.assertEqual(ground_row["channel"], "ch_ip_grounding")
            self.assertNotIn("错误的改写版", ground_row["messages"][0]["content"])
            self.assertEqual(by_sid["demo_sch_raft_03"]["channel"], "ch5_flash_rag_raft")

            sch_report = json.loads((out_dir / "build_sch_report.json").read_text(encoding="utf-8"))
            self.assertEqual(sch_report["label_true_but_empty_elements_count"], 1)

    def test_flash_qa_cot_v03_8gates_remapping_and_fallback(self) -> None:
        raw_records = cot_gen.load_jsonl(FLASH_DIR / "examples" / "qa_kb_raw_example.jsonl")
        # Append a record whose teacher trace intentionally violates Gate 8 + Gate 7 -> must fall back to Direct
        failing_rec = json.loads(json.dumps(raw_records[0], ensure_ascii=False))
        failing_rec["sample_id"] = "qa_force_fail_07"
        failing_rec["doc_family_id"] = "fam_force_fail_07"
        failing_rec["_mock_force_fail"] = True

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            manifest = cot_gen.run_cot_generation_pipeline(
                qa_records=raw_records + [failing_rec],
                output_dir=out_dir,
                mode="mock",
                smoke_sample=4,
                seed=42,
            )
            self.assertEqual(manifest["raw_type_aliases_fixed"].get("synonsense->synthesis"), 1)
            self.assertEqual(manifest["cot_passed_rows"], 6)
            self.assertEqual(manifest["direct_fallback_rows"], 1)
            self.assertEqual(manifest["dpo_rejected_cots_rows"], 1)
            self.assertEqual(manifest["smoke_sample_rows"], 4)

            cot_rows = cot_gen.load_jsonl(out_dir / "qa_cot_passed.jsonl")
            for r in cot_rows:
                self.assertTrue(r["enable_thinking"])
                self.assertEqual(r["channel"], "wiki_rag_qa_cot")
                self.assertIn("<think>", r["messages"][-1]["content"])

            fallback_rows = cot_gen.load_jsonl(out_dir / "qa_direct_fallback.jsonl")
            self.assertEqual(len(fallback_rows), 1)
            fb = fallback_rows[0]
            self.assertFalse(fb["enable_thinking"])
            self.assertEqual(fb["channel"], "wiki_rag_qa_direct")
            self.assertNotIn("<think>", fb["messages"][-1]["content"])

        # Explicit unit checks on individual 8-gate failure modes + soft upper bound
        src_text = "爱尔兰国旗自旗杆侧起依次为绿、白、橙三色竖条，标准宽高比1:2。普通条纹不适用。"
        dist_text = "科特迪瓦国旗自旗杆侧起依次为橙、白、绿三色竖条。"
        source_map = {
            "S1": {"source_key": "S1", "text": dist_text, "content_sha256": cot_gen.sha256_text(dist_text)},
            "S2": {"source_key": "S2", "text": src_text, "content_sha256": cot_gen.sha256_text(src_text)},
        }
        evidence = [{"source_key": "S2", "quote": "绿、白、橙三色竖条", "content_sha256": cot_gen.sha256_text(src_text)}]

        # 1. Citation recall miss + hindsight phrase + short length
        bad_report = cot_gen.verify_cot_8gates(
            reasoning="参考答案说明这是爱尔兰国旗。",
            final_answer="属于爱尔兰国旗。[S2]",
            gold_answer="属于爱尔兰国旗。[S2]",
            task_type="fact",
            answerable=True,
            gold_citations=["S2"],
            distractor_keys=["S1"],
            source_map=source_map,
            evidence=evidence,
        )
        self.assertFalse(bad_report["passed"])
        self.assertFalse(bad_report["gates"]["gate3_citation_recall"])
        self.assertFalse(bad_report["gates"]["gate7_dynamic_length_lower"])
        self.assertFalse(bad_report["gates"]["gate8_style_forbidden_words"])

        # 2. Soft upper bound >300 chars triggers soft_warnings but does NOT fail gate 7
        long_reasoning = (
            "先拆解问题约束条件，逐条检索比对材料细节。"
            "核对 [S2] 明确了色序特征与比例参数，能够建立直接依据；"
            "同步排查 [S1] 属于其他国家的反向色序干扰条目，应当予以排除。"
            + "进一步核对条文限定词与适用范围边界，确保不存在跨版本混用与跳步推理，每一项论证环节均与有效出处严密对应。" * 5
        )
        soft_report = cot_gen.verify_cot_8gates(
            reasoning=long_reasoning,
            final_answer="属于爱尔兰国旗。[S2]",
            gold_answer="属于爱尔兰国旗。[S2]",
            task_type="fact",
            answerable=True,
            gold_citations=["S2"],
            distractor_keys=["S1"],
            source_map=source_map,
            evidence=evidence,
        )
        self.assertTrue(soft_report["passed"], soft_report["errors"])
        self.assertTrue(len(soft_report["soft_warnings"]) >= 1)

        # 3. Remapping collision safety when S1 and S2 swap
        ctx = cot_gen.inject_distractors_and_remap(
            {
                "question": "Q",
                "answer": "依据[S1]与[S2]判定。",
                "citations": ["S1", "S2"],
                "sources": [
                    {"source_key": "S1", "doc_id": "d1", "text": "文本一"},
                    {"source_key": "S2", "doc_id": "d2", "text": "文本二"},
                ],
                "evidence": [{"source_key": "S1", "quote": "文本一"}],
                "distractors": [{"doc_id": "d3", "text": "干扰三"}],
            },
            seed=7,
        )
        self.assertEqual(len(ctx["distractor_keys"]), 1)
        for gc in ctx["gold_citations"]:
            self.assertIn(f"[{gc}]", ctx["gold_answer"])
            self.assertNotIn(gc, ctx["distractor_keys"])

    def test_pro_error_cluster_miner_50_30_20_and_sft_isolation(self) -> None:
        inputs = pro_miner.load_demo_records()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            manifest = pro_miner.run_pro_pipeline(
                error_logs=inputs["error_logs"],
                accepted_templates=inputs["accepted_templates"],
                v_replay=inputs["v_replay"],
                sft_candidates=inputs["sft_candidates"],
                output_dir=out_dir,
                total_cpt_rows=20,
                val_ratio=0.2,
                seed=42,
            )
            self.assertEqual(manifest["sft_overlap"], 0)
            self.assertEqual(manifest["index_prescriptions_count"], 2)
            self.assertEqual(manifest["actual_cpt_mix_shares"]["new_verified_gap"], 0.5)
            self.assertEqual(manifest["actual_cpt_mix_shares"]["accepted_template_replay"], 0.3)
            self.assertEqual(manifest["actual_cpt_mix_shares"]["v_series_replay"], 0.2)
            self.assertEqual(manifest["sft_excluded_rows"], 1)

            prescriptions = pro_miner.load_jsonl(out_dir / "index_prescriptions.jsonl")
            for p in prescriptions:
                self.assertFalse(p["requires_weight_update"])


if __name__ == "__main__":
    unittest.main()
