#!/usr/bin/env python3
"""Unit and integration tests for the Refactored V1-V4 -> CPT & Unified QA-SFT Pipeline."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FLASH_DIR = REPO_ROOT / "training" / "flash"
sys.path.insert(0, str(FLASH_DIR))

import build_mm_wiki_cpt as cpt_builder  # noqa: E402
import refactor_v1_v4_pipeline as router  # noqa: E402


class UnifiedV1V4PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.v1_v4_sample = FLASH_DIR / "examples" / "v1_v4_legacy_sample.jsonl"
        self.new_qa_sample = FLASH_DIR / "examples" / "style_sft_example.jsonl"
        self.sft_cfg_path = FLASH_DIR / "examples" / "configs" / "sft_27b_unified_v1_v4_qa_b300.json"
        self.unified_script = FLASH_DIR / "run_flash_pipeline.sh"

    def test_ternary_key_keeps_v4_multi_image_and_rejects_cross_pool_leak(self) -> None:
        raw_rows = [(r, self.v1_v4_sample) for r in cpt_builder.load_jsonl(self.v1_v4_sample)]
        cleaned, report = router.validate_and_dedup_v1_v4(raw_rows)
        self.assertEqual(report["validated_rows"], 8)
        # v4_demo_001 appears twice with different cate_id and images; both must survive.
        v4_001 = [r for r in cleaned if r["sample_id"] == "v4_demo_001"]
        self.assertEqual(len(v4_001), 2)

        # Cross-pool overlap on the exact ternary key must raise ValueError.
        dup_cross_pool = dict(cleaned[0])
        dup_cross_pool["pool"] = "final"
        with self.assertRaisesRegex(ValueError, "Cross-pool overlap"):
            router.validate_and_dedup_v1_v4(raw_rows + [(dup_cross_pool, None)])

    def test_v4_final_requires_enable_thinking_false(self) -> None:
        bad_v4_final = {
            "sample_id": "v4_bad",
            "cate_id": "kd",
            "variant": 4,
            "pool": "final",
            "enable_thinking": True,
            "images": ["a.jpg"],
            "messages": [
                {"role": "user", "content": "<image>\n描述"},
                {"role": "assistant", "content": "结果"},
            ],
        }
        with self.assertRaisesRegex(ValueError, "enable_thinking=false"):
            router.validate_and_dedup_v1_v4([(bad_v4_final, None)])

    def test_cpt_routing_filters_v2_and_final_and_upgrades_v4_rag(self) -> None:
        raw_rows = [(r, self.v1_v4_sample) for r in cpt_builder.load_jsonl(self.v1_v4_sample)]
        cleaned, _ = router.validate_and_dedup_v1_v4(raw_rows)
        cpt_rows, summary = router.route_v1_v4_to_cpt(cleaned)

        # In v1_v4_legacy_sample.jsonl:
        # - 2 final rows (v1_demo_002, v4_demo_002) reserved for SFT/eval
        # - 2 V2 rows (v2_demo_001, v2_demo_002) filtered by default
        # - Remaining 4 rows routed: v1_demo_001 (ch6), v3_demo_001 (ch6), v4_demo_001_a (ch5!), v4_demo_001_b (ch6)
        self.assertEqual(summary["cpt_routed_rows"], 4)
        self.assertEqual(summary["skipped_reasons"]["reserved_final_for_sft_eval"], 2)
        self.assertEqual(summary["skipped_reasons"]["filtered_v2_caption_dilution"], 2)

        channels = {r["sample_id"]: r["channel"] for r in cpt_rows}
        self.assertEqual(channels["cpt_v4_demo_001_knowledge_desc_a"], "ch5_flash_rag_raft")
        self.assertEqual(channels["cpt_v1_demo_001_ip-brand"], "ch6_general_replay")
        for row in cpt_rows:
            self.assertEqual(cpt_builder.validate_cpt_record(row), [])

    def test_end_to_end_refactor_integrates_with_cpt_builder_and_sft(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            refactor_dir = tmp_dir / "refactor"
            manifest = router.run_refactor_pipeline(
                [self.v1_v4_sample],
                refactor_dir,
                new_qa_paths=[self.new_qa_sample],
                sft_target_rows=24,
                val_ratio=0.25,
                seed=42,
            )
            self.assertEqual(manifest["unified_sft"]["doc_family_overlap"], 0)
            self.assertGreaterEqual(manifest["qwen_image_edit_seeds_count"], 2)

            sft_train = cpt_builder.load_jsonl(refactor_dir / "sft_unified_train.jsonl")
            sft_channels = {r["channel"] for r in sft_train}
            self.assertIn("V1", sft_channels)
            self.assertIn("wiki_rag_qa", sft_channels)

            # Feed CPT replay output + 6ch wiki demo into build_mm_wiki_cpt.py
            cpt_bundle_dir = tmp_dir / "cpt_bundle"
            wiki_records = cpt_builder.load_jsonl(FLASH_DIR / "examples" / "cpt_mm_6ch_example.jsonl")
            replay_records = cpt_builder.load_jsonl(refactor_dir / "cpt_v1_v4_replay.jsonl")
            cpt_manifest = cpt_builder.build_cpt_bundle(
                wiki_records + replay_records,
                cpt_bundle_dir,
                val_ratio=0.20,
                stage1_rows=24,
                stage2_rows=16,
                seed=42,
            )
            self.assertEqual(cpt_manifest["doc_family_overlap"], 0)

    def test_tier1_edit_seeds_and_cate_map_enrichment(self) -> None:
        sample = [
            {
                "sample_id": "s_tier1",
                "cate_id": "brand_logo",
                "variant": 3,
                "doc_family_id": "fam_t1",
                "images": ["img_t1.jpg"],
                "messages": [{"role": "assistant", "content": "ok"}],
            },
            {
                "sample_id": "s_mapped",
                "cate_id": "cate_99",
                "variant": 1,
                "doc_family_id": "fam_mapped",
                "images": ["img_mapped.jpg"],
                "messages": [{"role": "assistant", "content": "ok"}],
            },
        ]
        seeds = router.extract_qwen_image_edit_seeds(
            sample,
            cate_map={"cate_99": "示例品牌名"},
        )
        self.assertEqual(len(seeds), 2)
        by_sid = {s["source_sample_id"]: s for s in seeds}
        self.assertEqual(by_sid["s_tier1"]["cate_id"], "brand_logo")
        self.assertEqual(by_sid["s_mapped"]["cate_name"], "示例品牌名")

    def test_sft_native_config_valid(self) -> None:
        cfg = json.loads(self.sft_cfg_path.read_text(encoding="utf-8"))
        self.assertEqual(cfg["tuner_type"], "lora")
        self.assertTrue(cfg["enable_channel_loss"])
        self.assertTrue(cfg["freeze_vit"])
        self.assertFalse(cfg["freeze_aligner"])

    def test_unified_b300_shell_orchestrator_dry_run(self) -> None:
        syntax = subprocess.run(["bash", "-n", str(self.unified_script)], capture_output=True, text=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

        with tempfile.TemporaryDirectory() as tmp:
            for tuner in ("lora", "full"):
                work_dir = Path(tmp) / f"run_{tuner}"
                env = {
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin"),
                    "WORK_DIR": str(work_dir),
                    "DEMO": "1",
                    "DRY_RUN": "1",
                    "CPT_TUNER_TYPE": tuner,
                }
                proc = subprocess.run(
                    ["bash", str(self.unified_script)],
                    cwd=str(REPO_ROOT),
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertIn("megatron pt", proc.stdout)
                self.assertIn("--lr_decay_style WSD", proc.stdout)
                self.assertIn("--max_length 8192", proc.stdout)
                self.assertIn("swift export", proc.stdout)
                self.assertIn("megatron sft", proc.stdout)
                self.assertTrue((work_dir / "step0_qa_cot_v04" / "cot_manifest.json").exists())
                self.assertTrue((work_dir / "step0_v2_images" / "v2_pipeline_manifest.json").exists())
                self.assertTrue((work_dir / "step1_v1_v4_refactored" / "refactor_manifest.json").exists())
                self.assertTrue((work_dir / "step2_midtrain_bundle" / "midtrain_manifest.json").exists())

    def test_three_sft_comparison_arms(self) -> None:
        raw_rows = [(r, self.v1_v4_sample) for r in cpt_builder.load_jsonl(self.v1_v4_sample)]
        cleaned, _ = router.validate_and_dedup_v1_v4(raw_rows)
        qa_rows = [
            {
                "sample_id": "qa_cot_1",
                "doc_family_id": "fam_cot_1",
                "enable_thinking": True,
                "messages": [
                    {"role": "user", "content": "Q1"},
                    {"role": "assistant", "content": "<think>\n推演过程\n</think>\n\nA1"},
                ],
            },
            {
                "sample_id": "qa_dir_1",
                "doc_family_id": "fam_dir_1",
                "enable_thinking": False,
                "messages": [
                    {"role": "user", "content": "Q2"},
                    {"role": "assistant", "content": "A2"},
                ],
            },
        ]
        _, _, sum_ood = router.build_unified_sft_bundle(cleaned, qa_rows, sft_arm="ood_only", val_ratio=0.0)
        self.assertNotIn("wiki_rag_qa_cot", sum_ood["channels"])
        self.assertNotIn("wiki_rag_qa_direct", sum_ood["channels"])

        train_dir, _, sum_dir = router.build_unified_sft_bundle(
            cleaned, qa_rows, sft_arm="ood_plus_qa_direct", val_ratio=0.0
        )
        self.assertIn("wiki_rag_qa_direct", sum_dir["channels"])
        for row in train_dir:
            if row["channel"] == "wiki_rag_qa_direct":
                self.assertFalse(row["enable_thinking"])
                self.assertNotIn("<think>", row["messages"][-1]["content"])

        _, _, sum_cot = router.build_unified_sft_bundle(
            cleaned, qa_rows, sft_arm="ood_plus_qa_cot", cot_direct_ratio=0.8, val_ratio=0.0
        )
        self.assertIn("wiki_rag_qa_cot", sum_cot["channels"])
        self.assertIn("wiki_rag_qa_direct", sum_cot["channels"])


if __name__ == "__main__":
    unittest.main()
