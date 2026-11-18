#!/usr/bin/env python3
"""Unit and dry-run integration tests for the B300 27B Multimodal Wiki+RAG CPT pipeline."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FLASH_DIR = REPO_ROOT / "training" / "flash"
sys.path.insert(0, str(FLASH_DIR))

import build_mm_wiki_cpt as builder  # noqa: E402


class B300MultimodalCptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.example_path = FLASH_DIR / "examples" / "cpt_mm_6ch_example.jsonl"
        self.megatron_cfg_path = FLASH_DIR / "examples" / "configs" / "cpt_27b_mm_rag_wsd_megatron.json"
        self.swift_cfg_path = FLASH_DIR / "examples" / "configs" / "cpt_27b_mm_rag_swift_lora.json"
        self.cpt_script = REPO_ROOT / "training" / "cpt_27b" / "train_cpt_qwen3_5_27b_megatron.sh"

    def test_six_channel_example_jsonl_valid(self) -> None:
        self.assertIn("ch_ip_grounding", builder.CANONICAL_CHANNELS)
        self.assertIn("ch_ip_negative", builder.CANONICAL_CHANNELS)
        rows = builder.load_jsonl(self.example_path)
        self.assertEqual(len(rows), 6)
        channels = {r["channel"] for r in rows}
        expected_active = {
            "ch_text_doc",
            "ch_mm_doc",
            "ch_rel_jump",
            "ch2_std_logo_flag",
            "ch5_flash_rag_raft",
            "ch6_general_replay",
        }
        self.assertEqual(channels, expected_active)
        for row in rows:
            errors = builder.validate_cpt_record(row)
            self.assertEqual(errors, [], f"unexpected errors for {row.get('sample_id')}: {errors}")

    def test_validation_rejects_causal_prefix_and_image_mismatch(self) -> None:
        bad_prefix = {
            "sample_id": "bad_1",
            "doc_family_id": "fam_bad_1",
            "channel": "ch2_std_logo_flag",
            "images": ["a.png"],
            "messages": [
                {"role": "assistant", "content": "爱尔兰国旗如下：<image>\n左绿右橙。"}
            ],
        }
        errs = builder.validate_cpt_record(bad_prefix)
        self.assertIn("causal_prefix_violation:text_before_first_image", errs)

        bad_count = {
            "sample_id": "bad_2",
            "doc_family_id": "fam_bad_2",
            "channel": "ch3_qwen_img_synth",
            "images": ["a.png", "b.png"],
            "messages": [{"role": "assistant", "content": "<image>\n只有一张占位符。"}],
        }
        errs = builder.validate_cpt_record(bad_count)
        self.assertTrue(any(e.startswith("image_placeholder_mismatch:") for e in errs))

        bad_role = {
            "sample_id": "bad_3",
            "doc_family_id": "fam_bad_3",
            "channel": "ch1_wiki_text_stub",
            "images": [],
            "messages": [
                {"role": "user", "content": "问题"},
                {"role": "assistant", "content": "回答"},
            ],
        }
        self.assertIn("cpt_requires_single_assistant_message", builder.validate_cpt_record(bad_role))

    def test_demo_bundle_zero_leakage_and_stage2_tail_focus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "bundle"
            records = builder.build_demo_records()
            manifest = builder.build_cpt_bundle(
                records,
                out_dir,
                val_ratio=0.17,
                stage1_rows=24,
                stage2_rows=18,
                tail_oversample=2,
                seed=42,
            )
            self.assertEqual(manifest["doc_family_overlap"], 0)
            for key in ("stage1_train", "stage2_cooldown", "val"):
                info = manifest["artifacts"][key]
                path = Path(info["path"])
                self.assertTrue(path.exists())
                self.assertEqual(info["sha256"], builder.sha256_file(path))

            stage2_rows = builder.load_jsonl(out_dir / "stage2_cooldown.jsonl")
            ch_text_rows = [r for r in stage2_rows if r["channel"] == "ch_text_doc"]
            for r in ch_text_rows:
                # When unresolved tail stubs exist in ch_text_doc, Stage 2 drops solved head stubs.
                self.assertTrue(r.get("is_unresolved_tail", False))
            self.assertEqual(manifest["length_guard"]["dropped_rows"], 0)
            for stats in manifest["artifacts"]["stage1_train"]["summary"]["channels"].values():
                self.assertIn("effective_repeat_factor", stats)
                self.assertIn("unique_rows", stats)

    def test_length_guard_and_repeat_cap_protect_rare_channels(self) -> None:
        base = builder.build_demo_records()
        overlong = {
            "sample_id": "overlong_text_1",
            "doc_family_id": "fam_overlong_1",
            "channel": "ch_text_doc",
            "images": [],
            "messages": [{"role": "assistant", "content": "超" * 30000}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            # 1. Overlong row dropped & recorded; high target rows triggers repeat warning
            m_warn = builder.build_cpt_bundle(
                base + [overlong],
                tmp_dir / "warn_run",
                val_ratio=0.17,
                stage1_rows=20,
                stage2_rows=40,
                max_tokens=16384,
                seed=42,
            )
            self.assertEqual(m_warn["length_guard"]["dropped_rows"], 1)
            self.assertEqual(m_warn["length_guard"]["dropped_by_channel"], {"ch_text_doc": 1})
            self.assertTrue(len(m_warn["warnings"]) >= 1)

            # 2. Capping repeats keeps effective_repeat_factor <= max_row_repeats
            m_cap = builder.build_cpt_bundle(
                base,
                tmp_dir / "cap_run",
                val_ratio=0.17,
                stage1_rows=20,
                stage2_rows=40,
                max_tokens=16384,
                max_row_repeats=2,
                seed=42,
            )
            for stats in m_cap["artifacts"]["stage2_cooldown"]["summary"]["channels"].values():
                self.assertLessEqual(stats["effective_repeat_factor"], 2.0)

    def test_native_configs_valid_json_and_guardrails(self) -> None:
        m_cfg = json.loads(self.megatron_cfg_path.read_text(encoding="utf-8"))
        self.assertEqual(m_cfg["tuner_type"], "full")
        self.assertEqual(m_cfg["tensor_model_parallel_size"], 4)
        self.assertEqual(m_cfg["global_batch_size"], 32)
        self.assertEqual(m_cfg["mtp_num_layers"], 0)
        self.assertTrue(m_cfg["freeze_vit"])
        self.assertFalse(m_cfg["freeze_aligner"])
        self.assertTrue(m_cfg["enable_channel_loss"])
        self.assertEqual(m_cfg["lr_decay_style"], "WSD")

        s_cfg = json.loads(self.swift_cfg_path.read_text(encoding="utf-8"))
        self.assertEqual(s_cfg["tuner_type"], "lora")
        self.assertTrue(s_cfg["freeze_vit"])
        self.assertFalse(s_cfg["freeze_aligner"])
        self.assertTrue(s_cfg["enable_channel_loss"])

    def _run_cpt_script(self, extra_env: dict[str, str], check: bool = True) -> subprocess.CompletedProcess[str]:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin"),
            "DATASET": str(self.example_path),
            "DRY_RUN": "1",
        }
        env.update(extra_env)
        return subprocess.run(
            ["bash", str(self.cpt_script)],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=check,
        )

    def test_b300_27b_cpt_script_dry_run_variants_and_guards(self) -> None:
        syntax = subprocess.run(["bash", "-n", str(self.cpt_script)], capture_output=True, text=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

        # 1. PHASE=stage1 uses constant LR + 6% warmup + TP=4 + MTP=0
        p_s1 = self._run_cpt_script(
            {"BACKEND": "megatron", "PHASE": "stage1", "TUNER_TYPE": "full"}
        )
        self.assertIn("--lr_decay_style constant", p_s1.stdout)
        self.assertIn("--lr_warmup_fraction 0.06", p_s1.stdout)
        self.assertIn("--tensor_model_parallel_size 4", p_s1.stdout)
        self.assertIn("--mtp_num_layers 0", p_s1.stdout)

        # 2. PHASE=stage2 with STAGE2_DECAY_ITERS=105 uses WSD linear decay
        p_s2 = self._run_cpt_script(
            {
                "BACKEND": "megatron",
                "PHASE": "stage2",
                "TUNER_TYPE": "full",
                "STAGE2_DECAY_ITERS": "105",
            }
        )
        self.assertIn("--lr_decay_style WSD", p_s2.stdout)
        self.assertIn("--lr_wsd_decay_style linear", p_s2.stdout)
        self.assertIn("--lr_wsd_decay_iters 105", p_s2.stdout)

        # 3. Cold-start Stage 2 without saved optimizer gets automatic 5% warmup buffer
        p_s2_cold = self._run_cpt_script(
            {
                "BACKEND": "megatron",
                "PHASE": "stage2",
                "TUNER_TYPE": "full",
                "STAGE2_DECAY_ITERS": "25",
                "NO_SAVE_OPTIM": "true",
            }
        )
        self.assertIn("--lr_warmup_fraction 0.05", p_s2_cold.stdout)

        # 4. Megatron full WSD mode with explicit TRAIN_ITERS
        p_full = self._run_cpt_script(
            {
                "BACKEND": "megatron",
                "TUNER_TYPE": "full",
                "TRAIN_ITERS": "1000",
                "LR_DECAY_STYLE": "WSD",
                "VAL_DATASET": str(self.example_path),
            }
        )
        self.assertIn("megatron pt", p_full.stdout)
        self.assertIn("--lr_decay_style WSD", p_full.stdout)
        self.assertIn("--lr_wsd_decay_style linear", p_full.stdout)
        self.assertIn("--enable_channel_loss true", p_full.stdout)
        self.assertIn("--freeze_vit true", p_full.stdout)
        self.assertIn("--freeze_aligner false", p_full.stdout)

        # 5. Megatron LoRA smoke mode must omit --num_train_epochs
        p_smoke = self._run_cpt_script(
            {"BACKEND": "megatron", "TUNER_TYPE": "lora", "SMOKE": "1"}
        )
        self.assertIn("--train_iters 10", p_smoke.stdout)
        self.assertNotIn("--num_train_epochs", p_smoke.stdout)

        # 6. SWIFT HF deep MLP regex mode
        p_swift = self._run_cpt_script(
            {"BACKEND": "swift", "TUNER_TYPE": "lora", "LORA_SCOPE": "deep_mlp_regex"}
        )
        self.assertIn("swift pt", p_swift.stdout)
        self.assertIn("--target_regex", p_swift.stdout)

        # 7. Guardrails exit non-zero on invalid combinations (including TP=8 GQA check)
        bad_tp8 = self._run_cpt_script({"BACKEND": "megatron", "TP_SIZE": "8"}, check=False)
        self.assertEqual(bad_tp8.returncode, 2)

        bad_model = self._run_cpt_script({"MODEL": "Qwen/Qwen3.8-Flash-Next"}, check=False)
        self.assertEqual(bad_model.returncode, 2)

        bad_megatron_regex = self._run_cpt_script(
            {"BACKEND": "megatron", "TUNER_TYPE": "lora", "LORA_SCOPE": "deep_mlp_regex"},
            check=False,
        )
        self.assertEqual(bad_megatron_regex.returncode, 2)

        bad_wsd = self._run_cpt_script(
            {"BACKEND": "megatron", "LR_DECAY_STYLE": "WSD"},
            check=False,
        )
        self.assertEqual(bad_wsd.returncode, 2)

    def test_hf_deep_mlp_regex_with_check_target_regex_tool(self) -> None:
        script_text = self.cpt_script.read_text(encoding="utf-8")
        m = re.search(r"DEFAULT_HF_DEEP_MLP_REGEX='([^']+)'", script_text)
        self.assertIsNotNone(m)
        pattern = m.group(1)

        sample_names = [
            "model.visual.blocks.0.attn.qkv",
            "model.visual.merger.linear_fc1",
            "model.language_model.layers.0.mlp.gate_proj",
            "model.language_model.layers.21.mlp.down_proj",
            "model.language_model.layers.22.mlp.gate_proj",
            "model.language_model.layers.35.self_attn.q_proj",
            "model.language_model.layers.63.mlp.down_proj",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            names_file = Path(tmp) / "modules.txt"
            names_file.write_text("\n".join(sample_names) + "\n", encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "check_target_regex.py"),
                    pattern,
                    str(names_file),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            matched = set(proc.stdout.strip().splitlines())
            self.assertEqual(
                matched,
                {
                    "model.visual.merger.linear_fc1",
                    "model.language_model.layers.22.mlp.gate_proj",
                    "model.language_model.layers.63.mlp.down_proj",
                },
            )


if __name__ == "__main__":
    unittest.main()
