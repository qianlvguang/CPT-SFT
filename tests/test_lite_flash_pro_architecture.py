#!/usr/bin/env python3
"""Unit and integration tests for the v2.0 Single-Source Lite / Flash / Pro + CoT v0.4 + ImageGen v2 architecture."""
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
import build_mm_wiki_cpt as cpt_builder  # noqa: E402
import build_v2_image_pipeline as img_v2  # noqa: E402
import enhance_flash_to_pro as pro_enh  # noqa: E402
import error_cluster_miner as pro_miner  # noqa: E402
import gen_cot_from_qa as cot_gen  # noqa: E402
import slice_flash_to_lite as lite_slicer  # noqa: E402


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
        # Calibrated Lite profile: MBS=1, GBS=4, MAX_LENGTH=8192, constant 8e-6, SAVE_STEPS=40, val0
        lite_proc = subprocess.run(
            ["bash", str(wrapper), "lite", "--model", model_path],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--gradient_accumulation_fusion false", lite_proc.stdout)
        self.assertIn("--micro_batch_size 1", lite_proc.stdout)
        self.assertIn("--global_batch_size 4", lite_proc.stdout)
        self.assertIn("--tensor_model_parallel_size 4", lite_proc.stdout)
        self.assertIn("--max_length 8192", lite_proc.stdout)
        self.assertIn("--lr 8e-6", lite_proc.stdout)
        self.assertIn("--save_steps 40", lite_proc.stdout)
        self.assertIn("--num_train_epochs 2", lite_proc.stdout)
        self.assertIn("--lr_decay_style constant", lite_proc.stdout)
        self.assertIn("--lr_warmup_fraction 0.06", lite_proc.stdout)
        self.assertNotIn("--val_dataset", lite_proc.stdout)

        # Flash single-stage WSD profile: MBS=2, GBS=16, MAX_LENGTH=8192, linear WSD
        flash_proc = subprocess.run(
            ["bash", str(wrapper), "flash", "--model", model_path],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--micro_batch_size 2", flash_proc.stdout)
        self.assertIn("--global_batch_size 16", flash_proc.stdout)
        self.assertIn("--max_length 8192", flash_proc.stdout)
        self.assertIn("--lr 8e-6", flash_proc.stdout)
        self.assertIn("--lr_decay_style WSD", flash_proc.stdout)
        self.assertIn("--lr_wsd_decay_style linear", flash_proc.stdout)
        self.assertIn("--lr_wsd_decay_iters 60", flash_proc.stdout)

        # Even if --launch is passed, explicit --dry-run keeps stage2 safe
        safe_launch = subprocess.run(
            ["bash", str(wrapper), "stage2", "--model", model_path, "--stage2-decay-iters", "28", "--launch", "--dry-run"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--lr_decay_style WSD", safe_launch.stdout)
        self.assertIn("--lr_wsd_decay_iters 28", safe_launch.stdout)

        # Pro enhanced Mid-Training profile (WSD 6e-6) + Pro incremental profile (cosine 5e-6)
        pro_proc = subprocess.run(
            ["bash", str(wrapper), "pro", "--model", model_path],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--lr 6e-6", pro_proc.stdout)
        self.assertIn("--lr_decay_style WSD", pro_proc.stdout)

        pro_inc_proc = subprocess.run(
            ["bash", str(wrapper), "pro_inc", "--model", model_path],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--lr 5e-6", pro_inc_proc.stdout)
        self.assertIn("--lr_decay_style cosine", pro_inc_proc.stdout)

    def test_lite_sch_iso_elements_override_and_val0(self) -> None:
        sch, kb, supp = lite_builder.build_demo_inputs()
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
            self.assertEqual((out_dir / "bundle_lite_v3_all" / "lite_val.jsonl").read_text(encoding="utf-8"), "")

            compat_link = out_dir / "bundle_lite_compat" / "stage1_train.jsonl"
            self.assertTrue(compat_link.is_symlink())
            rows = lite_builder.load_jsonl(compat_link)
            self.assertEqual(len(rows), manifest["rows"])

            by_sid = {r["sample_id"]: r for r in rows}
            neg_row = by_sid["demo_sch_label_mismatch_02"]
            self.assertEqual(neg_row["channel"], "ch_ip_negative")
            self.assertTrue(neg_row["audit_raw_label"])
            self.assertIn("元素：无。", neg_row["messages"][0]["content"])
            self.assertTrue(neg_row["messages"][0]["content"].startswith("<image>\n"))

            ground_row = by_sid["demo_sch_ground_01"]
            self.assertEqual(ground_row["channel"], "ch_ip_grounding")
            self.assertNotIn("错误的改写版", ground_row["messages"][0]["content"])
            self.assertEqual(by_sid["demo_sch_raft_03"]["channel"], "ch5_flash_rag_raft")

            sch_report = json.loads((out_dir / "build_sch_report.json").read_text(encoding="utf-8"))
            self.assertEqual(sch_report["label_true_but_empty_elements_count"], 1)

    def test_flash_qa_cot_v03_8gates_remapping_and_fallback(self) -> None:
        raw_records = cot_gen.load_jsonl(FLASH_DIR / "examples" / "qa_kb_raw_example.jsonl")
        failing_rec = json.loads(json.dumps(raw_records[0], ensure_ascii=False))
        failing_rec["sample_id"] = "qa_force_fail_07"
        failing_rec["doc_family_id"] = "fam_force_fail_07"
        failing_rec["_mock_force_fail"] = True

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            manifest = cot_gen.run_cot_generation_pipeline(
                qa_records=raw_records + [failing_rec],
                output_dir=out_dir,
                cot_mode="rag_citation",
                dedup=False,
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

        # Explicit unit checks on individual v0.3 8-gate failure modes + soft upper bound
        src_text = "爱尔兰国旗自旗杆侧起依次为绿、白、橙三色竖条，标准宽高比1:2。普通条纹不适用。"
        dist_text = "科特迪瓦国旗自旗杆侧起依次为橙、白、绿三色竖条。"
        source_map = {
            "S1": {"source_key": "S1", "text": dist_text, "content_sha256": cot_gen.sha256_text(dist_text)},
            "S2": {"source_key": "S2", "text": src_text, "content_sha256": cot_gen.sha256_text(src_text)},
        }
        evidence = [{"source_key": "S2", "quote": "绿、白、橙三色竖条", "content_sha256": cot_gen.sha256_text(src_text)}]

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

    def test_flash_qa_cot_v04_internalized_clean_twins_and_cpt_prose(self) -> None:
        v04_records = cot_gen.load_jsonl(FLASH_DIR / "examples" / "qa_kb_v04_example.jsonl")
        # 7 rows include 1 intentional duplicate of row 0; append 1 distinct forcing fail row
        failing_rec = json.loads(json.dumps(v04_records[1], ensure_ascii=False))
        failing_rec["sample_id"] = "qa_v04_force_fail_test"
        failing_rec["question"] = "强制失败样本：缺少证明时是否可以直接开审？"
        failing_rec["_mock_force_fail"] = True

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            manifest = cot_gen.run_cot_generation_pipeline(
                qa_records=v04_records + [failing_rec],
                output_dir=out_dir,
                cot_mode="internalized",
                mode="mock",
                smoke_sample=6,
                seed=42,
            )
            self.assertEqual(manifest["schema"], "qa_cot_v0.4")
            self.assertEqual(manifest["dedup_dropped_rows"], 1)
            self.assertEqual(manifest["cot_passed_rows"], 6)
            self.assertEqual(manifest["clean_direct_twin_rows"], 6)
            self.assertEqual(manifest["direct_fallback_rows"], 1)
            self.assertEqual(manifest["midtrain_prose_rows"], 6)
            self.assertGreaterEqual(manifest["direct_bucket_clean_share"], 0.8)
            self.assertEqual(manifest["avg_entity_anchor_coverage"], 1.0)

            # Inspect passed CoT: zero `[Sx]`, zero `"根据材料"`, pure entity omitted from user turn
            for r in cot_gen.load_jsonl(out_dir / "qa_cot_passed.jsonl"):
                usr = r["messages"][1]["content"]
                ast = r["messages"][2]["content"]
                self.assertNotRegex(ast, r"\[S\d+\]")
                self.assertNotIn("根据材料", ast)
                if r["sample_id"] == "qa_v04_entity_fact_01":
                    self.assertNotIn("【本次审核给定规则】", usr)

            # Clean direct twins have zero `<think>` and open with rationale anchor line
            for twin in cot_gen.load_jsonl(out_dir / "qa_direct_clean_twins.jsonl"):
                self.assertFalse(twin["enable_thinking"])
                self.assertEqual(twin["direct_source"], "clean_twin")
                self.assertNotIn("<think>", twin["messages"][-1]["content"])

            # CPT prose rows are strictly 1-message assistant prose (zero `<think>`, zero `[Sx]`/`[Rx]`)
            for prose in cot_gen.load_jsonl(out_dir / "qa_midtrain_prose.jsonl"):
                self.assertEqual(prose["channel"], "ch_qa_knowledge_prose")
                self.assertEqual(len(prose["messages"]), 1)
                self.assertEqual(prose["messages"][0]["role"], "assistant")
                txt = prose["messages"][0]["content"]
                self.assertNotIn("<think>", txt)
                self.assertNotRegex(txt, r"\[(?:S|R)\d+\]")

        # Clause-level polarity & RAG-leak unit checks on verify_cot_v04_8gates
        bad_v04 = cot_gen.verify_cot_v04_8gates(
            reasoning="根据材料[S1]所示，我们确认存在干扰实体雪鼠W。前面先排除其他。但没有覆盖实体特征。" + "冗余填充" * 45,
            final_answer="结论。[S1]",
            clean_gold_answer="结论。",
            task_type="fact",
            answerable=True,
            gold_entity_anchors=["黄色体毛", "红色圆形电气囊", "黑色耳尖"],
            entity_sources=[{"title": "示例电鼠P", "text": "描述"}],
            entity_distractor_subjects=["示例雪鼠W"],
            rule_gold_keys=[],
            rule_distractor_keys=[],
            rule_source_map={},
            entity_source_map={},
        )
        self.assertFalse(bad_v04["passed"])
        self.assertFalse(bad_v04["gates"]["gate2_anti_rag_leak"])
        self.assertFalse(bad_v04["gates"]["gate3_entity_anchor_coverage"])
        self.assertFalse(bad_v04["gates"]["gate4_distractor_discrimination"])

    def test_v2_image_pipeline_routing_sports_recycle_and_packing(self) -> None:
        entities = img_v2.load_jsonl(FLASH_DIR / "examples" / "v2_image_entities_sample.jsonl")
        v1_audit = img_v2.load_jsonl(FLASH_DIR / "examples" / "v2_image_v1_audit_sample.jsonl")

        # Missing vector base on flag must return pending_vector_asset (forbid pure T2I)
        no_ref_flag = img_v2.route_entity_domain({"entity_id": "flag_no_ref", "domain": "flag", "mm": False, "ref_images": []})
        self.assertEqual(no_ref_flag["action"], "pending_vector_asset")
        self.assertIsNone(no_ref_flag["endpoint"])
        self.assertTrue(no_ref_flag["forbid_pure_t2i"])

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            manifest = img_v2.run_v2_image_pipeline(
                entities=entities,
                v1_audit_records=v1_audit,
                output_dir=out_dir,
                mode="mock",
            )
            self.assertEqual(manifest["sports_expanded_children"], 2)
            self.assertEqual(manifest["action_counts"]["stop"], 3)
            self.assertEqual(manifest["action_counts"]["throttle_skip"], 1)
            self.assertEqual(manifest["flag_pure_t2i_violations"], 0)
            self.assertEqual(manifest["max_doc_text_repeat"], 1)
            self.assertGreater(manifest["packed_cpt_doc_rows"], 5)
            self.assertGreaterEqual(manifest["cf_neg_pair_rows"], 2)
            self.assertEqual(
                manifest["v1_recycle_counts"],
                {"prune_discard": 2, "recycle_as_cf_negative": 2, "recycle_to_v2_positive": 1},
            )
            for row in img_v2.load_jsonl(out_dir / "v2_packed_cpt_docs.jsonl"):
                self.assertEqual(row["doc_text_repeat"], 1)
                self.assertEqual(row["messages"][0]["content"].count("<image>"), len(row["images"]))

    def test_midtrain_bundle_qa_risk_and_synth_image_guards(self) -> None:
        base_records = cpt_builder.build_demo_records()
        # Add raw ChatML `<think>` QA (must auto-sanitize!), SFT-eval overlapping prose, and synthetic multi-img docs
        extra_records = [
            {
                "sample_id": "raw_chatml_leak_risk",
                "doc_family_id": "fam_raw_chatml_01",
                "channel": "wiki_rag_qa_cot",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "问题：电鼠颜色？"},
                    {"role": "assistant", "content": "<think>\n推演过程。\n</think>\n\n黄色。"},
                ],
            },
            {
                "sample_id": "raw_chatml_leak_risk_2",
                "doc_family_id": "fam_raw_chatml_02",
                "channel": "wiki_rag_qa_cot",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "问题：国旗色序？"},
                    {"role": "assistant", "content": "<think>\n推演过程2。\n</think>\n\n绿白橙。"},
                ],
            },
            {
                "sample_id": "eval_overlap_prose",
                "doc_family_id": "fam_eval_sealed_01",
                "channel": "ch_qa_knowledge_prose",
                "is_qa_prose": True,
                "messages": [{"role": "assistant", "content": "应当被排除的评测同家族散文。"}],
            },
            {
                "sample_id": "synth_heavy_doc_01",
                "doc_family_id": "fam_synth_heavy_01",
                "channel": "ch_v2_multi_img_doc",
                "images": ["/a.png", "/b.png", "/c.png", "/d.png"],
                "is_synthetic_image": True,
                "synthetic_image_count": 3,
                "doc_text_repeat": 1,
                "messages": [{"role": "assistant", "content": "<image>\n<image>\n<image>\n<image>\n多图实体描述。"}],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            manifest = cpt_builder.build_midtrain_bundle(
                base_records + extra_records,
                Path(tmp),
                val_ratio=0.18,
                max_tokens=8192,
                max_row_repeats=2,
                max_synth_image_share=0.25,
                max_qa_prose_share=0.15,
                allow_raw_chatml_in_cpt=False,
                excluded_eval_families={"fam_eval_sealed_01"},
            )
            self.assertEqual(manifest["doc_family_overlap"], 0)
            self.assertLessEqual(manifest["synthetic_image_audit"]["synth_image_share"], 0.25)
            audit = manifest["qa_cpt_risk_audit"]
            self.assertEqual(audit["raw_chatml_in_cpt_count"], 0)
            self.assertEqual(audit["raw_chatml_sanitized_to_prose"], 2)
            self.assertLessEqual(audit["qa_prose_token_share"], 0.15)
            self.assertEqual(audit["eval_family_dropped"], 1)
            self.assertEqual(audit["eval_family_overlap"], 0)

    def test_lite_proportional_slice_and_orchestrators_dry_run(self) -> None:
        pool = lite_slicer._demo_flash_midtrain_pool()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            manifest = lite_slicer.slice_flash_to_lite(pool, out_dir, ratio=0.25)
            self.assertEqual(manifest["val_ratio"], 0.0)
            self.assertEqual(manifest["partial_family_splits"], 0)
            self.assertEqual(manifest["b300_step_projection"]["global_batch_size"], 4)
            self.assertEqual(manifest["b300_step_projection"]["max_length"], 8192)
            compat_link = out_dir / "bundle_lite_compat" / "stage1_train.jsonl"
            self.assertTrue(compat_link.is_symlink())
            self.assertEqual(len(lite_slicer.load_jsonl(compat_link)), manifest["lite_rows"])

            # End-to-end DRY_RUN checks on Lite (both modes) and Pro orchestrators
            for mode in ("proportional_flash", "sch_legacy"):
                work_lite = out_dir / f"orch_lite_{mode}"
                env = {
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin"),
                    "WORK_DIR": str(work_lite),
                    "LITE_MODE": mode,
                    "DEMO": "1",
                    "DRY_RUN": "1",
                }
                proc = subprocess.run(
                    ["bash", str(LITE_DIR / "run_lite_pipeline.sh")],
                    cwd=str(REPO_ROOT),
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertIn("--micro_batch_size 1", proc.stdout)
                self.assertIn("--global_batch_size 4", proc.stdout)
                self.assertIn("--max_length 8192", proc.stdout)

            work_pro = out_dir / "orch_pro"
            pro_proc = subprocess.run(
                ["bash", str(PRO_DIR / "run_pro_pipeline.sh")],
                cwd=str(REPO_ROOT),
                env={
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin"),
                    "WORK_DIR": str(work_pro),
                    "DEMO": "1",
                    "DRY_RUN": "1",
                },
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("--lr 6e-6", pro_proc.stdout)
            self.assertIn("--lr 5e-6", pro_proc.stdout)
            self.assertTrue((work_pro / "step1_pro_enhanced" / "pro_enhance_manifest.json").exists())
            self.assertTrue((work_pro / "step3_error_miner" / "iteration_ledger.json").exists())

    def test_pro_enhancer_and_multiround_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp) / "scratch"
            enh_out = Path(tmp) / "enh_unhealthy"
            bundle = pro_enh._build_demo_flash_inputs(scratch)

            # Unhealthy canary (general drop >2pp & Think-OFF collapse gap >8pp)
            manifest = pro_enh.enhance_flash_to_pro(
                flash_train_rows=bundle["flash_train"],
                flash_val_rows=bundle["flash_val"],
                cot_passed=bundle["cot_passed"],
                cot_rejected=bundle["cot_rejected"],
                v2_docs=bundle["v2_docs"],
                output_dir=enh_out,
                canary_metrics={
                    "base_general_acc": 76.5,
                    "cpt_general_acc": 73.5,
                    "base_think_off_acc": 74.0,
                    "cpt_think_on_acc": 85.0,
                    "cpt_think_off_acc": 70.0,
                },
            )
            self.assertEqual(manifest["doc_family_overlap"], 0)
            self.assertLessEqual(manifest["curation"]["qa_prose_token_share"], 0.15)
            self.assertLessEqual(manifest["curation"]["synthetic_image_audit"]["synth_image_share"], 0.25)
            self.assertGreater(manifest["alignment_summary"]["cross_modal_aligned_count"], 0)
            self.assertTrue(manifest["canary_and_wise_ft"]["wise_ft_merge_triggered"])
            self.assertEqual(manifest["canary_and_wise_ft"]["formula"], "0.8 * CPT + 0.2 * Base")
            self.assertTrue(manifest["canary_and_wise_ft"]["think_off_collapse_risk"])
            self.assertTrue(manifest["canary_and_wise_ft"]["opd_think_on_to_off_distill_recommended"])

            opd_seeds = cpt_builder.load_jsonl(enh_out / "pro_opd_bsd_seeds.jsonl")
            self.assertGreater(len(opd_seeds), 0)
            for s in opd_seeds:
                self.assertEqual(s["distill_mode"], "think_on_to_off_context_distill")
                self.assertTrue(s["teacher_enable_thinking"])
                self.assertFalse(s["student_enable_thinking"])

            healthy_gate = pro_enh.evaluate_canary_and_wise_ft(
                {
                    "base_general_acc": 76.0,
                    "cpt_general_acc": 75.5,
                    "base_think_off_acc": 73.0,
                    "cpt_think_on_acc": 80.0,
                    "cpt_think_off_acc": 75.5,
                }
            )
            self.assertFalse(healthy_gate["wise_ft_merge_triggered"])
            self.assertFalse(healthy_gate["think_off_collapse_risk"])

    def test_pro_error_cluster_miner_50_30_20_and_multiround_ledger(self) -> None:
        inputs = pro_miner.load_demo_records()
        with tempfile.TemporaryDirectory() as tmp:
            r1_dir = Path(tmp) / "round1"
            m1 = pro_miner.run_pro_pipeline(
                error_logs=inputs["error_logs"],
                accepted_templates=inputs["accepted_templates"],
                v_replay=inputs["v_replay"],
                sft_candidates=inputs["sft_candidates"],
                output_dir=r1_dir,
                total_cpt_rows=20,
                val_ratio=0.2,
                round_id="round_01",
                parent_ckpt="flash_v2_hf",
                seed=42,
            )
            self.assertEqual(m1["sft_overlap"], 0)
            self.assertEqual(m1["index_prescriptions_count"], 2)
            self.assertEqual(m1["actual_cpt_mix_shares"]["new_verified_gap"], 0.5)
            self.assertEqual(m1["actual_cpt_mix_shares"]["accepted_template_replay"], 0.3)
            self.assertEqual(m1["actual_cpt_mix_shares"]["v_series_replay"], 0.2)
            self.assertEqual(m1["sft_excluded_rows"], 1)

            # Round 2 adds an SFT candidate whose template_family_id was excluded in Round 1 -> must be dropped
            r1_ledger = r1_dir / "iteration_ledger.json"
            r1_excluded_tpl = json.loads(r1_ledger.read_text(encoding="utf-8"))["cumulative_excluded_template_families"][0]
            leaked_cand = {
                "sample_id": "sft_cross_round_leak",
                "template_family_id": r1_excluded_tpl,
                "doc_family_id": "fresh_doc_r2",
                "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}],
            }
            r2_dir = Path(tmp) / "round2"
            m2 = pro_miner.run_pro_pipeline(
                error_logs=inputs["error_logs"],
                accepted_templates=inputs["accepted_templates"],
                v_replay=inputs["v_replay"],
                sft_candidates=inputs["sft_candidates"] + [leaked_cand],
                output_dir=r2_dir,
                total_cpt_rows=20,
                round_id="round_02",
                parent_ckpt="pro_r1_hf",
                prev_ledger=r1_ledger,
                seed=43,
            )
            self.assertEqual(m2["sft_overlap"], 0)
            self.assertEqual(m2["sft_excluded_rows"], 2)
            r2_ledger = json.loads((r2_dir / "iteration_ledger.json").read_text(encoding="utf-8"))
            self.assertEqual(len(r2_ledger["rounds"]), 2)
            self.assertGreaterEqual(
                len(r2_ledger["cumulative_excluded_template_families"]),
                len(json.loads(r1_ledger.read_text(encoding="utf-8"))["cumulative_excluded_template_families"]),
            )


if __name__ == "__main__":
    unittest.main()
