"""仅用标准库测试故障保护与调度；不模拟真实 GPU 数值训练。"""
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "training" / "qwen35_27b"

spec = importlib.util.spec_from_file_location("recommended", PIPELINE_DIR / "train_s1_s3_recommended.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def make_model(path, step=None):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text('{}')
    header = json.dumps({"weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b'\0\0')
    if step is not None:
        (path.parent / "latest_checkpointed_iteration.txt").write_text(str(step))


def make_adapter(path, step=None):
    path.mkdir(parents=True, exist_ok=True)
    config = {"peft_type": "LORA", "r": 64, "lora_alpha": 128, "bias": "none",
              "rank_pattern": {r"visual\.merger\..*": 32, r"visual\..*": 16},
              "alpha_pattern": {r"visual\.merger\..*": 64, r"visual\..*": 32}}
    (path / "adapter_config.json").write_text(json.dumps(config))
    header = {}
    offset = 0
    for module in ["base_model.model.visual.merger.proj", "base_model.model.language_model.proj"]:
        for matrix in ["A", "B"]:
            header[f"{module}.lora_{matrix}.weight"] = {
                "dtype": "BF16", "shape": [1, 1], "data_offsets": [offset, offset + 2]}
            offset += 2
    encoded = json.dumps(header).encode()
    (path / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + b'\0' * offset)
    if step is not None:
        (path.parent / "latest_checkpointed_iteration.txt").write_text(str(step))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.say_patch = patch.object(pipeline, "say")
        self.say_patch.start()
        self.addCleanup(self.say_patch.stop)

    def test_config_only_and_truncated_model_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            (p / "config.json").write_text('{}')
            with self.assertRaises(FileNotFoundError):
                pipeline.model_fingerprint(p)
            make_model(p)
            f = p / "model.safetensors"
            f.write_bytes(f.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                pipeline.model_fingerprint(p)

    def test_shard_index_requires_real_tensors(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            make_model(p)
            (p / "model.safetensors.index.json").write_text(json.dumps(
                {"weight_map": {"missing_tensor": "model.safetensors"}}))
            with self.assertRaises(ValueError):
                pipeline.model_fingerprint(p)

    def test_weight_content_replacement_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            make_model(p)
            original = pipeline.model_fingerprint(p)
            f = p / "model.safetensors"
            f.write_bytes(f.read_bytes()[:-2] + b'\1\0')
            self.assertNotEqual(original, pipeline.model_fingerprint(p))

    def test_selection_excludes_before_one_epoch(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            for step in [25, 50, 75, 100]:
                make_adapter(p / f"checkpoint-{step}", step)
            evaluator = p / "evaluate.sh"
            evaluator.write_text('''#!/usr/bin/env bash
case "$1" in
  *checkpoint-25) score=0.99 ;;
  *checkpoint-75) score=0.90 ;;
  *) score=0.85 ;;
esac
printf '{"dataset_id":"eval_v3_5658","accuracy":%s,"f1":0.8}' "$score" > "$3"
''')
            chosen, selection, _ = pipeline.select_checkpoint(p, pipeline.RECIPE["stages"][0],
                                                              evaluator, False, os.environ, p)
            self.assertEqual(chosen.name, "checkpoint-75")
            self.assertEqual(selection["mode"], "eval_v3_5658")
            chosen, _, _ = pipeline.select_checkpoint(p, pipeline.RECIPE["stages"][2],
                                                       evaluator, False, os.environ, p)
            self.assertEqual(chosen.name, "checkpoint-100")

    def test_smoke_omits_epochs_and_frozen_stage_visual_lrs(self):
        cmd = pipeline.stage_command(pipeline.RECIPE["stages"][1], "parent", [{"path": "data"}],
                                     "out", "plugin", True, 1000000, adapter="previous_adapter")
        self.assertNotIn("--num_train_epochs", cmd)
        self.assertIn("--train_iters", cmd)
        self.assertNotIn("--vit_lr", cmd)
        self.assertNotIn("--aligner_lr", cmd)
        self.assertEqual(cmd[cmd.index("--finetune") + 1], "true")
        self.assertEqual(cmd[cmd.index("--merge_lora") + 1], "false")
        self.assertEqual(cmd[cmd.index("--adapters") + 1], "previous_adapter")
        self.assertEqual(cmd[cmd.index("--freeze_vit") + 1], "false")
        self.assertEqual(cmd[cmd.index("--vit_gradient_checkpointing") + 1], "false")

    def test_chain_reuse_and_changed_data_guard(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            make_model(p / "base")
            report = [{"stage": k, "variant": v, "path": str(p / f"s{k}v{v}.parquet"),
                       "rows": 10, "sha256": f"digest{k}{v}"} for k in [1, 2, 3] for v in [1, 2, 3, 4]]
            calls = []
            def training(cmd, env, cwd, log):
                calls.append(cmd)
                out = Path(cmd[cmd.index("--output_dir") + 1])
                make_adapter(out / "checkpoint-100", 100)
            env = {"WORK_DIR": str(p), "BASE_MODEL": str(p / "base"), "TEST_DIR": str(p),
                   "OUTPUT_ROOT": str(p / "output"), "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}
            with patch.dict(os.environ, env, clear=True), patch.object(pipeline.sys, "argv", ["script", "run"]), \
                 patch.object(pipeline, "preflight", return_value=report), \
                 patch.object(pipeline, "make_replay", return_value={"path": "replay", "rows": 10, "sha256": "replayhash"}), \
                 patch.object(pipeline, "runtime_versions", return_value={"ms-swift": "4.3.2"}), \
                 patch.object(pipeline.shutil, "which", return_value="megatron"), \
                 patch.object(pipeline.shutil, "disk_usage", return_value=type("Disk", (), {"free": 2 * 1024**4})()), \
                 patch.object(pipeline, "run_logged", side_effect=training):
                pipeline.main()
                self.assertEqual(len(calls), 3)
                for cmd in calls:
                    self.assertEqual(Path(cmd[cmd.index("--model") + 1]), (p / "base").resolve())
                    self.assertEqual(cmd[cmd.index("--merge_lora") + 1], "false")
                self.assertNotIn("--adapters", calls[0])
                self.assertEqual(Path(calls[1][calls[1].index("--adapters") + 1]),
                                 (p / "output/s1/checkpoint-100").resolve())
                self.assertEqual(Path(calls[2][calls[2].index("--adapters") + 1]),
                                 (p / "output/s2/checkpoint-100").resolve())
                pipeline.main()
                self.assertEqual(len(calls), 3)
                report[0]["sha256"] = "changed-content"
                with self.assertRaisesRegex(RuntimeError, "已变化"):
                    pipeline.main()
                self.assertFalse((p / "output/.pipeline.lock").exists())

    def test_training_failure_stops_chain_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            make_model(p / "base")
            report = [{"stage": k, "variant": 1, "path": "data", "rows": 10, "sha256": "digest"} for k in [1, 2, 3]]
            env = {"WORK_DIR": str(p), "BASE_MODEL": str(p / "base"), "TEST_DIR": str(p), "OUTPUT_ROOT": str(p / "output")}
            with patch.dict(os.environ, env, clear=True), patch.object(pipeline.sys, "argv", ["script", "run"]), \
                 patch.object(pipeline, "preflight", return_value=report), \
                 patch.object(pipeline, "make_replay", return_value={"path": "replay", "rows": 10}), \
                 patch.object(pipeline, "runtime_versions", return_value={}), \
                 patch.object(pipeline.shutil, "which", return_value="megatron"), \
                 patch.object(pipeline.shutil, "disk_usage", return_value=type("Disk", (), {"free": 2 * 1024**4})()), \
                 patch.object(pipeline, "run_logged", side_effect=RuntimeError("训练失败")) as run:
                with self.assertRaisesRegex(RuntimeError, "训练失败"):
                    pipeline.main()
                self.assertEqual(run.call_count, 1)
                self.assertFalse((p / "output/s1/complete.json").exists())
                self.assertFalse((p / "output/.pipeline.lock").exists())
                self.assertFalse((p / "output/s2").exists())

    def test_adapter_must_preserve_visual_and_complete_pairs(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            make_adapter(p)
            digest = pipeline.adapter_fingerprint(p)
            self.assertIn("adapter_model.safetensors", digest)
            config = json.loads((p / "adapter_config.json").read_text())
            config["rank_pattern"] = {}
            (p / "adapter_config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "rank_pattern"):
                pipeline.adapter_fingerprint(p)

    def test_visual_adapter_kept_but_frozen_before_optimizer(self):
        """替代框架导入；真实验证 requires_grad/视觉 dropout 和挂载后调用顺序。"""
        class Parameter:
            requires_grad = True
            def requires_grad_(self, value):
                self.requires_grad = value
            def numel(self):
                return 1
        class Model:
            def __init__(self):
                self.parameters = {
                    "visual.merger.proj.lora_A.default.weight": Parameter(),
                    "visual.merger.proj.lora_B.default.weight": Parameter(),
                    "language_model.proj.lora_A.default.weight": Parameter(),
                    "language_model.proj.lora_B.default.weight": Parameter(),
                    "language_model.proj.base_layer.weight": Parameter()}
                self.dropout = types.SimpleNamespace(p=0.05)
            def named_parameters(self):
                return self.parameters.items()
            def named_modules(self):
                return [("visual.merger.proj", types.SimpleNamespace(lora_dropout={"default": self.dropout}))]
        class RowPreprocessor:
            def batched_preprocess(self, row, **kwargs):
                return row
        events = []
        class Trainer:
            args = types.SimpleNamespace(adapters=["previous_adapter"])
            def _prepare_peft_model(self, models):
                events.append("adapter_loaded")
                return models
        def lora_config(*args, **kwargs):
            return types.SimpleNamespace(**kwargs)
        modules = {
            "swift.dataset.preprocessor.core": types.SimpleNamespace(RowPreprocessor=RowPreprocessor),
            "swift.megatron.utils": types.SimpleNamespace(utils=types.SimpleNamespace(LoraConfig=lora_config)),
            "swift.megatron.trainers.base": types.SimpleNamespace(BaseMegatronTrainer=Trainer)}
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {"PIPELINE_STAGE": "2"}):
            source = PIPELINE_DIR / "recommended_pipeline_plugin.py"
            plugin_spec = importlib.util.spec_from_file_location("test_plugin", source)
            plugin = importlib.util.module_from_spec(plugin_spec)
            plugin_spec.loader.exec_module(plugin)
            model = Model()
            Trainer()._prepare_peft_model([model])
            self.assertEqual(events, ["adapter_loaded"])
            self.assertEqual(len(model.parameters), 5)
            self.assertFalse(model.parameters["visual.merger.proj.lora_A.default.weight"].requires_grad)
            self.assertTrue(model.parameters["language_model.proj.lora_A.default.weight"].requires_grad)
            self.assertFalse(model.parameters["language_model.proj.base_layer.weight"].requires_grad)
            self.assertEqual(model.dropout.p, 0)
            plugin.enforce_trainable_adapters([model], 1)
            self.assertTrue(model.parameters["visual.merger.proj.lora_A.default.weight"].requires_grad)


if __name__ == "__main__":
    unittest.main()
