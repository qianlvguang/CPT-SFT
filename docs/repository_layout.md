# 目录迁移说明

2026-11-08：将原 `qsa_muon_35b_a3b/.git` 移到整个 `CPT-SFT/.git`，保留原有提交链、分支和远程地址。旧父目录的 27B 脚本同时纳入这个工作树。

以下旧路径以整理前的本地 `CPT-SFT/` 为基准；新路径以当前仓库根目录为基准。脚本之间的相对引用和工作区文档已同步更新。

| 旧路径 | 新路径 |
|---|---|
| `qsa_muon_35b_a3b/.gitignore` | [.gitignore](../.gitignore) |
| `qsa_muon_35b_a3b/Inference/Model_Guide/01-模型结构与训练范围.md` | [docs/model_guide/model_structure_and_training_scope.md](../docs/model_guide/model_structure_and_training_scope.md) |
| `qsa_muon_35b_a3b/Inference/Model_Guide/02-FP8与NVFP4.md` | [docs/model_guide/fp8_and_nvfp4.md](../docs/model_guide/fp8_and_nvfp4.md) |
| `qsa_muon_35b_a3b/Inference/Model_Guide/03-PLE量化研究.md` | [docs/model_guide/ple_quantization.md](../docs/model_guide/ple_quantization.md) |
| `qsa_muon_35b_a3b/Inference/Model_Guide/04-权重检查工具.md` | [docs/model_guide/weight_inspection.md](../docs/model_guide/weight_inspection.md) |
| `qsa_muon_35b_a3b/Inference/Model_Guide/05-Qwen3.5-27B.md` | [docs/model_guide/qwen35_27b.md](../docs/model_guide/qwen35_27b.md) |
| `qsa_muon_35b_a3b/Inference/Model_Guide/README.md` | [docs/model_guide/README.md](../docs/model_guide/README.md) |
| `qsa_muon_35b_a3b/Inference/Model_Guide/inspect_safetensors.py` | [tools/inspect_safetensors.py](../tools/inspect_safetensors.py) |
| `qsa_muon_35b_a3b/Inference/README.md` | [inference/README.md](../inference/README.md) |
| `qsa_muon_35b_a3b/Inference/TRAINING.md` | [training/swift/README.md](../training/swift/README.md) |
| `qsa_muon_35b_a3b/Inference/check-target-regex.py` | [tools/check_target_regex.py](../tools/check_target_regex.py) |
| `qsa_muon_35b_a3b/Inference/example-sft.jsonl` | [training/swift/example_sft.jsonl](../training/swift/example_sft.jsonl) |
| `qsa_muon_35b_a3b/Inference/serve-b300.sh` | [inference/serve_b300.sh](../inference/serve_b300.sh) |
| `qsa_muon_35b_a3b/Inference/serve-common.sh` | [inference/serve_common.sh](../inference/serve_common.sh) |
| `qsa_muon_35b_a3b/Inference/serve-qwen35-27b.sh` | [inference/serve_qwen35_27b.sh](../inference/serve_qwen35_27b.sh) |
| `qsa_muon_35b_a3b/Inference/serve-rtx-pro-6000.sh` | [inference/serve_rtx_pro_6000.sh](../inference/serve_rtx_pro_6000.sh) |
| `qsa_muon_35b_a3b/Inference/smoke-test.py` | [tests/smoke_test.py](../tests/smoke_test.py) |
| `qsa_muon_35b_a3b/Inference/train-b300-swift.sh` | [training/swift/train_b300_swift.sh](../training/swift/train_b300_swift.sh) |
| `qsa_muon_35b_a3b/Inference/train-qwen35-27b-swift.sh` | [training/swift/train_qwen35_27b_swift.sh](../training/swift/train_qwen35_27b_swift.sh) |
| `qsa_muon_35b_a3b/README.md` | [docs/qsa_muon_35b_a3b.md](../docs/qsa_muon_35b_a3b.md) |
| `qsa_muon_35b_a3b/cuda13_pytorch_notes.md` | [environment/cuda13_pytorch_notes.md](../environment/cuda13_pytorch_notes.md) |
| `qsa_muon_35b_a3b/ds_zero2_h200.json` | [training/qsa_muon_35b_a3b/ds_zero2_h200.json](../training/qsa_muon_35b_a3b/ds_zero2_h200.json) |
| `qsa_muon_35b_a3b/mcore_qsa_bridge_and_distill_plugin.py` | [training/qsa_muon_35b_a3b/mcore_qsa_bridge_and_distill_plugin.py](../training/qsa_muon_35b_a3b/mcore_qsa_bridge_and_distill_plugin.py) |
| `qsa_muon_35b_a3b/qsa_muon_plugin.py` | [training/qsa_muon_35b_a3b/qsa_muon_plugin.py](../training/qsa_muon_35b_a3b/qsa_muon_plugin.py) |
| `qsa_muon_35b_a3b/qwen3_5_jev_multimodal_training_and_quantization_report.md` | [docs/reports/qwen3_5_jev_multimodal_training_and_quantization_report.md](../docs/reports/qwen3_5_jev_multimodal_training_and_quantization_report.md) |
| `qsa_muon_35b_a3b/requirements_h200.txt` | [environment/requirements_h200.txt](../environment/requirements_h200.txt) |
| `qsa_muon_35b_a3b/run_cpt_sft_qsa_35a3_h200x8.sh` | [training/qsa_muon_35b_a3b/run_cpt_sft_qsa_35a3_h200x8.sh](../training/qsa_muon_35b_a3b/run_cpt_sft_qsa_35a3_h200x8.sh) |
| `qsa_muon_35b_a3b/scripts/env/build_from_source.sh` | [environment/scripts/build_from_source.sh](../environment/scripts/build_from_source.sh) |
| `qsa_muon_35b_a3b/scripts/env/install_official.sh` | [environment/scripts/install_official.sh](../environment/scripts/install_official.sh) |
| `qsa_muon_35b_a3b/scripts/env/verify_arch.sh` | [environment/scripts/verify_arch.sh](../environment/scripts/verify_arch.sh) |
| `qsa_muon_35b_a3b/serve_vllm_sglang_h200.sh` | [inference/serve_vllm_sglang_h200.sh](../inference/serve_vllm_sglang_h200.sh) |
| `recommended_pipeline.md` | [training/qwen35_27b/README.md](../training/qwen35_27b/README.md) |
| `recommended_pipeline_plugin.py` | [training/qwen35_27b/recommended_pipeline_plugin.py](../training/qwen35_27b/recommended_pipeline_plugin.py) |
| `sweep_configs.txt` | [training/qwen35_27b/legacy/sweep_configs.txt](../training/qwen35_27b/legacy/sweep_configs.txt) |
| `sweep_pipeline.sh` | [training/qwen35_27b/legacy/sweep_pipeline.sh](../training/qwen35_27b/legacy/sweep_pipeline.sh) |
| `test_recommended_pipeline.py` | [tests/test_recommended_pipeline.py](../tests/test_recommended_pipeline.py) |
| `train_s1_s3_recommended.py` | [training/qwen35_27b/train_s1_s3_recommended.py](../training/qwen35_27b/train_s1_s3_recommended.py) |
| `train_s1_s3_recommended.sh` | [training/qwen35_27b/train_s1_s3_recommended.sh](../training/qwen35_27b/train_s1_s3_recommended.sh) |
| `train_sft_qwen3_5_27b_megatron_lora_h200_pipeline_s1_s3.sh` | [training/qwen35_27b/legacy/train_sft_qwen3_5_27b_megatron_lora_h200_pipeline_s1_s3.sh](../training/qwen35_27b/legacy/train_sft_qwen3_5_27b_megatron_lora_h200_pipeline_s1_s3.sh) |
| `train_sft_qwen3_5_27b_megatron_lora_h200_vis_double.sh` | [training/qwen35_27b/legacy/train_sft_qwen3_5_27b_megatron_lora_h200_vis_double.sh](../training/qwen35_27b/legacy/train_sft_qwen3_5_27b_megatron_lora_h200_vis_double.sh) |
| `train_sft_qwen3_5_27b_megatron_lora_h200_vis_double_L.sh` | [training/qwen35_27b/legacy/train_sft_qwen3_5_27b_megatron_lora_h200_vis_double_L.sh](../training/qwen35_27b/legacy/train_sft_qwen3_5_27b_megatron_lora_h200_vis_double_L.sh) |
| `train_sft_qwen3_5_27b_megatron_lora_h200_vis_s3.sh` | [training/qwen35_27b/legacy/train_sft_qwen3_5_27b_megatron_lora_h200_vis_s3.sh](../training/qwen35_27b/legacy/train_sft_qwen3_5_27b_megatron_lora_h200_vis_s3.sh) |
| `train_sft_qwen3_5_27b_megatron_lora_h200_vis_tripple.sh` | [training/qwen35_27b/legacy/train_sft_qwen3_5_27b_megatron_lora_h200_vis_tripple.sh](../training/qwen35_27b/legacy/train_sft_qwen3_5_27b_megatron_lora_h200_vis_tripple.sh) |
| `进展日志.md` | [docs/history/progress_log.md](../docs/history/progress_log.md) |

## 保留规则

- `training/qwen35_27b/legacy/` 内原有脚本和配置，以及 `docs/history/progress_log.md` 的历史正文保持原始内容。历史文件名中的 `_L` 保留，以便对照记录。
- Finder 元数据与旧字节码缓存保留在本地，并由 `.gitignore` 排除；模型、数据和训练产物不纳入仓库。
- 本次整理不改训练配方、服务器路径、模型名称或 checkpoint 名称，也不代表完成了目标 GPU 上的验证。
- 原提交链保持不变；整理提交后，重命名文件通常可用 `git log --follow -- <新路径>` 追溯。对拆分或重写的文档，也可按上表的旧路径查询原提交内容。
