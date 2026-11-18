# H200 与 QSA-Muon 历史归档目录 (`archive/`)

本目录统一归档已退出当前 **AutoVision 8×B300（`sm_103`）Qwen3.5-27B Lite/Flash/Pro 主线** 的早期 **H200** 与 **35B-A3B QSA/Muon** 资产，保留原始实现与回归测试可追溯性，避免干扰当前主线开发与发射。

## 归档清单与路径映射

| 原路径 | 归档路径 | 内容说明 |
|---|---|---|
| `training/qwen35_27b/` | [`training/qwen35_27b_h200/`](training/qwen35_27b_h200/README.md) | 早期 H200×8 `ms-swift 4.3.2` 三阶段 V1～V4 SFT 推荐脚本、`recommended_pipeline_plugin.py` 与 `legacy/` 历史 sweep |
| `training/qsa_muon_35b_a3b/` | [`training/qsa_muon_35b_a3b/`](training/qsa_muon_35b_a3b/run_cpt_sft_qsa_35a3_h200x8.sh) | Qwen3.5-35B-A3B H200×8 QSA 注意力替换与 Muon 优化器实验 |
| `inference/serve_vllm_sglang_h200.sh` | [`inference/serve_vllm_sglang_h200.sh`](inference/serve_vllm_sglang_h200.sh) | H200 35B-A3B QSA vLLM/SGLang 推理启动脚本 |
| `environment/requirements_h200.txt` | [`environment/requirements_h200.txt`](environment/requirements_h200.txt) | H200 QSA/Muon Python 依赖清单 |
| `docs/qsa_muon_35b_a3b.md` | [`docs/qsa_muon_35b_a3b.md`](docs/qsa_muon_35b_a3b.md) | 35B-A3B QSA 与 Muon 改造方案文档 |
| `docs/training_plans/domain_cpt_plan.md` | [`docs/domain_cpt_plan_h200.md`](docs/domain_cpt_plan_h200.md) | 2026-11-08 早期 H200×8 100M 纯文本 CPT 草案 |

## 回归兼容验证

归档的 H200 推荐流水线仍受 [`tests/test_recommended_pipeline.py`](../tests/test_recommended_pipeline.py) 覆盖：

```bash
python3 tests/test_recommended_pipeline.py -v
bash archive/training/qwen35_27b_h200/train_s1_s3_recommended.sh plan
```
