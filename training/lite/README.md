# Phase A：Lite 按比例切片增益对照线 (`training/lite/`)

面向 **AutoVision 8×B300 (`sm_103`)** 的 `Qwen3.5-27B` 首轮领域适应快速验证实验。默认 **与 Flash 共用四桶 Mid-Training 源池，按 `doc_family_id` 严格整家族分层切片 15%（`val0`）**，并以 `MBS=1, GBS=4, MAX_LENGTH=8192`（`32.8K tok/step`）跑满 ~100~150 steps，彻底解决旧版 `95% sch + GBS=32*16384` 导致 **7 steps 刷完** 的问题（详案见 [`docs/training_plans/lite/cpt_27b_runbook.md`](../../docs/training_plans/lite/cpt_27b_runbook.md)）。

## 核心组件

| 文件 | 职责 |
|---|---|
| [`slice_flash_to_lite.py`](slice_flash_to_lite.py) | **默认主入口 (`LITE_MODE=proportional_flash`)**：从 Flash `midtrain_train.jsonl` 按 `--ratio 0.15` 整家族分层抽样（四桶配比不变、零跨族分裂、`val0`），统计 `256 tok/img` 真实 Token 并输出 B300 `MBS=1/GBS=4/8192` 步数预测与软链 `bundle_lite_compat/stage1_train.jsonl` |
| [`build_cpt_from_sch.py`](build_cpt_from_sch.py) | **历史回退 (`LITE_MODE=sch_legacy`)**：原 `sch iso` 96 模板改造（以 `elements` 为准判通道，`label` 降级审计） |
| [`run_lite_pipeline.sh`](run_lite_pipeline.sh) | Phase A 端到端编排（切片/建包 → `run_cpt_27b_b300.sh lite` → 导出 HF 权重 → 双臂外部 OOD SFT + `Think-ON`/`Think-OFF` 双模保留门对照） |
| [`examples/lite_sch_kb_sample.jsonl`](examples/lite_sch_kb_sample.jsonl) | 本地演示样例（含 `sch_legacy` 与 OOD SFT 对照样本） |

## 快速本地预演（无 GPU）

```bash
DEMO=1 DRY_RUN=1 bash training/lite/run_lite_pipeline.sh
# 兼容旧 sch_legacy 模式预演：
LITE_MODE=sch_legacy DEMO=1 DRY_RUN=1 bash training/lite/run_lite_pipeline.sh
```

## 集群直接发射命令

```bash
bash /data/autovision-cfs/autovision-notebook/CPT-SFT/training/cpt_27b/run_cpt_27b_b300.sh lite \
     --model /data/autovision-cfs/autovision-notebook/algo4/models/Qwen3.5-27B --launch
```
