# Phase A：Lite 增益对照线 (`training/lite/`)

面向 **AutoVision 8×B300 (`sm_103`)** 的 `Qwen3.5-27B` 首轮领域适应增益实验，对应详案见 [`docs/training_plans/lite/cpt_27b_runbook.md`](../../docs/training_plans/lite/cpt_27b_runbook.md)。

## 核心组件

| 文件 | 职责 |
|---|---|
| [`build_cpt_from_sch.py`](build_cpt_from_sch.py) | `sch iso` 改造与 `bundle_lite_v3_all·val0` 构建器（问句剥除、`<image>` 置顶、原子 `desc + elements` 自拼、以 `elements` 为准判别 `ch_ip_negative`/`ch_ip_grounding`/`ch5_flash_rag_raft` 并将 `label` 降级为审计字段、输出 MD5/SHA256 台账与 `bundle_lite_compat` 软链） |
| [`run_lite_pipeline.sh`](run_lite_pipeline.sh) | Phase A 端到端编排（构建 Lite 包 → 调用 `cpt_27b/run_cpt_27b_b300.sh lite` → 导出 HF 权重并对齐 Arm A `CPT+OOD SFT` vs Arm B `Base+OOD SFT`） |
| [`examples/lite_sch_kb_sample.jsonl`](examples/lite_sch_kb_sample.jsonl) | 含 `label=True` 但 `elements=无` 负例纠偏、RAG 升格与 KB 补充条目的本地演示样例 |

## 快速本地预演（无 GPU）

```bash
DEMO=1 DRY_RUN=1 bash training/lite/run_lite_pipeline.sh
```

## 集群直接发射命令

```bash
bash /data/autovision-cfs/autovision-notebook/CPT-SFT/training/cpt_27b/run_cpt_27b_b300.sh lite \
     --model /data/autovision-cfs/autovision-notebook/algo4/models/Qwen3.5-27B --launch
```
