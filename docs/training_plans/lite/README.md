# Phase A：Lite 按比例切片增益对照方案

本目录聚焦 **AutoVision 8×B300（`sm_103`）`Qwen3.5-27B` 首轮域适应增益验证（Lite 线）**：

- [CPT-27B 战役 Runbook（Lite 增益对照线）](cpt_27b_runbook.md)：涵盖 **7-step 崩溃根因复盘**（`sch` 短行 + `GBS=32*16384`）、校准后 `slice_flash_to_lite.py`（从 Flash 四桶整家族切片 `15%`、`val0`）、`MBS=1 / GBS=4 / 8192`（~100–150 steps）、TE `sm_103` 排雷与双臂 OOD SFT + `Think-ON/OFF` 双模保留门。
- 可执行代码入口：[`training/lite/slice_flash_to_lite.py`](../../../training/lite/slice_flash_to_lite.py)、[`training/lite/build_cpt_from_sch.py`](../../../training/lite/build_cpt_from_sch.py)（`sch_legacy` 回退）、[`training/lite/run_lite_pipeline.sh`](../../../training/lite/run_lite_pipeline.sh) 与 [`training/cpt_27b/run_cpt_27b_b300.sh`](../../../training/cpt_27b/run_cpt_27b_b300.sh)。
