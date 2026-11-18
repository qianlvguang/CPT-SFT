# Phase A：Lite 增益对照方案

本目录聚焦 **AutoVision 8×B300（`sm_103`）`Qwen3.5-27B` 首轮域适应增益验证（Lite 线）**：

- [CPT-27B 战役 Runbook（Lite 增益对照线）](cpt_27b_runbook.md)：`bundle_lite_v3_all·val0`（12,085 行 / 9.55M tok）、`sch iso` 改造规则、B300 MBS=4 配方、TE `sm_103` 混搭排雷台账与双臂外部 OOD SFT 对照协议。
- 可执行代码入口：[`training/lite/build_cpt_from_sch.py`](../../../training/lite/build_cpt_from_sch.py)、[`training/lite/run_lite_pipeline.sh`](../../../training/lite/run_lite_pipeline.sh) 与 [`training/cpt_27b/run_cpt_27b_b300.sh`](../../../training/cpt_27b/run_cpt_27b_b300.sh)。
