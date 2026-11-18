# 训练入口与目录分工

本仓库围绕 **AutoVision 8×B300 (`sm_103`) `Qwen3.5-27B`** 域适应主线，采用「**公共 B300 发射底盘 (`lib` + `cpt_27b`) + `lite / flash / pro` 三阶段流水线**」架构；历史 H200 与 35B-A3B QSA/Muon 实验已移入 [`archive/`](../archive/README.md)。

| 目录 | 定位与阶段 | 核心入口 | 说明 |
|---|---|---|---|
| [`lib/`](lib/hw_b300.sh) | **B300 公共层** | `_core.sh`、`hw_b300.sh` | 固化 `TP=4`、`MBS=4`、`GBS=32`、`16384`、APEX `--gradient_accumulation_fusion false`、TE `sm_103` 预检与 `--launch` 安全门禁 |
| [`cpt_27b/`](cpt_27b/run_cpt_27b_b300.sh) | **权威发射入口** | `run_cpt_27b_b300.sh`、`train_cpt_qwen3_5_27b_megatron.sh` | 统一承接 `lite` / `stage1` / `stage2` / `pro` 四档 Profile，缺省 dry-run，传 `--launch` 实发 |
| [`lite/`](lite/README.md) | **Phase A：Lite 增益对照** | `build_cpt_from_sch.py`、`run_lite_pipeline.sh` | `sch` 96 模板 iso 改造（以 `elements` 判通道）、`bundle_lite_v3_all·val0` 构建与双臂外部 OOD SFT 对照 |
| [`flash/`](flash/README.md) | **Phase B：Flash 全量线** | `build_mm_wiki_cpt.py`、`gen_cot_from_qa.py`、`refactor_v1_v4_pipeline.py`、`run_flash_pipeline.sh` | 全量两阶段 WSD CPT（含 `sch v2.3` 并入）、QA→CoT v0.3（`qwen3.8-flash` + 八道质检门 + 8:2 混配）与三方对照 SFT |
| [`pro/`](pro/README.md) | **Phase C：Pro 错例闭环** | `error_cluster_miner.py`、`montessori/`、`run_pro_pipeline.sh` | 四类错例归因（固定权重先验索引/路由）、50/30/20 增量 CPT、`sft_overlap=0` 隔离与 Montessori 影响函数对齐 |
| [`swift/`](swift/README.md) | **单模型独立入口** | `train_b300_swift.sh`、`train_qwen35_27b_swift.sh` | `Qwen3.8-Flash-Next` B300 单独 LoRA 尝试与单卡 27B SFT（供 Montessori/Datawheel 复用） |
| [`../archive/`](../archive/README.md) | **📦 历史 H200 / QSA 归档** | `qwen35_27b_h200/`、`qsa_muon_35b_a3b/` | 原 H200×8 `ms-swift 4.3.2` 三阶段纯 SFT 与 35B-A3B QSA/Muon 实验（保留回归测试） |

> 真实训练前仍须在目标 B300 集群核对底座路径、数据 MD5、启动日志 `train_iters`（偏离预期 >10% 立即停机登记）与保存恢复。
