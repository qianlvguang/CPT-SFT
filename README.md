# CPT-SFT：Qwen3.5-27B 域适应（Lite / Flash / Pro）与 QA→CoT v0.3

围绕 **AutoVision 8×B300（`sm_103`）`Qwen3.5-27B`** 电商侵权 / IP 认知域适应战役，采用「**公共 B300 发射底盘 (`training/lib` + `training/cpt_27b`) + `Lite / Flash / Pro` 三阶段流水线**」架构；历史 H200 与 35B-A3B QSA/Muon 实验已整体归档至 [`archive/`](archive/README.md)。

![CPT-SFT Lite / Flash / Pro 全链路工作流程图](docs/training_plans/assets/cpt_sft_lite_flash_pro_workflow.png)

> 整个 `CPT-SFT/` 是一个 Git 仓库，对应 [qianlvguang/CPT-SFT](https://github.com/qianlvguang/CPT-SFT)。AI 编码代理在此仓库工作前请先阅读 [AGENTS.md](AGENTS.md)。

## 目录架构

```text
CPT-SFT/
├── AGENTS.md                        # AI 编码代理工作约定与跨项目（Datawheel / OPD）互通说明
├── training/
│   ├── lib/                         # B300 公共硬件层（_core.sh, hw_b300.sh: TP4/MBS4/GBS32, TE sm_103 预检）
│   ├── cpt_27b/                     # 权威 B300 发射链（run_cpt_27b_b300.sh, train_cpt_qwen3_5_27b_megatron.sh）
│   ├── lite/                        # Phase A：Lite 增益对照线（sch iso 12,085 行 / 9.55M tok, val0）
│   ├── flash/                       # Phase B：Flash 全量 WSD 线 + QA→CoT v0.3 + 三方对照统一 SFT
│   ├── pro/                         # Phase C：Pro 错例簇闭环（50/30/20 增量 + sft_overlap=0 + Montessori）
│   └── swift/                       # 单模型独立 SFT 入口（Flash-Next B300 尝试与单卡 27B）
├── archive/                         # 📦 历史归档（H200 V1～V4 推荐线、H200 纯文本草案、35B-A3B QSA/Muon）
├── inference/                       # vLLM 0.32.0 B300 / RTX PRO 6000 服务脚本
├── environment/                     # CUDA/PyTorch 架构检查与编译说明
├── docs/
│   ├── training_plans/
│   │   ├── lite/                    # Phase A Runbook 与实测台账
│   │   ├── flash/                   # Phase B 全量 WSD、QA→CoT v0.3 详案（含 v2.1 全链路图）与风格方案
│   │   └── pro/                     # Phase C 错例闭环演进与 QAT/QLoRA 管理
│   ├── model_guide/                 # 模型结构、FP8/NVFP4、PLE 与权重检查
│   ├── research_snapshots/          # 官方来源 revision、配置字段与内存推导
│   ├── reports/                     # JEV 多模态训练与量化方案报告
│   └── history/                     # 历史实验记录
├── tools/                           # 权重头部读取、模块正则检查与工作流渲染器
└── tests/                           # 标准库单元测试与 Dry-Run 回归套件
```

## 按阶段选择入口

| 阶段 / 模块 | 核心代码与方案入口 | 验证边界 |
|---|---|---|
| **B300 权威发射链** | [`training/cpt_27b/run_cpt_27b_b300.sh`](training/cpt_27b/run_cpt_27b_b300.sh)（支持 `lite` / `stage1` / `stage2` / `pro`，默认 dry-run，传 `--launch` 实发） | 内置 `TP=4`、`MBS=4`、`GBS=32`、`--gradient_accumulation_fusion false`（APEX 修复）及 `TE sm_103` 预检；pod 卡 5 `TE` 前向已过 smoke，B300 全量实训按启动日志 `train_iters` 复核 |
| **Phase A：Lite 增益对照** | [`training/lite/README.md`](training/lite/README.md)、[`build_cpt_from_sch.py`](training/lite/build_cpt_from_sch.py)、[Lite 战役 Runbook](docs/training_plans/lite/cpt_27b_runbook.md) | `sch iso`、`elements` 通道判别（修正 `label=True` 但 `elements=无` 假阳）、`val0` 与双臂外部 OOD SFT 编排已通过本地测试与 `DRY_RUN=1` |
| **Phase B：Flash 全量 WSD + QA→CoT v0.3** | [`training/flash/README.md`](training/flash/README.md)、[`gen_cot_from_qa.py`](training/flash/gen_cot_from_qa.py)、[`build_mm_wiki_cpt.py`](training/flash/build_mm_wiki_cpt.py)、[QA→CoT v0.3 详案](docs/training_plans/flash/qa_cot_plan.md) | 八道质检门、同域干扰项混排重映射、冷酷核验风格三禁、`180/350/500` 硬下限、失败置空 `reasoning=""` 降级与 `8:2` 混配已通过本地 mock 测试与 `DRY_RUN=1` |
| **Phase C：Pro 错例闭环演进** | [`training/pro/README.md`](training/pro/README.md)、[`error_cluster_miner.py`](training/pro/error_cluster_miner.py)、[`montessori/`](training/pro/montessori/README.md)、[Pro 闭环方案](docs/training_plans/pro/pro_closed_loop_plan.md) | 四类错例归因（固定权重补索引/改路由门控）、`50/30/20` 增量构建、`sft_overlap=0` 隔离与 Montessori 探针已通过本地单元测试与 `DRY_RUN=1` |
| **单模型独立 SFT / 在线推理** | [`training/swift/`](training/swift/README.md)、[`inference/`](inference/README.md) | 独立 `Flash-Next` / 单卡 `27B` 入口及 vLLM 0.32.0 推理脚本 |
| **📦 历史归档（H200 / QSA）** | [`archive/README.md`](archive/README.md) | 原 H200 V1～V4 推荐线（`test_recommended_pipeline.py` 9/9 通过）与 35B-A3B QSA/Muon 实验 |

## 跨项目生态互通（Datawheel × OPD × CPT-SFT）

| 联动项目 | 相对路径 | 协同职责与数据流 |
|---|---|---|
| **Datawheel** | [`../Datawheel/`](../Datawheel/README.md) | 为 27B 中训练提供 `variant=6`（`loss_V6`）细粒度视觉定位与参考图匹配数据；通过 [`diagnose_and_refill_v6.py`](../Datawheel/tools/diagnose_and_refill_v6.py) 反哺多模态 DPO/KTO/SFT 与 Pro 阶段定向补采 |
| **OPD** | [`../OPD/`](../OPD/README.md) | 提供 ToDi 前/反向双通道 KL 蒸馏、闭源/异构 API 桥接自蒸馏（[`bsd_api_bridge.py`](../OPD/bsd_api_bridge.py)）、深度特化正交 LoRA 路由（[`depth_lora_router.py`](../OPD/depth_lora_router.py)）、多模态对齐诊断与 **[全域监控面板](../OPD/dashboard.html)** |

## 本地检查（无 GPU / 零下载）

```bash
python3 tests/test_lite_flash_pro_architecture.py -v
python3 tests/test_b300_mm_cpt_pipeline.py -v
python3 tests/test_unified_v1_v4_pipeline.py -v
python3 tests/test_montessori_pipeline.py -v
python3 tests/test_recommended_pipeline.py -v
DEMO=1 DRY_RUN=1 bash training/lite/run_lite_pipeline.sh
DEMO=1 DRY_RUN=1 bash training/flash/run_flash_pipeline.sh
DEMO=1 DRY_RUN=1 bash training/pro/run_pro_pipeline.sh
bash training/cpt_27b/run_cpt_27b_b300.sh lite --model /data/autovision-cfs/autovision-notebook/algo4/models/Qwen3.5-27B
DRY_RUN=1 bash inference/serve_qwen35_27b.sh
```
