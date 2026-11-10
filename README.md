# CPT-SFT：Qwen 训练、推理与模型研究

整理 Qwen3.5-27B 多阶段 LoRA SFT、35B-A3B 的 QSA/Muon 架构实验，以及 Qwen3.8-Flash-Next / Qwen3.5-27B 的训练与推理入口。

整个 `CPT-SFT/` 是一个 Git 仓库，对应 [qianlvguang/CPT-SFT](https://github.com/qianlvguang/CPT-SFT)。按用途分类，再在训练目录内区分模型和方案；模型权重、数据集、虚拟环境和运行产物单独存放。AI 编码代理在此仓库工作前请先阅读 [AGENTS.md](AGENTS.md)。

## 目录

```text
CPT-SFT/
├── AGENTS.md                # AI 编码代理工作约定与跨项目（Datawheel / OPD）互通说明
├── training/
│   ├── qwen35_27b/          # H200×8，V1～V4 累计 LoRA 推荐路线
│   │   └── legacy/          # 历史训练与 sweep 脚本，保留原始内容
│   ├── qsa_muon_35b_a3b/    # H200×8，QSA 预热、CPT、SFT 与插件
│   ├── swift/               # B300 Flash-Next 与单卡 27B 微调入口
│   └── domain_adaptation/   # 法规/文章/Wiki CPT、QA 合成、引用风格方案
├── inference/               # vLLM / SGLang 服务脚本
├── environment/             # H200 依赖、CUDA/PyTorch 说明及环境脚本
├── docs/
│   ├── model_guide/         # 模型结构、FP8/NVFP4、PLE 与权重检查
│   ├── research_snapshots/  # 官方来源 revision、配置字段与内存推导
│   ├── reports/             # JEV 多模态训练与量化方案报告
│   ├── training_plans/      # 领域 CPT、推理制 QA、严谨引用风格 SFT
│   └── history/             # 历史实验记录
├── tools/                   # 权重头部读取、模块正则检查
└── tests/                   # 调度回归测试、在线服务冒烟客户端
```

## 按任务选择入口

| 任务 | 入口与说明 | 验证边界 |
|---|---|---|
| 27B 多阶段 SFT | [推荐路线](training/qwen35_27b/README.md)、[启动脚本](training/qwen35_27b/train_s1_s3_recommended.sh) | 调度器有标准库回归测试；推荐数据组合尚未完成 H200 实训验证 |
| 35B-A3B QSA + Muon | [方案](docs/qsa_muon_35b_a3b.md)、[训练入口](training/qsa_muon_35b_a3b/run_cpt_sft_qsa_35a3_h200x8.sh) | 架构、算子与优化器需在目标环境验证 |
| B300 / 单卡 27B 微调 | [MS-SWIFT 说明](training/swift/README.md) | 不同模型使用各自入口；目标硬件尚未验证 |
| 领域知识与回答风格 | [CPT → QA → 风格总方案](docs/training_plans/README.md)、[配置与示例](training/domain_adaptation/README.md)、[Montessori-Instruct 管线](training/domain_adaptation/montessori/README.md) | 法规/文章/Wiki 原文适配，含单步 Loss 差分探针、黑盒 API 对比引导（ICDPO）与 DPO/KTO/GRPO 多格式偏好导出（已通过本地单元测试与 dry-run） |
| 在线推理 | [服务说明](inference/README.md)、[H200 QSA 服务](inference/serve_vllm_sglang_h200.sh) | B300/RTX 脚本固定 vLLM 0.32.0；实际权重与硬件需分别验证 |
| 环境安装与检查 | [环境说明](environment/README.md) | 安装脚本按需运行，不随本地检查自动安装依赖 |
| 模型与量化研究 | [模型指南](docs/model_guide/README.md)、[JEV 报告](docs/reports/qwen3_5_jev_multimodal_training_and_quantization_report.md) | 研究方案和报告中的估算不能视作实测结果 |
| 历史复盘 | [训练进展日志](docs/history/progress_log.md)、[历史脚本](training/qwen35_27b/legacy/) | 日志中的远端路径、参数和评测保留历史含义 |

这些方案对应不同模型、训练框架和并行配置。27B 推荐路线仍是 V1～V4 的一条固定配方；`legacy/` 中的 sweep 仅用于历史追溯。

27B 的新增调研入口：[模型结构详解](docs/model_guide/qwen35_27b_architecture.md)、[量化方案与部署顺序](docs/model_guide/qwen35_27b_quantization.md)。其中明确区分官方 FP8 配置、NVFP4A16 工具示例、W4A4 的额外条件，以及自训模型尚待完成的验证。

领域知识新增路线：[法规/文章/Wiki 原文 CPT](docs/training_plans/domain_cpt_plan.md) → [有证据的 QA 合成与 SFT](docs/training_plans/qa_synthesis_plan.md) → [严谨详尽、带依据引用的风格收尾](docs/training_plans/answer_style_sft_plan.md)。推理端同时保留可更新的检索来源，避免把引用外观当成引用正确。

训练与管理补充：[27B QAT/QLoRA、WSD 收尾、新 Megatron 支持及文件管理](docs/training_plans/qwen35_27b_advanced_training_and_management.md)。含原生配置与薄启动器设计示例，已核对官方 release 元数据，尚未实现新启动器或在目标 GPU 验证。

## 跨项目生态互通（Datawheel × OPD × CPT-SFT）

| 联动项目 | 相对路径 | 协同职责与数据流 |
|---|---|---|
| **Datawheel** | [`../Datawheel/`](../Datawheel/README.md) | 为 27B 中训练提供 `variant=6`（`loss_V6`）细粒度视觉定位与参考图匹配数据；通过 [`diagnose_and_refill_v6.py`](../Datawheel/tools/diagnose_and_refill_v6.py) 将 VLM 知识缺口与 BBox 失准样本转化为本仓库可直接训练的多模态 DPO/KTO/SFT 数据集 |
| **OPD** | [`../OPD/`](../OPD/README.md) | 提供 ToDi 前/反向双通道 KL 蒸馏、闭源 API 桥接自蒸馏（[`bsd_api_bridge.py`](../OPD/bsd_api_bridge.py)）、深度特化正交 LoRA 路由（[`depth_lora_router.py`](../OPD/depth_lora_router.py)）、多模态图文对齐诊断引擎（[`multimodal_datawheel_bridge.py`](../OPD/multimodal_datawheel_bridge.py)）及 **[全域监控面板 HTML](../OPD/dashboard.html)** |

## 本地检查

以下命令从仓库根目录执行，不启动 GPU 训练或下载模型：

```bash
python3 tests/test_recommended_pipeline.py
python3 tests/test_montessori_pipeline.py
bash training/qwen35_27b/train_s1_s3_recommended.sh plan
DRY_RUN=1 bash training/domain_adaptation/montessori/run_montessori_pipeline.sh
DATASET=training/swift/example_sft.jsonl DRY_RUN=1 bash training/swift/train_qwen35_27b_swift.sh
DATASET=training/swift/example_sft.jsonl DRY_RUN=1 bash training/swift/train_b300_swift.sh
DRY_RUN=1 bash inference/serve_qwen35_27b.sh
```

推理启动脚本的 dry-run 仍会创建缓存和模型目录，可通过 `HF_CACHE`、`MODEL_DIR` 指定位置。连接真实服务的客户端位于 [tests/smoke_test.py](tests/smoke_test.py)，需等目标服务启动后另行运行。

## 目录迁移

2026-11-08 将原嵌套仓库提升到整个 `CPT-SFT/`，保留原有 Git 提交历史，并将父目录的 27B 脚本纳入同一工作树。新旧路径对照见 [目录迁移说明](docs/repository_layout.md)。
