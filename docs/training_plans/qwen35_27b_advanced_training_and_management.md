# Qwen3.5-27B：QAT、QLoRA、末期降 LR、Megatron 与文件管理

调研日期：2026-11-08。本文给出本项目的实施建议、原生配置示例和管理约定；没有安装或升级环境，没有启动 GPU 训练、量化或模型下载。文中的新增配置文件和启动器是设计示例，尚未生成可执行实现；现有训练脚本与历史配方保持原样。

## 1. 推荐如何选

| 目标 | 本项目建议 | 使用位置 |
|---|---|---|
| H200×8 上做领域知识适配 | BF16 语言主干 CPT；新增一个 WSD 收尾候选 | 法规/文章/Wiki 原文阶段 |
| 单卡显存受限时训练问答与风格 | Transformers + bitsandbytes NF4 QLoRA | CPT 之后的 QA / 风格 SFT |
| 最终低精度部署损失不可接受 | 先确认 PTQ 回归，再针对相同部署格式做 QAT；需要教师信号时做 QAD | 内容与风格 SFT 完成之后 |
| 简化脚本配置 | 框架原生配置文件 + 薄启动入口 + 每次运行的解析后快照 | 所有新增工作线 |

这三种训练技术可以出现在不同阶段，但不应同时改变基座精度、优化器、数据配比和调度器后再归因。现有 [领域 CPT](domain_cpt_plan.md) 的 cosine 仍是对照；新增 WSD 是一条单独候选，不覆盖旧实验。

## 2. QLoRA：节省适配训练显存

经典 QLoRA 冻结低比特基座，通过基座反向传播来训练 LoRA；NF4、double quantization 与分页优化器是原论文的主要机制。它不是“把所有参数以 4-bit 做全参更新”，也不自动得到目标 NVFP4 W4A4 部署模型。[QLoRA 原论文](https://arxiv.org/abs/2305.14314)

### 2.1 27B 的一个明确起点

以已选定 CPT 产物为基座，使用 Transformers / PEFT / bitsandbytes 路线：

| 项目 | 建议值或行为 |
|---|---|
| 量化与计算 | NF4、double quantization、BF16 计算；记录实际转换的模块 |
| 更新范围 | 语言主干 LoRA r32 / alpha64 / dropout0.05；视觉塔与 merger 冻结 |
| QA SFT | LR `2e-5`，1 epoch，先以 2048 长度、MBS1、梯度累积16验证 |
| 风格收尾 | 延续同一 adapter，LR `5e-6`，按已有 [风格方案](answer_style_sft_plan.md) 回放已验真 QA |
| 节省激活 | 开启 gradient checkpointing；先关闭 packing，待混合注意力边界验证后再单独开启 |

长度 2048 只是短跑起点；正式带证据 QA 需要按输入分布增加长度或重新切分来源，不能截断引用、关键条件或答案以凑显存。96 GB 卡是本项目讨论的目标之一，这里没有宣称具体卡型能够容纳任意多图/长文配置。

HF/PEFT 的通用低比特训练准备及 NF4 参数已有文档支持；训练只更新额外参数，`device_map="auto"` 的推理自动分配不能直接作为分布式训练策略。[HF bitsandbytes](https://huggingface.co/docs/transformers/main/quantization/bitsandbytes)、[PEFT 量化训练](https://huggingface.co/docs/peft/main/en/developer_guides/quantization)

### 2.2 27B 需要专门核对的地方

- 27B 有 48 层 GDN、16 层全注意力。只挂 `q_proj/v_proj` 不代表覆盖 GDN；量化转换清单与 LoRA 注入清单要分别检查。
- `freeze_vit/freeze_aligner` 控制训练范围，不保证加载时没有量化这些模块。当前 SWIFT 的 BNB 加载器会根据模型架构建立视觉/aligner/LM head 跳过清单；实际名称、GDN 小门控及自定义 JEV head 仍需检查，不能把官方 FP8 排除名单当成已自动应用的 NF4 配置。
- 如果保护范围无法用现有加载器表达，增加一个明确的小型 loader/plugin 后再训练；不要在 JSON 中填不存在的量化排除参数。
- 梯度应进入预期 LoRA；递推状态精度、融合 GDN 算子、adapter 保存/恢复和多模态前向均需短跑验证。Qwen3 或小尺寸模型的成功案例不能代替这份 27B 的验证。

依据：[SWIFT 量化加载源码](https://github.com/modelscope/ms-swift/blob/main/swift/arguments/base_args/quant_args.py)、[27B 结构详解](../model_guide/qwen35_27b_architecture.md)。

### 2.3 产物如何进入部署

保留“精确基座 + 量化加载配置 + adapter”组合。若要把 adapter 合并到原始 BF16 基座，再量化为 FP8/NVFP4，应把它视作新的导出候选，重新检查输出和概率；原始 BF16 权重与训练时 NF4 的有效基座有量化误差，不能默认两条路径完全等价。对 NF4 直接反量化也不会恢复已经丢失的原始精度。

ModelOpt 另有真实 NVFP4 压缩基座的 QLoRA 示例，这是另一套实现和导出格式。其当前示例明确提示 **QLoRA export 尚不支持 FSDP2**，不能把通用 QAT 的 FSDP2 支持直接套过来；本项目首轮 QLoRA 仍采用 NF4 路线。[ModelOpt QLoRA 示例](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_qat#qlora-real-quantization)

## 3. QAT / QAD：针对部署精度恢复质量

QAT 在训练前向中模拟目标量化误差，通常仍保留高精度可训练参数与优化器状态；它不天然具有 QLoRA 的低显存特征。QAD 再加入高精度教师信号，适合研究激进量化后的恢复，但教师、logits 和额外前向会增加显存与算力开销。[ModelOpt QAT/QAD 指南](https://nvidia.github.io/Model-Optimizer/guides/quantization_aware_training_and_distillation.html)

### 3.1 本项目的一条候选方案

1. 从**已经完成 QA 与风格验收**的 BF16 checkpoint 出发，固定测试集、模板和 head 校准参数；先报告同格式 PTQ 的真实回归。
2. 首个 NVFP4 QAT 候选只量化语言 FFN 投影，GDN、全注意力、视觉栈、embedding、LM head 和自定义决策头保留原有高精度；KV cache 也固定。这个选择是降低适配复杂度的项目建议，不是已验证最优保护范围。
3. 使用 ModelOpt 的 HF QAT 路线，先做单个 batch 的前向、反向、保存和导出；多卡时按实际 `Qwen3_5DecoderLayer` 类型核对 FSDP 包装，不能沿用示例的 `Qwen3DecoderLayer` 名称。
4. 首轮取已验真 QA 训练集的固定 10% 子集，最多一遍；全参 QAT 学习率先定为 `5e-6`。这是恢复试验起点，不能保证充分收敛；同时观察引用、风格与 JEV 概率。
5. 导出后在固定的 vLLM 0.32.0 环境做真实加载与评测。fake-quant 模型表现正常，不等于 packed checkpoint、scale 布局和目标 GPU kernel 均正确。

ModelOpt 当前主分支的 HF 示例列出 Qwen 3.5 dense、混合精度和 assistant-only 标签支持，但这**不是本项目 27B、特定 GDN kernel、自定义 head、多模态、MTP 与最终导出组合的实测证明**。该示例的默认部分配方还启用 FP8 KV；本项目若要只研究权重与激活，必须在配方中明确保留 KV 精度。[HF QAT 示例与支持范围](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_qat)

### 3.2 何时加 QAD

若 QAT 后依然存在显著量化回归，再研究同一 BF16 产物作为教师的 QAD。区分 token-level CE、教师 KL 与自定义决策分布蒸馏；仅对语言 logits 蒸馏不能保证额外决策头得到正确约束。248320 词表下，长序列完整 logits 很大，应核对分块/显存策略和所选蒸馏损失的实际实现。

恢复评测至少包括 held-out loss、事实/条件、引用支持、风格、多模态，以及 JEV 的 NLL/Brier 和阈值翻转。所有结论以导出后实际部署模型为准。训练 GPU 能模拟某种格式，与部署 GPU 有原生 FP4 kernel，是两项独立检查。[原有量化与硬件方案](../model_guide/qwen35_27b_quantization.md)

## 4. 中训练后期快速降 LR：WSD 收尾

用户提出的末期快速下降，可以具体化为 **Warmup → Stable → Decay**：先适应新数据分布，在稳定 LR 下继续学习，然后用较短的退火阶段降低优化噪声。相关研究支持这种设计思路，但并未证明以下 27B 配方优于现有 cosine。[WSD 研究](https://arxiv.org/abs/2410.05192)、[Cooldown 动力学研究](https://arxiv.org/abs/2508.01483)

### 4.1 一条固定的首轮候选

以原生 BF16 语言 CPT 为对象：warmup 占 3%，stable 占 87%，最后 10% **线性**降到峰值的 10%。峰值 `1e-5`，末值 `1e-6`，AdamW 的其余参数和数据混合保持对照一致；不同时修改 beta2 或在末期换语料。

令 `p` 为完成训练预算的比例，学习率定义为：

```text
0 ≤ p < 0.03：lr = 1e-5 × p / 0.03
0.03 ≤ p ≤ 0.90：lr = 1e-5
0.90 < p ≤ 1：lr = 1e-6 + (1e-5 - 1e-6) × (1-p) / 0.10
```

如果 1000 次更新只是调度算例，则 warmup 30 步、stable 870 步、decay 100 步；第 900 步仍是 `1e-5`，第 950 步为 `5.5e-6`，第 1000 步为 `1e-6`。它约对应 262M 满载 token，不是此前 100M pilot 的同一预算。

此前 100M pilot 在 GBS32、8192 满载假设下约需 382 步：可取 warmup 12、decay 39、stable 331。数据不足、不满 packing、变长 batch 或改变全局批量时必须重新计算，并记录真实非 padding token；不能只看 epoch 百分比。

### 4.2 框架参数与恢复

Megatron-SWIFT 文档已有 `lr_decay_style=WSD`、`lr_wsd_decay_iters`、`lr_wsd_decay_style`。其默认 decay shape 是 exponential，本方案要显式写 **linear**；`lr_decay_iters` 指整个调度总长度，而非只指末尾 10%。[SWIFT 调度参数](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Command-line-parameters.html)、[Megatron 调度器实现](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer_param_scheduler.py)

HF 使用自己的 `get_wsd_schedule` / Trainer scheduler 接口，参数名不同；不能把 Megatron 字段直接传给 Transformers。[HF WSD 接口](https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.get_wsd_schedule)

保留退火前完整 checkpoint。连续跑进收尾阶段时保留优化器、RNG、数据位置和累计步数；不要用“重新加载权重 + 重置 scheduler”冒充原调度的继续。SWIFT 的 HF `model/adapters` 权重加载与 MCore 完整恢复语义不同；恢复需核对 `mcore_model/mcore_adapter`、`finetune=false` 和优化器/RNG 加载，调度覆盖还需检查具体版本。[SWIFT 保存与恢复](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Command-line-parameters.html)

已经跑过 cosine 的旧任务，也不能仅加载末档、突然改低 LR 后声称复现 WSD。若另开退火分支，明确记录父档、追加 token 和新的调度定义。比较时对齐总 token、数据、父模型与下游评测，不能只比较训练 loss 最低的点。

## 5. Megatron 新版本支持如何判断

### 5.1 名称和职责

| 组件 | 负责什么 | 本项目的含义 |
|---|---|---|
| Megatron Core / Megatron-LM | 并行层、训练基础设施、优化器与分布式状态 | 底层能力不等于任意 HF 架构已自动适配 |
| NVIDIA Megatron Bridge | 模型 provider、HF↔MCore 映射及 NVIDIA 训练 recipes | 使用其绑定的 Core、TE 与容器依赖 |
| ModelScope `mcore-bridge` | ModelScope 的模型定义与权重映射 | 与 NVIDIA 的 `megatron-bridge` 是两个项目 |
| Megatron-SWIFT | `megatron pt/sft/rlhf`、数据模板与训练集成 | 现有脚本主要使用这条入口和 `mcore-bridge` |

### 5.2 截至调研日核实的 release

以下通过官方 GitHub release API 核对，并按 Asia/Shanghai 的 2026-11-08 截止；是各仓库发布情况，**不是一套已经验证兼容的安装组合**。原始结果见 [版本快照](../research_snapshots/training_stack_versions_2026_11_08.json)。

| 仓库 | 查询到的最新稳定 release | 发布时间（北京时间） |
|---|---|---|
| Megatron Core | [core_v0.20.0](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.20.0) | 2026-10-30 |
| NVIDIA Megatron Bridge | [v0.7.0](https://github.com/NVIDIA-NeMo/Megatron-Bridge/releases/tag/v0.7.0) | 2026-10-30 |
| MS-SWIFT | [v4.6.0](https://github.com/modelscope/ms-swift/releases/tag/v4.6.0) | 2026-10-13 |
| ModelScope mcore-bridge | [v1.6.5](https://github.com/modelscope/mcore-bridge/releases/tag/v1.6.5) | 2026-10-13 |
| NVIDIA Model Optimizer | [0.48.0](https://github.com/NVIDIA/Model-Optimizer/releases/tag/0.48.0) | 2026-11-02 |

版本选择采用官方 `releases/latest`，并检查发布时间；release 列表第一条可能是旧分支补丁，例如较晚发布的 ModelOpt 0.47.1，不能据此覆盖官方 latest 的 0.48.0。前文部分能力来自 SWIFT `4.7.0.dev` / ModelOpt main 文档，未必已包含在上述 stable tag；实施时按所需功能固定 commit。网页的 `latest/nightly`、源码 changelog 的未来版本号和实际发布 tag 不混为一谈。

### 5.3 模型与任务支持边界

| 对象 | 当前找到的证据 | 实施时仍需验证 |
|---|---|---|
| Qwen3.5-27B Dense | NVIDIA Bridge 模型文档列出 dense 桥接与 27B；ModelScope 桥接也覆盖 Qwen3.5 | 27B 的具体文本/多模态、全参/LoRA、packing、恢复与导出组合；不能把 9B recipe 改模型名就视作验证 |
| Qwen3.8-Flash-Next | SWIFT 专项文档提供 Megatron SFT LoRA 及依赖要求 | 原生 QSA、Hyper-Connections、PLE 与对应权重映射；不能用普通 Qwen3.5 provider 代替 |
| 本项目 35B-A3B QSA 改造 | 本仓库的自定义插件与方案 | 官方原生模型支持不涵盖此架构手术；旧 patch 注入点、state dict、算子和导出全部重验 |
| BNB NF4 QLoRA | SWIFT/HF 的低比特加载与 PEFT 路线 | 不将 Transformers QLoRA 参数直接当成 Megatron Core 的原生 QLoRA 支持 |
| QAT / QAD | ModelOpt 有 HF 与 Megatron 系入口 | 按具体模型、量化格式和导出链核对；不代表 Megatron-SWIFT 多加一个 flag 就支持 |

依据：[Bridge Qwen3.5 文档](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/qwen/qwen35-vl.html)、[ModelScope 桥接固定 tag](https://github.com/modelscope/mcore-bridge/tree/v1.6.5)、[SWIFT Flash-Next 专项文档](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html)。

NVIDIA Bridge v0.7.0 的 `pyproject.toml` 要求 Python `>=3.12,<3.13`，Core 使用项目内子模块，TE 也有绑定来源；这说明“把现有环境所有包各自升级到最新”不是有证据的兼容方案。[固定 tag 的依赖定义](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/v0.7.0/pyproject.toml)

Core 0.20 的 Megatron-FSDP v2 仍标为实验路径：其适配器要求 TP1 和 expert TP1，优化器 checkpoint 还存在 EP 布局限制。它不能直接套到本项目原来的 TP4 配方。[Core 0.20 发布说明](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.20.0)

Flash-Next 的专项文档要求 Transformers ≥5.16、较新的 mcore-bridge、FLA ≥0.5.2 等，且 `PLE_CPU_OFFLOAD=1` 会冻结 PLE 表。训练框架、新模型依赖、训练精度和 vLLM 推理镜像分别锁定；当前 vLLM 0.32.0 不因此自动升级。

### 5.4 一次升级的验收顺序

新建独立候选环境，先做依赖/ABI 检查和模型导入，再做 HF↔MCore 转换一致性；随后测试前向/反向、可训练参数、10～20 步短跑、完整状态恢复、导出与服务。纯文本通过后，才扩展到多模态、packing 或新并行方式。记录最大 logits 差异和业务差异，容差按数值精度及任务预先确定。

原插件对 trainer 内部方法的 patch 必须检查新版本实际调用路径；Python 能编译、旧单元测试通过，不能证明插件仍作用于新训练循环。具体依赖锁应来自通过上述验收的环境或项目推荐容器，而不是仅靠 `>=` 下限。

## 6. 科学的文件与产物管理

### 6.1 代码仓库与大文件各有一个根目录

保留当前 `training/inference/environment/docs/tools/tests` 分类。新增配置集中到 `training/domain_adaptation/configs/`，数据的实际内容与运行产物存到用户指定的 `CPT_SFT_STORAGE`。以下是目标结构设计，不执行现有模型或数据的搬迁：

```text
CPT-SFT/                              # Git：代码、小配置、文档与来源登记
├── training/domain_adaptation/
│   ├── configs/                       # 每条工作线的原生配置
│   ├── prompts/                       # QA / 验真 / 风格模板
│   ├── examples/                      # 小型格式示例
│   └── launch.sh                      # 需要时再实现的薄启动器
└── environment/locks/                 # 验收后形成的版本/镜像锁

$CPT_SFT_STORAGE/                      # Git 外的独立存储
├── models/<model_id>/<revision>/      # 原始基座只读复用
├── datasets/<dataset_id>/<version>/
│   ├── raw/                          # 原文快照只读
│   ├── clean/                        # 清洗后文档
│   ├── shards/                       # CPT / QA / SFT 训练分片
│   └── manifest.json                 # 文件 hash、来源、切分、token 分布
├── runs/<run_id>/
│   ├── config_resolved.json
│   ├── lineage.json
│   ├── environment.json
│   ├── logs/
│   ├── metrics/
│   ├── checkpoints/                  # 完整恢复状态
│   └── status.json
└── exports/<export_id>/               # 独立的部署产物及验收记录
```

基座、数据版本和训练产物使用不可变 ID；`run_id` 可采用“带时区时间戳 + 阶段 + 配置指纹前8位”。物理路径由运行环境绑定，数据和父模型的身份由内容指纹确定；换挂载点不应冒充换了数据。

### 6.2 每次运行至少固定这些内容

| 对象 | 必须保存的信息 |
|---|---|
| 配置 | 合并 CLI 覆盖后的实际参数、配置 SHA256、数据/模型绑定、optimizer/scheduler 全部语义 |
| 代码 | Git commit；有未提交修改时保存补丁及其 hash；自定义插件、提示词、清洗与 evaluator 版本 |
| 数据 | 原文/分片 hash、去重簇、文档族 split、来源版本、有效 token、采样配比、tokenizer 与模板 hash |
| 模型谱系 | 原始基座 revision、父 run/checkpoint、adapter/head/quantizer 状态及其指纹 |
| 环境 | GPU、驱动、CUDA、PyTorch、Core、桥接、TE、FLA、conv、BNB/ModelOpt 的版本或 commit、镜像 digest |
| 评测 | 固定测试集 ID、推理参数、scorer、逐样本输出和指标分母；引用与风格指标独立保存 |

`config_resolved.json` 供审计，计算语义指纹时排除时间戳和输出目录等易变字段，纳入真实内容身份与代码/软件版本。完整权重 hash 首次登记时顺序计算一次，后续复用登记结果并对变动文件重新核验；不能只凭文件名或 mtime 确认内容未变。不要把访问 token 和完整进程环境快照写入这些文件。

### 6.3 完成标记、断点与清理

- 运行状态使用 `prepared/running/failed/trained/evaluated/exported` 等明确字段。只有全部 rank 写完、权重与索引完整、关键产物校验通过后，才在同一文件系统内原子发布完成记录；`config.json` 存在不是完成依据。
- checkpoint 分为“可恢复训练状态”和“部署权重”。连续 CPT/WSD 需要 optimizer、scheduler、RNG、sampler/数据位置；QAT 还要保存 quantizer/scale 状态。只有 adapter 的目录不具备相同恢复语义。
- 优先保留退火前档、当前可恢复档、按固定验证口径选中的候选及其父档。配置保留规则时去重引用，避免重复复制大权重；先核算容量，再决定保存间隔。
- 删除分两步：生成待清理清单并检查依赖，再由用户确认大目录清理。被子任务或部署引用的父档保持可访问；正式模型、数据版本和历史输出不为整理改名。

## 7. 配置与脚本怎样最简洁

建议采用**每个后端一份原生配置，启动入口只负责选阶段和传入资源路径**。稳定超参写文件，GPU 选择和路径在启动时显式给出，最终实际参数保存到 run 目录。原有 `recipe_example.json` 是跨阶段设计说明，不直接传给任一训练器；不要把它误认为已实现的统一 CLI。

SWIFT 原生支持 `swift sft xxx.json` / YAML 及 CLI 覆盖，Megatron-SWIFT 同样有原生配置入口。ModelOpt 的训练脚本使用自己的 `--config`，Accelerate 的分布式配置则是另一份文件。[SWIFT 配置格式](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html#yaml-json-support)、[ModelOpt 配置接口](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_qat)

### 7.1 CPT/WSD 原生配置示例

这是 **1000 步调度算例**，需在数据与显存预检后使用；前述 100M pilot 应改为 382/12/39。未设置的模型、数据、验证集和输出目录在启动时传入。先按实际模板预编码并确认超长样本为零；框架的合法 `delete` 策略必须配合过滤数量审计，发现非零删除就停止并修复数据，不能静默丢行。

```json
{
  "tuner_type": "full",
  "torch_dtype": "bfloat16",
  "freeze_vit": true,
  "freeze_aligner": true,
  "max_length": 8192,
  "truncation_strategy": "delete",
  "strict": true,
  "tensor_model_parallel_size": 4,
  "pipeline_model_parallel_size": 1,
  "context_parallel_size": 1,
  "micro_batch_size": 1,
  "global_batch_size": 32,
  "train_iters": 1000,
  "optimizer": "adam",
  "lr": 1e-5,
  "min_lr": 1e-6,
  "lr_decay_style": "WSD",
  "lr_decay_iters": 1000,
  "lr_warmup_iters": 30,
  "lr_wsd_decay_iters": 100,
  "lr_wsd_decay_style": "linear",
  "no_save_optim": false,
  "no_save_rng": false
}
```

`optimizer=adam` 是该 Megatron 接口名称，权重衰减和其他 Adam 参数仍要写入解析后快照。重计算、packing、保存与验证频率按通过预检的机器配置补齐，不把上述核心片段当成完整生产配方。

### 7.2 NF4 QLoRA 原生配置示例

```json
{
  "tuner_type": "lora",
  "torch_dtype": "bfloat16",
  "quant_method": "bnb",
  "quant_bits": 4,
  "bnb_4bit_quant_type": "nf4",
  "bnb_4bit_use_double_quant": true,
  "bnb_4bit_compute_dtype": "bfloat16",
  "lora_rank": 32,
  "lora_alpha": 64,
  "lora_dropout": 0.05,
  "target_modules": ["all-linear"],
  "freeze_vit": true,
  "freeze_aligner": true,
  "learning_rate": 2e-5,
  "num_train_epochs": 1,
  "max_length": 2048,
  "truncation_strategy": "delete",
  "strict": true,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 16,
  "gradient_checkpointing": true,
  "packing": false
}
```

这份片段不表达自定义量化排除规则。按 §2.2 核对实际保护与注入范围后才能使用；`all-linear` 在冻结视觉的目标后端应只留下预期语言 LoRA。`strict=true` 不是“禁止超长删除”的替代品，仍需上文的预编码和过滤数量审计。当前 SWIFT 文档没有列出 `truncation_strategy=raise`，因此不把它当作可直接使用的原生值。单卡有效 batch 为16，多卡时还需乘 DP 并重新核对数据预算。CPT 使用 `lr`，此 HF SFT 使用 `learning_rate`，两者明确保留各自接口。

### 7.3 启动入口的职责

下例是将来需要重复启动时可实现的薄入口，仅分发两条已选后端；配置文件需按上文另行落盘。它没有实现 manifest 解析、预检、运行快照或恢复，因此不是当前可直接运行的交付脚本。

```bash
#!/usr/bin/env bash
set -euo pipefail
stage="${1:?需要 cpt 或 qlora}"
shift
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
case "$stage" in
  cpt) exec megatron pt training/domain_adaptation/configs/cpt_wsd.json "$@" ;;
  qlora) exec swift sft training/domain_adaptation/configs/qlora_nf4.json "$@" ;;
  *) echo "未知阶段：$stage" >&2; exit 2 ;;
esac
```

QAT 保留上游清晰的“量化初始化 → QAT/QAD 训练 → 导出”三步，由独立配置描述；先确认 27B 量化与导出可用，再考虑加入同一入口。启动脚本不承担自动安装、自动清理、静默回退量化算法或自动覆盖旧输出的职责。

## 8. 下一步实施与验收

1. 先按本文核对目标机器的实际软件栈，并产出一个可回滚的候选环境锁；当前 requirements 与既有环境不直接改动。
2. 将两份原生配置落盘，补齐真实数据 manifest、来源保护、评测和保存参数；先解析/打印实际参数与编码标签。
3. 先验证原生 27B BF16 的训练、恢复和导出，再测试 WSD；需要省显存时单独验证 NF4 QLoRA。
4. 只有 PTQ 在固定评测上出现需要修复的回归，才启动同部署格式的 QAT/QAD 验证。
5. 接受标准同时包含内容、引用、风格、JEV 概率、恢复一致性、导出一致性、峰值显存与吞吐；不把局部 loss 降低或 JSON 能解析当作 GPU 训练成功。

本文中的版本查询、参数源码和公式可以检查；实际 27B 的 QAT/QLoRA、WSD 优劣与新 Megatron 组合均未在目标 GPU 验证。
