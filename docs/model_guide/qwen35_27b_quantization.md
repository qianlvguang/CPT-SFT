# Qwen3.5-27B 量化方案：FP8、NVFP4 与自训模型部署

调研日期：2026-11-08。覆盖本仓库的 H200、B300 和 96 GB RTX PRO 6000 场景。已核对官方配置、权重索引、量化工具示例与 vLLM 文档；本次没有量化模型、下载权重或运行 GPU 性能测试。

**当前推荐顺序：自训 BF16 模型及任务基线 → 同一模型的 FP8 → 业务评测。** 只有在容量或吞吐目标仍未满足时，再单独评估 NVFP4A16 或经过确认的 NVFP4 W4A4 路径。FP8/NVFP4 都是候选精度，不能预先承诺准确率损失或加速倍数。

## 1. 已核实的事实

| 项目 | 当前证据 | 可以得出的结论 |
|---|---|---|
| 官方 27B FP8 | `Qwen/Qwen3.5-27B-FP8` 的配置和 tensor 索引 | 有明确的官方混合精度基线 |
| Qwen3.5-27B NVFP4A16 | LLM Compressor 0.12.0.1 的专用 VLM 示例 | 有针对 27B 的工具方案，但跳过视觉栈、LM head 和 GDN |
| NVFP4 W4A4 | NVIDIA 格式说明、ModelOpt/vLLM 支持文档 | 原生低精度 GEMM 需要匹配硬件、导出格式和 kernel；不能由文件名判定 |
| 自训 JEV / LoRA | 必须核对自己的导出权重与读取逻辑 | 公共原版量化权重不包含本次自训结果 |

来源：[官方 FP8 配置](https://huggingface.co/Qwen/Qwen3.5-27B-FP8/blob/97f5941bf617e31c5e237364a8602ce3f03a551a/config.json)、[27B NVFP4A16 示例](https://docs.vllm.ai/projects/llm-compressor/en/0.12.0.1/key-models/qwen3.5/nvfp4-vl-example/)、[vLLM 0.32 ModelOpt](https://docs.vllm.ai/en/v0.32.0/features/quantization/modelopt/)。

Qwen3.6、Qwen3.8、MoE 版本的量化结果，以及社区对 27B 的测速，均不直接证明这份自训 3.5-27B 在相同硬件上的效果。这里也没有将未经核对的第三方 checkpoint 选为默认依赖。

## 2. FP8 具体量化了哪些部分

固定 revision 的官方配置采用动态 activation 和 128×128 权重块。排除名单共 **259 项**：112 项属于视觉栈，144 项属于 GDN 的 `conv1d/in_proj_a/in_proj_b`，另有 embedding、LM head 和 `mtp.fc`。这份名单与实际模型名称绑定；它不表示所有 GDN 大矩阵都未量化，也不表示所有浮点状态都变成 FP8。

直接读取两个官方索引的 `metadata.total_size`，得到如下 **tensor 数据字节**，没有下载任何 shard：

| 官方模型 | 索引字节数 | 十进制 GB | GiB |
|---|---:|---:|---:|
| BF16 | 55,562,872,800 | 55.563 | 51.747 |
| FP8 | 30,866,680,160 | 30.867 | 28.747 |

这不是 GPU 驻留显存，也不包括文件头、tokenizer、运行时重排和缓存。不能再用“27B FP8 就是 27 GB”估算服务容量。原始读取结果、revision 和 SHA256 位于 [来源快照](../research_snapshots/qwen35_27b_source_summary_2026_11_08.json)。[BF16 官方索引](https://huggingface.co/Qwen/Qwen3.5-27B/blob/fc05daec18b0a78c049392ed2e771dde82bdf654/model.safetensors.index.json)、[FP8 官方索引](https://huggingface.co/Qwen/Qwen3.5-27B-FP8/blob/97f5941bf617e31c5e237364a8602ce3f03a551a/model.safetensors.index.json)

对于自训模型，建议先采用与上述保护范围相近的 FP8 方案，再按真实模块名排除额外决策头。这里只给保守部署建议，尚未证明它是该任务的最优保护集合。

## 3. NVFP4A16 与 NVFP4 W4A4 分开讨论

NVFP4 使用 4-bit E2M1 数据、每 16 个值的 E4M3 block scale，以及张量级 scale。它与 MXFP4 的格式和缩放规则不同，不能把二者名称互换。只计算数据与 block scale 时，平均为 `4 + 8/16 = 4.5 bit/权重`；这仍未包含其余高精度模块和运行时开销。[NVIDIA NVFP4 格式说明](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

| 路径 | 权重 / 激活 | 关键区别 |
|---|---|---|
| NVFP4A16 | FP4 / 16-bit | 权重压缩路径，不等于原生 W4A4 的计算收益 |
| NVFP4 W4A4 | FP4 / FP4 | 需要 activation scale、合适的校准/转换器及原生 kernel |
| INT4 W4A16 | 整数 4-bit / 16-bit | AWQ/GPTQ 等方案；与 NVFP4 的编码和加载器不同 |
| NF4 / QLoRA | 训练时的低比特基座与 adapter | 不自动产生可供 vLLM 使用的 NVFP4 推理 checkpoint |

LLM Compressor 的 27B 专用示例采用 data-free NVFP4A16，并跳过整个 `linear_attn`，理由是该示例的融合路径不兼容；这是**该版本工具路径的限制**，不是 GDN 永远不能量化。它还专门把未由 HF 模型加载的 MTP 张量补存到输出。[示例及导出步骤](https://docs.vllm.ai/projects/llm-compressor/en/0.12.0.1/key-models/qwen3.5/nvfp4-vl-example/)

用已核对的矩阵尺寸估算：64 层 FFN 约 17.113B、16 层全注意力投影约 1.678B。若只有这两部分按每权重 0.5625 字节存储，其余保持 BF16 原始字节：

```text
估算 tensor 字节 = 55,562,872,800
                 - (17,112,760,320 + 1,677,721,600) × (2 - 0.5625)
                 = 28,551,555,040 字节 ≈ 26.591 GiB
```

这是按排除范围建立的理想估算，未运行转换器，不是导出体积或显存保证。它说明：**保留大量 GDN 与视觉模块的 4-bit 方案，可能与官方 FP8 处于相近体积范围，不能直接宣称整个 27B 只需约 14 GB。** 进一步压缩 GDN 应作为另一套明确的实验，重新验证支持范围与质量。

## 4. 硬件对应的实施顺序

| 硬件 | 第一条路线 | 后续条件分支 |
|---|---|---|
| H200 | BF16 作为质量基线，先测自己的 FP8 | 需要省容量时再测 W4A16；不预期获得 Blackwell 原生 FP4 计算收益 |
| B300 | 先用匹配版本的 FP8 跑通实际服务与评测 | 只有明确找到适配该模型的 W4A4 导出和 kernel 时才比较 NVFP4 |
| RTX PRO 6000 Blackwell 96 GB | 单卡 FP8、16K 初始上下文、小并发 | 容量仍不足或有吞吐目标时，单独评估 W4A16 / W4A4 的具体实现 |

Hopper 与 Blackwell 的原生低精度能力不同；B300 与 RTX 的软件支持也不能只按“都是 Blackwell”推定一致。vLLM 可根据平台为 NVFP4 选择 kernel，不具备原生 FP4 路径时可能使用 Marlin W4A16，需检查实际加载日志。[NVIDIA 硬件支持矩阵](https://nvidia.github.io/TransformerEngine/support_matrix.html)、[vLLM 0.32 ModelOpt 后端说明](https://docs.vllm.ai/en/v0.32.0/features/quantization/modelopt/)

本仓库现有 B300/RTX 服务继续固定 `vllm/vllm-openai:v0.32.0`。外部 `latest` 示例只用于研究；若所需量化方案依赖更新版本，应另建环境记录差异，不能悄悄升级已有服务。量化配置由导出 checkpoint 决定，不能给任意 BF16 模型加一个量化 flag 就当成完成离线校准和导出。

## 5. 自训 LoRA / JEV 的一条落地流程

1. **固定训练产物与协议**：记录基座 revision、adapter 指纹、head 权重、processor、chat template、特殊 token、决策读取位置和校准参数。保留现有 BF16 产物。
2. **导出独立 BF16 候选**：按训练框架正确合并需要合并的 LoRA，并保留 `modules_to_save`、视觉增量、决策头和需要的 MTP。导出前后在同一组输入上检查结果一致性。自定义 head 如果不能被目标框架加载，先解决 serving 适配，量化不能代替这一步。
3. **只做一个 FP8 候选**：在独立量化环境导出，先沿用官方配置的高精度保护思路，并保护自定义 head。工具未识别模块时停止，不能忽略报错后认为整模已完成量化。
4. **独立验证后再部署**：固定 vLLM 镜像 digest，核对加载器、tensor dtype、scale、跳过模块和缓存精度，然后在完全相同的业务输入上比较 BF16 与 FP8。
5. **需要时才进入 4-bit**：先复现有明确定义的 NVFP4A16 路径；需要 W4A4 时另行核对 activation 量化、GDN 支持与 kernel。质量不满足时，再考虑 QAT/QAD，而不是同时展开多个默认配方。

对需要数据的量化算法，建议先固定 1024 条代表样本作为一次校准起点，按真实模态、任务和长度分层抽样，并保存 ID 与 SHA256；这是工程起点，未验证为最优数量。data-free 方案不强行添加校准步骤，但仍需相同的独立评测。多模态任务不能仅凭纯文本校准就宣称覆盖视觉分布。

官方工具路线入口：[LLM Compressor Qwen3.5](https://docs.vllm.ai/projects/llm-compressor/en/0.12.0.1/key-models/qwen3.5/)、[ModelOpt PTQ](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_ptq)。本次没有把未在目标机器验证的转换过程包装成“一键成功”脚本。

## 6. 验证应回答的问题

| 方面 | 固定条件与输出 |
|---|---|
| 任务质量 | 相同独立测试集、模板、解码及 scorer；分类统计每类 precision/recall/F1，保留 BF16/量化逐样本差异 |
| JEV 概率 | Brier / NLL、分桶校准、阈值附近的翻转；temperature/bias 固定比较，并另报重新拟合后的结果 |
| 多模态与长文 | 图像/视频、OCR、小字、多图及分长度测试；确认没有丢图、截断或特殊 token 漂移 |
| 生成与流式 | 首 token、持续输出、结束原因、特殊标记及分块重组；不能把决策结果分段返回当成逐 token 生成 |
| 性能 | 固定输入/输出 token、并发和 batch；记录冷/热启动、TTFT、TPOT、吞吐、P50/P95 及峰值显存 |
| 缓存 | 先固定高精度缓存比较权重，再单独评估 KV 量化；记录 GDN 状态 dtype、Mamba cache 策略和 prefix cache |

若用候选 token 的 logprobs 还原概率，必须核对是否被 top-k/top-p 等后处理截断，确保读取完整候选集合；对自定义 head 则使用该 head 的真实 logits 和训练时约定的归一化。生成文本能正常显示，不证明这些概率已对齐。

权重量化、KV cache 量化与 GDN 状态精度是独立维度；具体内存公式见 [结构说明](qwen35_27b_architecture.md)。在相同权重结果稳定前，不同时切换 KV 精度、MTP、上下文扩展和模板。[vLLM 0.32 KV cache 文档](https://docs.vllm.ai/en/v0.32.0/features/quantization/quantized_kvcache/)

## 7. 当前完成范围

已完成配置、索引、模块尺寸和来源 revision 核对，并保存可检查的字段与推导。尚未完成自训模型导出、量化、GPU 加载、业务质量评估或性能测试；本文不提供未经实测的“损失低于某百分比”“吞吐提升若干倍”结论。

QAT/QAD 与训练显存相关的后续设计见 [27B 进阶训练与管理](../training_plans/qwen35_27b_advanced_training_and_management.md)，其中区分 NF4 QLoRA、ModelOpt NVFP4 QLoRA、QAT 和 QAD 的支持与导出边界。
