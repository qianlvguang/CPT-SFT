# FP8 与 NVFP4：混合精度与部署

本页前半部分讨论 Flash-Next 的两个具体 checkpoint。2026-11-08 新增的 27B 量化调研见 [Qwen3.5-27B 量化方案](qwen35_27b_quantization.md)，不能把这里的 expert/PLE 精度分配复制到稠密 27B。

## 概念

FP8 只有 8 个数据位，但 E4M3、E5M2 的动态范围和精度不同；还必须指明缩放粒度、激活是否动态量化、排除层和 kernel。NVFP4 的数值为 E2M1，典型实现每 16 个元素共享一个 FP8 E4M3 block scale，再配全张量 FP32 scale。W4A4 是指参与对应算子的权重和激活都采用 4-bit 量化，**不代表整个模型所有张量、所有算子或 KV cache 都是 4-bit**。[NVIDIA NVFP4 格式](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)

PTQ 是训练完成后通过校准/量化导出；QAT 是在训练中模拟或使用低精度，需要额外验证优化器、主权重和收敛。这里讨论的是推理 checkpoint，不把 FP8/NVFP4 推理格式误写成 LoRA 训练精度。权重量化、运行时激活量化、KV cache 量化、CPU offload 是四个独立选择。

## 两个具体 checkpoint

| 部件 | Qwen 官方 FP8 | NVIDIA NVFP4 |
|---|---|---|
| 主 LLM routed expert | FP8，权重 128×128 block scale；配置动态激活 | NVFP4 W4A4，16 元素分组；MSE 校准权重 scale |
| 主 LLM attention/GDN、shared expert、router 等 | 参考 checkpoint 保留原 dtype；检查清单 | BF16，见模型卡 |
| PLE n-gram 大表 | per-tensor FP8 | per-tensor FP8，与官方 FP8 表一致 |
| MTP routed experts | 参考模型清单及张量实际 dtype | FP8 128×128 block scale |
| ViT/aligner、其他未列项目 | 检查 checkpoint 清单和 dtype | NVIDIA 的清单排除 `model.visual*`；核对实际 dtype |
| KV cache | 单独配置，不由名称决定 | metadata 标记 `kv_cache_quant_algo: null`；可用性取决于运行时 |

以上是特定仓库的当前模型卡/配置描述，并非所有叫 FP8、NVFP4 的模型都相同。来源：[Qwen FP8 配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/blob/main/config.json)、[NVIDIA 模型卡](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)、[TensorRT-LLM 指南](https://nvidia.github.io/TensorRT-LLM/deployment-guide/deployment-guide-for-qwen3.8-flash-next-on-trtllm.html)。NVIDIA 模型卡指出 NVFP4 无微调，其 MTP 和 PLE 与 Qwen FP8 权重逐字节相同；模型卡基于所列设置的评测不能替代自身业务验证。

## 部署方案

1. **BF16 基线**：用原始权重、固定输入/模板/版本，记录质量、显存、RAM、吞吐、首 token 和 decode 延迟。
2. **官方 FP8**：使用 `Qwen/Qwen3.8-Flash-Next-FP8`，保留模型自己的 metadata。现有 `serve_b300.sh` 使用它；短上下文、小并发先验证，再按机器卡数/拓扑调整 TP、EP。vLLM recipe 推荐 GB300 的 4 卡 TP+EP，但它的调优基于 nightly，并非本机 0.32.0 的验证报告。
3. **NVIDIA 混合 NVFP4**：使用 `nvidia/Qwen3.8-Flash-Next-NVFP4`，保留其 `MIXED_PRECISION`、逐层 quant config、scale 和分片。其模型卡列出需要的 vLLM commit 及 MTP 额外要求；**不能仅由版本号或支持 NVFP4 推断 vLLM 0.32.0 一定能运行该 checkpoint**。检查所用 0.32.0 代码是否包含对应提交/修复，再做加载、输出和 MTP 测试。该仓库许可条款也不同于原始模型，应分别审阅。
4. 不要将 BF16 权重只通过指定 `--quantization modelopt` 当作已生成 NVFP4 checkpoint；量化需支持该架构的转换器、校准、正确元数据与运行时加载器。不要给混合 checkpoint 强行覆盖单一全局精度。

B300 和 RTX PRO 6000 都属于 Blackwell，但 GPU 计算能力、可用内核、单卡显存和互联不同；相同 checkpoint、TP/EP 和 attention/MoE kernel 不能假定有相同性能或兼容性。先检查目标设备与 vLLM 0.32.0 的支持矩阵，按本机实际规模做 A/B。评估多模态、长上下文、工具调用、代码及业务集，关注精度、完整输出和稳定性，而不是只看吞吐。详见 [vLLM 官方模型 recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)。

## 2026-11-08 补充：按模型和导出路径选方案

| 工作线 | 第一条量化路线 | 需要单独处理的部分 |
|---|---|---|
| Qwen3.5-27B Dense | 自训 BF16 对照后先测 FP8 | GDN 门控、视觉栈、自定义决策头和 MTP；专用 NVFP4A16 示例会跳过整段 GDN |
| 原生 Flash-Next | 官方 FP8 建立服务基线，再核对 NVIDIA 混合 NVFP4 | routed/shared experts、router、PLE、MTP 的格式和 scale 不同 |
| 本仓库 35B-A3B QSA 改造 | 先验证改造后的 BF16 导出和推理，再制定 FP8 保护范围 | 自增 indexer、自定义算子和缓存；原模型支持量化不表示改造模型已受支持 |

对自训产物，公共量化模型只能作为原版参考。完整部署仍需包含自己的 adapter 增量、全量保存模块和校准信息。若采用量化基座动态挂载 LoRA，必须另行确认目标模块、rank、视觉适配、后端及具体量化格式的组合支持；它不自动等价于“先合并 BF16、再量化”。

格式与执行路径也要分别记录：ModelOpt 的 `NVFP4`、`W4A16_NVFP4`、混合精度导出，与 compressed-tensors 的 NVFP4A16 元数据并不相同。vLLM 的加载器选择由 checkpoint 决定；缺少原生 FP4 kernel 时，某些路径会转为 Marlin W4A16，不能只看模型名称就按 W4A4 估算吞吐。[vLLM 0.32 ModelOpt 文档](https://docs.vllm.ai/en/v0.32.0/features/quantization/modelopt/)

本项目统一先固定权重、输入模板、任务评测和高精度缓存，再单独切换量化方案。每个候选记录模型 revision、导出器 commit、量化与排除规则、镜像 digest、实际 kernel、任务误差和显存；不把权重量化、KV 量化、MTP 和上下文扩展一起更改后再归因于某一个因素。
