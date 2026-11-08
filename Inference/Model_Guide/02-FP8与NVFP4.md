# FP8 与 NVFP4（W4A4）

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
2. **官方 FP8**：使用 `Qwen/Qwen3.8-Flash-Next-FP8`，保留模型自己的 metadata。现有 `serve-b300.sh` 使用它；短上下文、小并发先验证，再按机器卡数/拓扑调整 TP、EP。vLLM recipe 推荐 GB300 的 4 卡 TP+EP，但它的调优基于 nightly，并非本机 0.32.0 的验证报告。
3. **NVIDIA 混合 NVFP4**：使用 `nvidia/Qwen3.8-Flash-Next-NVFP4`，保留其 `MIXED_PRECISION`、逐层 quant config、scale 和分片。其模型卡列出需要的 vLLM commit 及 MTP 额外要求；**不能仅由版本号或支持 NVFP4 推断 vLLM 0.32.0 一定能运行该 checkpoint**。检查所用 0.32.0 代码是否包含对应提交/修复，再做加载、输出和 MTP 测试。该仓库许可条款也不同于原始模型，应分别审阅。
4. 不要将 BF16 权重只通过指定 `--quantization modelopt` 当作已生成 NVFP4 checkpoint；量化需支持该架构的转换器、校准、正确元数据与运行时加载器。不要给混合 checkpoint 强行覆盖单一全局精度。

B300 和 RTX PRO 6000 都属于 Blackwell，但 GPU 计算能力、可用内核、单卡显存和互联不同；相同 checkpoint、TP/EP 和 attention/MoE kernel 不能假定有相同性能或兼容性。先检查目标设备与 vLLM 0.32.0 的支持矩阵，按本机实际规模做 A/B。评估多模态、长上下文、工具调用、代码及业务集，关注精度、完整输出和稳定性，而不是只看吞吐。详见 [vLLM 官方模型 recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)。
