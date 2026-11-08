# Qwen3.8-Flash-Next：结构、训练范围与量化

本目录面向 Qwen3.8-Flash-Next、MS-SWIFT 4.7.0.dev 和 vLLM 0.32.0。资料核对日期：2026-11-07。模型仓库和开发版文档会变化，请为实际实验记录仓库 revision、软件 commit 和镜像 digest。

1. [模型权重结构与训练范围](01-模型结构与训练范围.md)：ViT、aligner、LLM、router、expert、PLE。
2. [FP8 与 NVFP4 方案](02-FP8与NVFP4.md)：实际 checkpoint 的混合精度、部署与验证。
3. [PLE 量化专项研究](03-PLE量化研究.md)：FP8 与 4-bit 的证据、收益、限制和实验计划。
4. [权重检查工具说明](04-权重检查工具.md)：离线扫描 safetensors 头部，按实际 dtype 和模块归类。
5. [Qwen3.5-27B 专项说明](05-Qwen3.5-27B.md)：结构、B300/RTX 推理、MS-SWIFT 训练以及 FP8/NVFP4。

配套推理与训练入口分别在上级目录的 [README](../README.md) 和 [TRAINING](../TRAINING.md)。本目录的量化方案是研究及验证计划，未在目标 B300 或 RTX PRO 6000 上实测，也没有生成量化权重。
