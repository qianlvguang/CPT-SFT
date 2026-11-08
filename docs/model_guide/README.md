# Qwen 模型结构、训练范围与量化

本目录覆盖 Qwen3.8-Flash-Next 与 Qwen3.5-27B，配套 MS-SWIFT 训练及固定 vLLM 0.32.0 推理入口。Flash-Next 资料核对日期为 2026-11-07，27B 结构与量化补充于 2026-11-08。模型仓库和开发版文档会变化，请为实际实验记录仓库 revision、软件 commit 和镜像 digest。

1. [模型权重结构与训练范围](model_structure_and_training_scope.md)：ViT、aligner、LLM、router、expert、PLE。
2. [FP8 与 NVFP4 方案](fp8_and_nvfp4.md)：实际 checkpoint 的混合精度、部署与验证。
3. [PLE 量化专项研究](ple_quantization.md)：FP8 与 4-bit 的证据、收益、限制和实验计划。
4. [权重检查工具说明](weight_inspection.md)：离线扫描 safetensors 头部，按实际 dtype 和模块归类。
5. [Qwen3.5-27B 专项说明](qwen35_27b.md)：结构、B300/RTX 推理、MS-SWIFT 训练以及 FP8/NVFP4。
6. [Qwen3.5-27B 结构详解](qwen35_27b_architecture.md)：官方尺寸、GDN/全注意力投影、视觉 merger、参数分布和缓存推导。
7. [Qwen3.5-27B 量化方案](qwen35_27b_quantization.md)：实际 FP8 字节数、NVFP4A16/W4A4 边界、硬件选择及自训 JEV 的验证流程。

27B 的固定 revision、配置摘要与推导数据见 [来源快照](../research_snapshots/qwen35_27b_source_summary_2026_11_08.json)。

配套推理与训练入口分别在 [README](../../inference/README.md) 和 [TRAINING](../../training/swift/README.md)。本目录的量化方案是研究及验证计划，未在目标 B300 或 RTX PRO 6000 上实测，也没有生成量化权重。
