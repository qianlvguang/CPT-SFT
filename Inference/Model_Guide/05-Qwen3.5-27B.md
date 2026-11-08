# Qwen3.5-27B：结构、训练和量化

Qwen3.5-27B 是稠密多模态模型，有 ViT、视觉 merger/aligner 和语言模型；与 Qwen3.8-Flash-Next 不同，它**没有该 MoE 的 router、routed/shared experts 或 51B PLE n-gram 表**。因此 router/PLE 分析、EP 并行和 PLE CPU offload 不适用。[Qwen 模型卡](https://huggingface.co/Qwen/Qwen3.5-27B)、[模型配置](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json)

| 项目 | Qwen3.5-27B | Qwen3.8-Flash-Next |
|---|---|---|
| 类型 | Dense 27B | 125B 主模型 MoE，另有约 51B PLE、4B MTP |
| LLM | 64 层；48 GDN + 16 Gated Attention，每层 dense FFN | 48 层；36 GDN + 12 QSA，每层 MoE |
| LLM hidden / FFN | 5120 / 17408 | 2560 / expert 中间维度 640 |
| ViT / aligner | ViT depth 27、hidden 1152；merger 输出 5120 | ViT depth 27、hidden 1152；merger 输出 2560 |
| Router / PLE | 无 | 512 routed experts、top-10 + shared expert；PLE 大表 |
| 原生上下文 | 262144 | 262144 |

HF 零基下 Gated Attention 在层 3、7、11……63；其余为 GDN。典型 HF 路径包括 `model.visual`、`model.visual.merger`、`model.language_model.layers.N` 和 `lm_head`。`mlp` 是 dense FFN；名称中的 `gate_proj` 是 FFN 的门控投影，并非 MoE router。MS-SWIFT 对该模型使用 `qwen2_vl` 架构映射；实际版本和训练后端的模块命名可能变化，使用正则选择层时应检查实际 `named_modules()`。[MS-SWIFT 模型注册](https://raw.githubusercontent.com/modelscope/ms-swift/main/swift/model/models/qwen.py)

## 推理

上级目录 [serve-qwen35-27b.sh](../serve-qwen35-27b.sh) 默认使用 `Qwen/Qwen3.5-27B-FP8`，服务名 `qwen35-27b`，TP1、16K 上下文；同一脚本用于 B300 和 RTX PRO 6000，目标硬件需实测。BF16 用 `MODEL=Qwen/Qwen3.5-27B`。官方 vLLM recipe 列出单卡 FP8 和更高显存下的 BF16，实际显存还依赖缓存、上下文、图像、并发和运行时。脚本不默认启用 MTP。[vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.5-27B)

## B300 上 MS-SWIFT LoRA 训练

使用已有 MS-SWIFT 4.7.0.dev 环境，执行：

```bash
cd /path/to/Inference
chmod +x train-qwen35-27b-swift.sh
DATASET=/data/train.jsonl DRY_RUN=1 ./train-qwen35-27b-swift.sh
DATASET=/data/train.jsonl MODEL=/data/models/Qwen3.5-27B \
  OUTPUT_DIR=/data/output/qwen35-27b ./train-qwen35-27b-swift.sh
```

该入口按 MS-SWIFT 推荐的 dense 模型 Transformers 后端 `swift sft` 设计，默认单卡 BF16 LoRA、冻结 ViT 和 aligner、`all-linear`、2K 长度。它是起始配置，尚未在 B300 测试；显存需求依数据长度、图像、LoRA 范围和优化器而变化。单卡容不下时请降低长度/批量或按上游指南配置多卡/ZeRO；增加 `NPROC_PER_NODE` 前要检查 GPU、分布式训练配置和有效全局批量。多模态数据需匹配上游 JSONL 图片/视频格式，样例文本数据不能代替真实训练集。

`LORA_SCOPE=modules TARGET_MODULES='q_proj v_proj'` 可选 HF 模块后缀；`LORA_SCOPE=regex TARGET_REGEX='...'` 可按当前实际 HF 名称选择层；`MODULES_TO_SAVE` 可指定额外全量训练模块。`FREEZE_VIT`、`FREEZE_ALIGNER`、`VIT_LR`、`ALIGNER_LR` 可按需设置，但目标匹配、可训练参数和数据须对应检查；仅设置 LR 不会解冻模块。`LR` 是基础学习率；无任意层分别设置 LR/rank 的实现。继续保留原有 Flash-Next 的限制说明。[MS-SWIFT 最佳实践](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_8-Best-Practice.html)

## FP8 和 NVFP4

官方 `Qwen/Qwen3.5-27B-FP8` 的 config 显示 dynamic activation、128×128 权重 block 与排除清单，例如 embedding、lm_head 和部分 GDN 参数。具体 tensor 和 dtype 由 checkpoint 与运行时决定，不要把 Flash-Next 的逐 expert 量化表套用到 27B。[FP8 配置](https://huggingface.co/Qwen/Qwen3.5-27B-FP8/blob/main/config.json)

NVFP4 方面可评估 ModelOpt/社区 checkpoint，但需要单独确认其是否为 **W4A4**、哪些 attention/GDN/FFN 层被量化、ViT/merger 和 MTP 是否保留、calibration、模型许可与 vLLM 0.32.0 的加载支持；不能因为模型名含 NVFP4 就认为全模型 W4A4 或支持 MTP。先做 BF16 对照，再固定数据和硬件比较 FP8/NVFP4 的输出质量、显存、prefill/decode 延迟和多模态表现。已有 Flash-Next 的 PLE 专项量化方案不适用于本模型。
