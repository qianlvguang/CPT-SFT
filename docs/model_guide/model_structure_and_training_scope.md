# 模型结构、权重层与训练范围

## 先看整体

```text
图片/视频 → 视觉预处理 → ViT visual → merger / aligner ─┐
                                                         ├→ LLM → lm_head → 文本
文本 → tokenizer → token embedding ──────────────────────┘
                     │
                     └→ n-gram 哈希查表 → PLE 注入到指定层

LLM: 48 层，每 4 层为 3 层 Gated DeltaNet + 1 层 QSA；每层还有 MoE
MoE: router → 每 token 选择 10 个 routed experts → 聚合；另有 shared expert
MTP: 额外的预测模块，用于投机解码等，不等同于主干层
```

模型卡将主模型记为 125B、每 token 激活约 6B，另有约 51B n-gram embedding 和 4B MTP。激活 6B 不意味着只需装载 6B 权重。原生上下文 262,144 tokens。数据来源：[Qwen 模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)、[原始 config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)。

| 部分 | 职责 | HF 模型中可核对的名称或配置 | 训练观察点 |
|---|---|---|---|
| ViT / vision tower | 图片或视频 patch 编码 | `model.visual`；depth 27、hidden size 1152 | 冻结时不更新视觉编码器；纯文本数据通常不能充分训练它 |
| aligner / merger | 将视觉特征压缩并投影到语言侧 | `model.visual.merger`；输出维度 2560 | 它位于 visual 子树中，但 MS-SWIFT 单独识别；要检查冻结逻辑的优先级 |
| LLM 主干 | 文本、多模态 token 的序列建模 | `model.language_model`；48 层、hidden 2560 | GDN 和 QSA 名称不同；LoRA 目标按框架而异 |
| 输出头 | 隐状态映射至词表 logits | `lm_head`，词表 248320；未绑定输入词嵌入 | 不要假定默认 LoRA 会训练它 |
| router | 对 routed experts 评分并分配 token | HF 配置例 `model.language_model.layers.0.mlp.gate` | 与 shared expert gate 不同；检查 requires_grad、梯度与负载 |
| routed experts | 每个 token 经选中的 10 个 expert 的 FFN 计算 | `model.language_model.layers.N.mlp.experts` | 512 个 expert；EP 分片后每个 rank 只负责一部分 |
| shared expert | 不同于 top-k 路由池的共享 FFN 分支 | `...mlp.shared_expert`、`...mlp.shared_expert_gate` | 不应把 shared gate 误作 routed router |
| PLE | n-gram 查表并向语言隐状态注入信息 | HF `model.language_model.layers.1.ple.ple_embedding.ngram_embedding` | 大表和配套投影/门控不是同一类权重；offload 可能冻结大表 |
| MTP | 多 token 预测辅助模块 | `text_config.mtp`；一层 | 是否加载、训练、量化、启用由具体配置决定 |

上表名称来源于 Qwen 模型配置、[NVIDIA 混合量化清单](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/blob/main/hf_quant_config.json) 和 [MS-SWIFT 模型注册](https://raw.githubusercontent.com/modelscope/ms-swift/main/swift/model/models/qwen.py)。HF 是 checkpoint 命名；Megatron 并行模块、分片和名字可能不同。MS-SWIFT 在当前源码中将该模型注册到 `qwen2_vl` 分组，映射 `model.visual`、`model.visual.merger`、`model.language_model` 和 `lm_head`，见[架构映射](https://raw.githubusercontent.com/modelscope/ms-swift/main/swift/model/model_arch.py)。实际 dev commit 应再次核对。

## 层号与模块

HF 从 0 编号，GDN 层是 `0,1,2,4,5,6,...,44,45,46`；QSA 层是 `3,7,11,...,47`。config 中 `ple_layer_ids: [2]` 与实际 HF 量化清单 `layers.1.ple` 相互印证为第二层，但这是模型特定约定，不要将 config 数值直接当 HF 零基索引。PP 的本地层号与全局层号也可能不同。

每个 QSA 层中 attention 和索引器承担稀疏检索；GDN 使用线性注意力状态。二者之后都有 MoE，另有 hyper-connection/gated residual 的混合参数。输入/输出 embedding、归一化、视觉塔及 MTP 不属于同一种线性层。`linear_qkv` 是 Megatron 融合 QKV 名称，`linear_fc1` 常表示 gate/up 融合、`linear_fc2` 对应 down；不能直接把 HF 的 `q_proj` 等名称当作 Megatron 名称。请参见 [Megatron 参数文档](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Command-line-parameters.html)。

## 训练时的区别

- `freeze_vit`、`freeze_aligner`、`freeze_llm` 的含义随 full/LoRA 和 `target_modules` 选择而变；启动日志、参数名及梯度是最终依据。
- router 可使用上游 `all-router` 选择器；LoRA 是否实际附加和导出取决于当前实现。记录 expert 选择频率、负载均衡、aux loss、router 梯度，防止出现专家塌缩；aux loss 并不等于 router 权重一定训练。
- expert 的 `linear_fc1/linear_fc2` 后缀可能同时覆盖 shared expert 或其他 MLP。若只选特定层/某类 expert，需验证每个 PP/EP rank 上的命名、实际可训练参数和保存恢复，不能仅凭后缀推断。
- `PLE_CPU_OFFLOAD=1` 的上游 Megatron 实现会冻结被卸载的 N-gram 表；LoRA 本来不更新该表，但它不等于完整 PLE 分支都已冻结或已训练。全参数训练尤其要核对。[官方最佳实践](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html)
- `vit_lr`、`aligner_lr` 与统一 LLM 学习率是上游明确提供的类别；本项目脚本不支持同一次训练为 router/expert/各层单独指定 LR 或 rank。训练和推理量化是不同工作流。
