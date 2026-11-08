# B300：MS-SWIFT Megatron LoRA SFT

Qwen3.5-27B 是稠密模型，请使用独立的 [train-qwen35-27b-swift.sh](./train-qwen35-27b-swift.sh)，用法和架构差异见 [Qwen3.5-27B 专项说明](./Model_Guide/05-Qwen3.5-27B.md)。不要把本页 Flash-Next 的 router、expert、PLE 和 Megatron 并行配置套用于 27B。

`train-b300-swift.sh` 基于 MS-SWIFT 的 Qwen3.8-Flash-Next 专用 Megatron SFT 示例，使用原始 BF16 权重，进行文本监督微调（LoRA）。保持现有 **MS-SWIFT 4.7.0.dev** 环境，不会自动安装、升级软件或下载模型以外的依赖。该配置假设单机 **8 张 B300**；实际 GPU 数、显存、CPU RAM 与互联不同，需重新设计并验证并行配置，不要只修改 `NPROC_PER_NODE`。

## 环境准备

在 Linux 目标机激活现有环境，确认 `megatron` 命令可用，并检查 MS-SWIFT 的精确 dev 版本和源码 commit，以及 PyTorch/CUDA、驱动、Megatron-Core、mcore-bridge、Transformer Engine 和相关算子版本。dev 版本可能随 commit 变化，因此应记录这些信息，不能仅凭 `4.7.0.dev` 确认兼容性。

MS-SWIFT 模型专用文档列出 `transformers>=5.16`、`qwen_vl_utils>=0.0.14`、`mcore-bridge`、Transformer Engine、`flash-linear-attention>=0.5.2`、`causal-conv1d` 等依赖，并要求 Megatron 后端。请依据当前环境和官方 Megatron 指南核对；不要直接覆盖现有 PyTorch/CUDA 环境。SFT 脚本不调用 vLLM；之前固定的 vLLM 0.32.0 用于推理，与训练环境分别验证。

BF16 模型文件约 335 GiB，准备额外缓存、训练输出空间和充足的 CPU 内存。脚本设置 `PLE_CPU_OFFLOAD=1`，将 51B N-gram 表卸载到主机；官方文档指出它在 BF16 下约 95 GiB，训练时该表被冻结。LoRA 本身不会更新它，但将本脚本改为全参数训练时必须重新考虑这个行为和内存需求。

## 数据和启动

准备 UTF-8 JSONL，每行一个 `messages` 对话，格式参见 [example-sft.jsonl](./example-sft.jsonl)。示例只有两条，**仅用于检查格式，不是有效训练集**。真实训练应使用有代表性的训练集，并单独准备验证集或配置适当的数据集切分。请确保样本中的 assistant 回答是期望学习的内容，选择适当的思考内容格式和数据许可。

```bash
cd /path/to/Inference
chmod +x train-b300-swift.sh
# 先检查生成的命令，不启动训练或下载模型
DATASET=/data/train.jsonl DRY_RUN=1 ./train-b300-swift.sh
# 确认 8 张卡及资源后启动
DATASET=/data/train.jsonl MODEL=/data/models/Qwen3.8-Flash-Next \
  OUTPUT_DIR=/data/output/qwen-lora ./train-b300-swift.sh
```

也可以使用模型 ID 自动下载：省略 `MODEL` 时默认 `Qwen/Qwen3.8-Flash-Next`。默认使用逻辑 GPU 0–7，可用 `CUDA_VISIBLE_DEVICES=...` 指定 8 张可见卡。`DATASET` 支持本地文件或 MS-SWIFT 可识别的数据集 ID；路径均由训练进程直接访问。

常用可调变量包括 `EPOCHS`、`MAX_LENGTH`、`GLOBAL_BATCH_SIZE`、`LR`、`LORA_RANK`、`LORA_ALPHA`、`SAVE_STEPS` 和 `OUTPUT_DIR`；默认值与上游示例一致。其他 Megatron 参数可在脚本后传入，例如：

```bash
DATASET=/data/train.jsonl ./train-b300-swift.sh --val_dataset /data/val.jsonl --logging_steps 1
```

默认 TP2、EP4、PP2，第一流水线阶段 12 层，开启重计算、padding-free；这是参考上游的起始配置，不是对本机的性能或收敛保证。第一次请用少量真实数据先做短跑，检查训练 loss、实际可训练参数、显存、保存与恢复，再开展正式训练。

训练 checkpoint 的保存格式、导出/合并及能否直接被 vLLM 0.32.0 加载取决于实际 MS-SWIFT/Megatron 版本和 LoRA 目标模块；请以该版本的导出文档为准，先在独立目录导出并做推理一致性验证，勿假设 Megatron 输出可直接用于 vLLM。不要用推理用 FP8 checkpoint 替代此脚本默认的 BF16 训练权重。

## 选择层、router 和 expert

Qwen3.8-Flash-Next 的 48 层按 3 个 Gated DeltaNet 层加 1 个 QSA 层重复排列，每层还有 MoE。模型有 512 个路由 expert、一个 shared expert 和路由器。Megatron 中的模块名称和 Transformers checkpoint 不相同，GDN、QSA、MoE 的实际模块也不同。上游基线 `in_proj out_proj linear_proj linear_qkv` 并不表示已经训练所有 expert 或 router。

通过 `LORA_SCOPE` 选择：

- `baseline`（默认）：上游基线的四类后缀。
- `baseline_router`：基线加 `all-router`；`router`：仅 `all-router`。这是框架的特殊目标选择器，实际注入、梯度和保存行为必须在当前版本核验。
- `modules`：通过空格分隔的 `TARGET_MODULES` 指定 Megatron 后缀，如 `linear_fc1 linear_fc2`；这些后缀也可能匹配 shared expert 或其他 MLP，不能视作仅 routed expert。
- `regex`：提供 `TARGET_REGEX` 按实际 Megatron 模块全名匹配层及模块；此时上游会忽略 `target_modules`，所以 router 必须包含在正则中，不能同时依赖 `all-router`。不要假设训练进程里的层号一定等于原始模型全局层号，PP 每个 stage 的命名和偏移需在当前版本检查。

```bash
DATASET=/data/train.jsonl LORA_SCOPE=baseline_router ./train-b300-swift.sh
DATASET=/data/train.jsonl LORA_SCOPE=modules TARGET_MODULES='linear_fc1 linear_fc2' ./train-b300-swift.sh
# 按实际 Megatron 模块名称替换此表达式；此处仅演示传参形式
DATASET=/data/train.jsonl LORA_SCOPE=regex TARGET_REGEX='YOUR_VERIFIED_REGEX' DRY_RUN=1 ./train-b300-swift.sh
```

若要仅匹配某些层、routed expert、shared expert 或 router，请先在目标环境检查各 PP/EP rank 实际 `named_modules()` 和 `named_parameters()`，以实际命名设计正则，确认并行切分后所有预期模块覆盖且其他模块未被选中。可将收集到的模块名称每行写入一个文件，用 `python3 check-target-regex.py '正则' module-names.txt` 离线检查 Python `re.fullmatch` 的覆盖；还须结合框架实际匹配方式和日志核对。少量训练步之后检查可训练参数名、非零梯度、权重变化以及保存/恢复。分组参数为空或匹配了错误层时，应停止训练。当前环境不能加载该模型，因此这里不提供未经验证的层号或 expert 正则。

`MODULES_TO_SAVE='模块名 ...'` 可传递上游 `--modules_to_save`，用于额外训练并保存指定原始模块；这属于全量训练这些模块，而非给它们添加 LoRA。具体 router 或 expert 能否通过此方式保存并重新加载取决于实际模块实现、并行切分及版本；必须做梯度、checkpoint 和恢复检查。避免让相同模块同时匹配 LoRA 和 modules-to-save，除非已确认框架行为。

## 学习率及训练参数

`LR` 是当前 LoRA 和 `modules_to_save` 等所选可训练参数的**统一基础学习率**，支持 `LR_DECAY_STYLE`、`MIN_LR`、`LR_WARMUP_FRACTION`、`WEIGHT_DECAY`、`CLIP_GRAD`、`LORA_RANK`、`LORA_ALPHA`、`LORA_DROPOUT`；`MOE_AUX_LOSS_COEFF` 和 `MOE_Z_LOSS_COEFF` 可按需设置，未设置时保留框架/模型默认值。辅损数值需通过实验确定；并非开启 router 训练的替代品。额外原生命令参数仍可追加在脚本后。

**当前脚本没有实现 router、expert、不同层在同一次训练中各自使用独立 LR 或各自 LoRA rank。** 已查阅的 Megatron-SWIFT 接口只直接暴露统一 LoRA rank 和基础 LR，以及 ViT/aligner 专用 LR；不能把它们伪装成 router/expert 分组。实现任意分组 LR 和不同 rank 需要与实际安装的 MS-SWIFT、mcore-bridge、Megatron-Core commit 对齐的自定义适配，并验证分布式优化器参数组、学习率调度、checkpoint 恢复和导出。可先用不同 `LORA_SCOPE` 与 `LR` 分别进行独立实验；这不等价于同一次训练的分组 LR。

参考：[MS-SWIFT 4.7.0.dev 模型最佳实践](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html)、[上游训练脚本](https://github.com/modelscope/ms-swift/blob/main/examples/models/qwen4_exp/megatron_sft.sh)、[Megatron-SWIFT 指南](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Quick-start.html)。
