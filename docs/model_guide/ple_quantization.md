# PLE / N-gram embedding 量化研究

## PLE 是什么

PLE 通过 token 的 bigram/trigram 哈希索引大 embedding 表，读取少量行并与当前隐状态结合，注入语言模型早期层。它不是普通视觉 aligner，也不是与注意力 KV cache 相同的缓存。配置中 `ngram_vocab_size_base=20000000`、`ple_embed_dim=2560`、`heads_per_ngram=8`、`ple_layer_ids=[2]`；这些字段不能单独作为实际张量 shape，实际分片、padding 和存储要读权重头部。

51B 参数仅作量级估算：BF16 约 95 GiB，FP8 约 47.5 GiB；4-bit 原始数据约 23.7 GiB，加 1 个 8-bit scale/16 值约 26.7 GiB，另有张量全局 scale、padding、索引和运行时 buffer。数值按 51×10^9 参数 / 2^30 计算，不是实际显存承诺。NVIDIA 和 Qwen 的实际分片、51.2B 与四舍五入会产生差异。卸载到 CPU 只改变驻留位置，不自动减小表，也会引入主机 RAM、传输与延迟。

## 已有证据与边界

| 方案 | 证据 | 结论 |
|---|---|---|
| BF16 PLE | [MS-SWIFT 模型实践](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html) 说明表约 95 GiB，支持 CPU offload | 高精度基线；训练 offload 会冻结表 |
| FP8 PLE | [NVIDIA 模型卡](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) 明确其 PLE 为 per-tensor FP8，来自 Qwen 官方 FP8 | 已存在明确的 checkpoint 及混合精度实现；具体 vLLM 版本需要实测 |
| NVFP4 PLE | [vLLM recipe 源](https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.8-Flash-Next.yaml) 将 Inferact 变体描述为包含 NVFP4 per-layer embedding；但另一[社区仓库检查](https://github.com/zmarty/model-scripts-rtx-pro-6000/blob/main/README-Qwen3.8-Flash-Next-vLLM.md)对 Inferact 的 PLE dtype 描述为 BF16 | 来源存在冲突，不能根据名称断言表是 NVFP4；必须检查确切 revision 的 tensor dtype、shape、scale 和加载器。不能把 per-layer embedding 中的投影与 n-gram 大表混同 |
| 自定义 4-bit PLE | [社区原型](https://github.com/starkweatherdigital/qwen3.8-flash-next-nvfp4-recipe) 描述以 NVFP4 E2M1 和 block scale 压缩表 | 这是第三方工程方案，可能依赖自定义 kernel/patch；未验证其在 vLLM 0.32.0 的正确性、质量或可移植性 |

重要的实践证据：历史上混合 ModelOpt NVFP4 + FP8 PLE 的 vLLM 加载曾出现 `ngram_embedding.weight_scale` 不匹配，见 [vLLM issue](https://github.com/vllm-project/vllm/issues/54765)。NVIDIA 模型卡列了修复所需 commit。保存为 4-bit 并不自动让运行时认识 4-bit embedding；若加载时反量化成 BF16，显存收益可能消失。

## 推荐实验：逐项隔离，先 FP8 再 4-bit

1. **固定基线**：固定 BF16 主干和 BF16 PLE、tokenizer、模板、版本、随机种子、评测数据、解码配置；记录 checkpoint revision 和 sha256。获取对照集 logits、PPL、业务指标和显存/RAM、prefill/decode 时延。
2. **PLE-only FP8**：其余部分不变，仅将 PLE 表量化为 FP8。使用明确可复现的 per-tensor scale 及饱和处理，保留 scale/分片的加载约定。记录读取行的相对误差、cosine、最大误差、饱和率；再比较模型 logits、PPL、长上下文和业务准确率。官方 FP8 checkpoint 的其他层也改变，不能直接将 BF16 与整个官方 FP8 的差异归因于 PLE。
3. **PLE-only 4-bit**：验证实际 kernel 和 checkpoint 约定后，再比较 per-row/per-block 方案，包括 E2M1 block16 + scale 的 NVFP4 候选；不能假定 GEMM 的 W4A4 实现直接支持 embedding gather。分别测 GPU 常驻和 CPU offload，评估 CPU 解码或 GPU 解码、传输量、访存及短/长序列性能。此处是候选研究，不是已经验证的导出流程。
4. **覆盖稀有数据**：按真实 token/ngram 的访问频率、语言、代码、低频词、长上下文切分；同时检查被访问和未访问表行。校准/采样分布不足可能掩盖低频行退化。确定业务可接受阈值后再选格式。
5. **组合评估**：在单变量通过后，再将 PLE FP8/4-bit 与 expert FP8/NVFP4、KV cache 精度和 MTP 组合。分别统计质量、初始化是否成功、实际占用、吞吐/时延和输出差异；量化收益不保证叠加，尤其在 offload 或带宽受限时。

实验结果应包含真实 tensor 名、shape、dtype、scale 粒度、shard 索引、GPU/CPU 驻留及解码位置。针对每个 rank 检查并行切分、加载校验与重启一致性。不要原地覆盖原始权重。当前工作区没有目标模型及 GPU，以上未做定量实验。
