# 离线检查实际权重

`inspect_safetensors.py` 只读取本地 `.safetensors` 文件头部，不加载模型 tensor 或依赖 PyTorch、safetensors 库；对巨型 checkpoint 更省内存。传入模型目录，默认扫描该目录的所有 safetensors shard，按 HF 名称启发式分类并列出 dtype、数量和文件内原始数据区间大小。

```bash
python3 tools/inspect_safetensors.py /data/models/Qwen3.8-Flash-Next-FP8
python3 tools/inspect_safetensors.py /data/models/Qwen3.8-Flash-Next-FP8 --filter 'ngram_embedding|weight_scale' --list
python3 tools/inspect_safetensors.py /data/models/Qwen3.8-Flash-Next-FP8 --filter 'visual|merger|mlp.gate|experts' --list
```

在 CPT-SFT 仓库根目录执行。结果是磁盘格式中的 tensor 数据字节，不等于 GPU/CPU 驻留；NVFP4 可能用 `U8` 等 packed 容器，**不能从 dtype 单独推断有效 bit 数或 W4A4**。须同时检查 `config.json`、`hf_quant_config.json`、同目录 scale 张量、模型卡和加载日志。脚本只分类当前可见的本地 shard，缺失分片会导致统计不完整；不读取 checkpoint index 验证完整性。PLE 大表可能分成很多文件，`--filter` 可缩小输出。匹配规则是名称启发式，未知或不同版本命名请手动核对，不用于生成训练目标。
