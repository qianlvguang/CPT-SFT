# Qwen3.8-Flash-Next 推理环境

## Qwen3.5-27B

已增加 [单卡推理入口](./serve-qwen35-27b.sh)，适用于 B300 和 96 GB RTX PRO 6000 的 FP8 起始配置；实际上下文、并发和显存仍需实测。它使用独立服务名 `qwen35-27b`：

```bash
chmod +x ./*.sh
DRY_RUN=1 ./serve-qwen35-27b.sh
./serve-qwen35-27b.sh
# 在另一终端
SERVED_MODEL_NAME=qwen35-27b python3 smoke-test.py
# BF16：先确认显存余量
MODEL=Qwen/Qwen3.5-27B ./serve-qwen35-27b.sh
```

两个服务同时运行时请分别设置 `PORT`。推理仍固定 `vllm/vllm-openai:v0.32.0`。训练入口、模型结构及量化范围见 [Qwen3.5-27B 专项说明](./Model_Guide/05-Qwen3.5-27B.md)。

面向 Linux + NVIDIA GPU，使用 `vllm/vllm-openai:v0.32.0` Docker 镜像；提供 B300 和 RTX PRO 6000 启动入口。默认使用官方 `Qwen/Qwen3.8-Flash-Next-FP8` 权重，暴露 OpenAI 兼容 API。vLLM 0.32.0 是按要求固定的版本；目前不是最新版本。模型的 BF16 原始权重可通过 `MODEL` 切换。

## 准备

1. 在目标 Linux 主机安装支持目标 GPU 和镜像 CUDA 运行时的 NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit。先用 `nvidia-smi` 查看 GPU、显存和驱动，并检查 `docker run --rm --gpus all --entrypoint nvidia-smi vllm/vllm-openai:v0.32.0`。
2. 准备网络及足够的磁盘空间；FP8 checkpoint 约 172.78 GiB，BF16 约 335.28 GiB，缓存和下载临时文件还需要额外空间。首次启动会下载模型；如需认证，在终端设置 `export HF_TOKEN=...`，不要把 token 写入脚本。
3. N-gram embedding 可卸载至主机内存；官方建议至少 51 GB RAM 加运行余量。实际 RAM、显存和性能取决于权重、并行方式、上下文和并发。确保共享内存和系统内存充足。
4. 将此目录复制到目标机，赋予权限：`chmod +x ./*.sh`。下面命令均在该目录执行。

## B300 / GB300

默认单卡 FP8，适合先验证服务；官方 recipe 记录单卡 GB300 可容纳 FP8，但未在本项目验证 B300 实机。

```bash
./serve-b300.sh
```

如果有 4 张 GPU，可试官方 GB300 recipe 的 TP+EP 方案（B300 仍需自行验证）：

```bash
TP_SIZE=4 EXPERT_PARALLEL=1 ./serve-b300.sh
```

BF16 示例（需要足够的显存，官方 recipe 的 GB300 TP2 实测每卡约 190 GiB；B300 需实测）：

```bash
MODEL=Qwen/Qwen3.8-Flash-Next TP_SIZE=2 ./serve-b300.sh
```

## RTX PRO 6000 Blackwell

默认假设 **4 张 96 GB 卡**、官方 FP8 权重和 TP+EP；这是保守的起始配置，并非官方 recipe 对 RTX PRO 6000 的实测保证，须检查卡数、显存、RAM、互联和兼容性。GPU 少于 4 张时不要直接使用默认值。尤其单张 96 GB 卡不能假定能直接装载官方 FP8 权重；需选用经验证的量化权重、卸载策略及匹配的硬件，并验证质量与性能。

```bash
./serve-rtx-pro-6000.sh
```

通过 `TP_SIZE`、`EXPERT_PARALLEL`、`MODEL` 调整方案。不同权重的量化格式、分片和可用内核各不相同，不保证任意 TP 数或第三方 checkpoint 可用。官方 FP8 在 8 卡时须启用 EP，不能使用纯 TP8。

## 常用配置与验证

```bash
# 查看最终命令，不启动容器（仍会创建缓存与模型目录）
DRY_RUN=1 ./serve-b300.sh
# 本地权重：主机目录只读挂载到容器 /models
MODEL_DIR=/data/models MODEL=/models/Qwen3.8-Flash-Next-FP8 ./serve-b300.sh
# 改上下文和端口；默认监听本机、上下文 16384、最大并发序列 16
MAX_MODEL_LEN=32768 MAX_NUM_SEQS=16 PORT=8001 ./serve-b300.sh
# 可追加 vLLM 原生参数（例如按需启用工具解析）
./serve-b300.sh --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
# 另开终端，在服务加载完成后测试；Python 3，无第三方依赖
python3 smoke-test.py
```

自定义端口时测试命令使用 `API_BASE=http://127.0.0.1:8001 python3 smoke-test.py`。可设置 `SERVED_MODEL_NAME` 统一更改服务端和客户端模型名。若设置服务端 `--api-key`，测试时设置 `API_KEY`。`HF_CACHE` 可更改 Hugging Face 缓存目录；`CUDA_VISIBLE_DEVICES` 可选择容器内 GPU。脚本默认绑定主机 `127.0.0.1`，远程访问需自行设置 `BIND_ADDRESS=0.0.0.0` 并配置认证和网络访问控制。

模型原生上下文为 262,144 tokens，默认使用 16,384 降低初始资源需求。更长上下文和并发需单独测量显存及质量。脚本不默认开启投机解码、FP8 KV cache 或其他 nightly recipe 优化；建议基础服务验证后再测试。模型默认思考模式；测试请求显式关闭思考模式。

## 训练说明

vLLM 主要用于推理服务，并不是完整的训练框架。B300 上使用 MS-SWIFT 4.7.0.dev 进行 Megatron LoRA SFT 的脚本及说明见 [TRAINING.md](./TRAINING.md)。训练与推理环境、训练输出和 vLLM 兼容性均需在目标机器验证。

## 故障排查

- 容器无法看到 GPU：先检查驱动、Docker 的 NVIDIA runtime 和 GPU 选择。
- 启动 OOM：检查实际卡数和显存，降低上下文或并发，增加卡数，或采用经过验证的量化/卸载方案；权重本身放不下时，仅降低上下文未必有效。
- 加载缓慢或失败：检查磁盘余量、网络、模型访问权限、CPU RAM 和完整日志；首次下载耗时可能很长。
- 版本及启动参数以固定的 0.32.0 镜像为准。官方 recipe 使用 nightly 调优，不代表其全部参数已经在 0.32.0 和这里的硬件上验证。

参考：[模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)、[官方 FP8 权重](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)、[vLLM 模型 recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)、[vLLM 0.32.0 发布说明](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)。
