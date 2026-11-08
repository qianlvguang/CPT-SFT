# 环境与依赖

| 文件 | 用途 |
|---|---|
| [requirements_h200.txt](requirements_h200.txt) | H200 QSA/Muon 训练与推理的依赖清单 |
| [cuda13_pytorch_notes.md](cuda13_pytorch_notes.md) | CUDA/PyTorch 环境资料、轮子架构检查与编译说明 |
| [scripts/install_official.sh](scripts/install_official.sh) | 安装官方轮子，支持 `--dry-run` |
| [scripts/verify_arch.sh](scripts/verify_arch.sh) | 检查已装 PyTorch 或离线轮子的架构 |
| [scripts/build_from_source.sh](scripts/build_from_source.sh) | 需要自编译时使用的脚本骨架 |

本目录主要服务 H200 方案。B300/RTX 推理采用 [inference/](../inference/) 中固定版本的容器，MS-SWIFT 微调使用目标机现有训练环境，详见 [训练说明](../training/swift/README.md)。不同方案的依赖应分别核对，避免直接覆盖已有训练环境。
