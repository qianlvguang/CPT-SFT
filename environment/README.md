# 环境与依赖

| 文件 | 用途 |
|---|---|
| [requirements_h200.txt (已归档)](../archive/environment/requirements_h200.txt) | 历史 H200 QSA/Muon 训练与推理依赖清单（已归档至 `archive/environment/`） |
| [cuda13_pytorch_notes.md](cuda13_pytorch_notes.md) | CUDA/PyTorch 环境资料、轮子架构检查与编译说明 |
| [scripts/install_official.sh](scripts/install_official.sh) | 安装官方轮子，支持 `--dry-run` |
| [scripts/verify_arch.sh](scripts/verify_arch.sh) | 检查已装 PyTorch 或离线轮子的架构 |
| [scripts/build_from_source.sh](scripts/build_from_source.sh) | 需要自编译时使用的脚本骨架 |

B300 训练主线（`tpz_caitlin` TE `sm_103` 混搭与 APEX fusion 修复）环境矩阵详见 [Lite 战役 Runbook](../docs/training_plans/lite/cpt_27b_runbook.md#5-环境矩阵与两枚实弹排险)；B300/RTX 推理采用 [inference/](../inference/) 中固定版本的容器，MS-SWIFT 微调入口见 [training/README.md](../training/README.md)。不同方案的依赖应分别核对，避免直接覆盖已有训练环境。
