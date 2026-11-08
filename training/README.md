# 训练方案索引

| 目录 | 模型与任务 | 使用方式 |
|---|---|---|
| [qwen35_27b/](qwen35_27b/README.md) | Qwen3.5-27B，H200×8 多阶段 LoRA SFT | 默认阅读推荐路线；历史脚本单独归档在 `legacy/` |
| [qsa_muon_35b_a3b/](../docs/qsa_muon_35b_a3b.md) | Qwen3.5-35B-A3B，QSA 预热、Muon CPT、全参或 LoRA SFT | 训练脚本、插件和 DeepSpeed 配置放在同一目录 |
| [swift/](swift/README.md) | Qwen3.8-Flash-Next 的 B300 Megatron LoRA，以及单卡 Qwen3.5-27B LoRA | 两个模型分别使用自己的启动脚本 |
| [domain_adaptation/](domain_adaptation/README.md) | 法规/文章/Wiki 原文 CPT、QA 合成、严谨详尽且有引用的风格 SFT | 新增研究方案与示例，未启动真实训练 |

移动目录只调整组织方式和引用，不改变原训练配方。真实训练前仍需在目标硬件验证数据、权重、环境、可训练参数及保存恢复。
