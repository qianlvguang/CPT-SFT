# 训练入口与目录分工

为避免新老方案混淆，本仓库将训练目录明确划分为**当前 B300 统一主线**、**历史 H200 归档基线**与**独立实验/单模型入门入口**：

| 目录 | 定位与状态 | 适用模型与硬件 | 说明 |
|---|---|---|---|
| [domain_adaptation/](domain_adaptation/README.md) | **✅ 当前 B300 主线** | Qwen3.5-27B（8×B300 / MS-SWIFT 4.7） | **多模态 Wiki+RAG 6 通道 WSD CPT + V1～V4 融入重构 + Montessori QA 合成 + 统一 SFT** 全部核心脚本、提示词与示例 |
| [swift/](swift/README.md) | **🧪 单模型独立入口** | Qwen3.8-Flash-Next (B300) / 单卡 27B | 两个独立单脚本：`train_b300_swift.sh`（Flash-Next 单独 LoRA）与 `train_qwen35_27b_swift.sh`（单卡 27B SFT，供 Montessori/Datawheel 复用） |
| [qwen35_27b/](qwen35_27b/README.md) | **📦 历史 H200 归档基线** | Qwen3.5-27B（8×H200 / `ms-swift 4.3.2`） | 早期 H200 V1～V4 三阶段纯 SFT 推荐线与 `legacy/` 历史脚本；保留兼容与回归测试 |
| [qsa_muon_35b_a3b/](../docs/qsa_muon_35b_a3b.md) | **🔬 独立架构探索实验** | Qwen3.5-35B-A3B（8×H200） | QSA 注意力替换与 Muon 优化器改造实验，独立于 27B 主线 |

> 真实训练前仍须在目标硬件验证数据、权重、环境、可训练参数及保存恢复。
