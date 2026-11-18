# Phase C：Pro 错例簇闭环演进与跨仓蒸馏方案

面向已验收 Flash 基线之后的**长周期滚动迭代**，遵循「每轮只改变一个待验证因子、先不动权重验索引与映射、再发起小步增量 CPT 与异模板 SFT」原则。

配套实现：[`training/pro/error_cluster_miner.py`](../../../training/pro/error_cluster_miner.py) · [`training/pro/montessori/README.md`](../../../training/pro/montessori/README.md) · [`training/pro/run_pro_pipeline.sh`](../../../training/pro/run_pro_pipeline.sh)。

---

## 1. 四类错例簇归因与权重更新门控

| 错例类别 | 典型症状 | 对应动作 | 是否更新权重 |
|---|---|---|---|
| `missing_index_evidence` | 缺参考图、实体别名或新版条文导致空召回 | 提名 Top-K 候选并生成索引/别名补全处方（`index_prescriptions.jsonl`），先固定权重复测 Recall | ❌ 否（先验索引） |
| `disposition_mapping_drift` | 实体/关系识别正确，但线上处置动作或模板版本映射错误 | 输出模板映射/版本修正配置（`template_disposition_mapping_fix`），固定权重对照复测 | ❌ 否（先修路由） |
| `style_collision_false_positive` | 撞风格原创作品被误判为二创或同构相似商标误杀 | 构造同载体/同画风正负对照图文样本，经独立人审/核验后入增量 CPT | ✅ 是（CPT） |
| `insufficient_forced_judgment` | 有证据时会判，但关键特征遮挡或无授权证据时强判 | 构造「证据存在 / 证据抹除」配对样本：领域边界入 CPT，不确定性规范表达入**异模板 SFT** | ✅ 是（CPT + 异模板 SFT） |

---

## 2. 50 / 30 / 20 增量 CPT 配比与 `sft_overlap = 0` 隔离

- **增量 CPT 配比**（默认低学习率 `cosine 5e-6`，从已验收父 CPT 开分支）：
  - **50%** 新核验缺口案例（`new_verified_gap`）；
  - **30%** 已验收旧线上模板案例（`accepted_template_replay`）；
  - **20%** 独立 V 系列历史回放（`v_series_replay`）。
- **严格隔离校验**：SFT 候选集强制排除所有出现在增量 CPT、已验收线上模板集与错例修复簇中的 `template_family_id` 与 `doc_family_id`，程序断言 **`sft_overlap == 0`**。

---

## 3. 跨仓进阶闭环（Montessori × OPD × Datawheel）

1. **Montessori-Instruct 局部影响探针**（[`training/pro/montessori/`](../../../training/pro/montessori/README.md)）：
   - 在独立 $\mathcal{D}_{\text{ref}}$ 上计算 $\mathcal{I}(x,y;\mathcal{D}_{\text{ref}})$，结合 QA→CoT 阶段被八道门拦截下的失败 CoT（`qa_dpo_rejected_cots.jsonl`）构成天然同源硬负例，导出 MS-SWIFT DPO / KTO / GRPO 数据。
2. **OPD 异构教师桥接与双通道蒸馏**（[`Coding/OPD`](../../../../OPD/README.md)）：
   - 利用 `bsd_api_bridge.py` 将异构强教师（或人审修正轨迹）作为特权上下文 $z$ 注入冻结自教师计算逐 Token 惩罚权；配合 `depth_lora_router.py`（前中层管技能路由 $\perp$ 末层管事实注入）与 `kl_objectives.py`（ToDi 双通道门控）防止持续演进中的灾难性遗忘。
3. **Datawheel + Qwen-Image 2.1 定向补采**（[`Coding/Datawheel`](../../../../Datawheel/README.md)）：
   - 针对视觉误杀高发实体生成背景重光照与保护区外反事实抹除孪生样本（`diagnose_and_refill_v6.py`），回流至下一轮 Pro 增量 CPT。
4. **低精度量化与 QAT/QLoRA 收尾**：参见同目录 [27B 进阶训练与管理](qwen35_27b_advanced_training_and_management.md)。
