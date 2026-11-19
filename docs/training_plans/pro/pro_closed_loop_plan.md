# Phase C：Pro 同源程序增强、Thinking-OFF 蒸馏与多轮闭环演进方案

面向已验收 Flash 基线，Pro 遵循「**与 Flash 共用同一份原始数据资产，依靠 4 项程序化增强拉高 Mid-Training 质量上限，并以多轮单调版本账本持续迭代**」原则。

配套实现：[`training/pro/enhance_flash_to_pro.py`](../../../training/pro/enhance_flash_to_pro.py) · [`training/pro/error_cluster_miner.py`](../../../training/pro/error_cluster_miner.py) · [`training/pro/montessori/README.md`](../../../training/pro/montessori/README.md) · [`training/pro/run_pro_pipeline.sh`](../../../training/pro/run_pro_pipeline.sh)。

---

## 1. 同源 4 项程序增强（`enhance_flash_to_pro.py`，零新增原始语料）

1. **信息密度与长尾精炼（`curate_high_value_midtrain`）**：对 Flash 同一套 `midtrain_train` Pool 按长尾未闭合、多图信息量、CF/干扰对比强度打分筛选，裁减重复模板噪点，同时锁定 `max_row_repeats <= 2`、`qa_prose <= 15%` 与 `doc_family_overlap == 0`；
2. **跨模态实体锚点对齐审计（`audit_cross_modal_anchor_alignment`）**：关联 `v2_packed_cpt_docs` 多图实体与 `qa_cot_passed`（CoT v0.4），标注 `cross_modal_aligned=True` 并导出纯视觉/纯文本断链清单（`alignment_audit.json`）；
3. **双模 Canary + WiSE-FT 权重插值门控（`evaluate_canary_and_wise_ft`）**：
   - 通用推理/指令遵循/OCR 基准掉点 `> 2.0pp` → 自动触发 **WiSE-FT**（`0.8 × CPT + 0.2 × Base`）生成 `wise_ft_merge_recipe.json`；
   - **Thinking-OFF 塌陷检测**：若 `think_off_vs_base_drop_pp > 2.0pp` 或 `think_on_off_gap_pp > 8.0pp` → 自动置 `think_off_collapse_risk=True`，推荐启动 `OPD` 上下文蒸馏；
4. **学生偏好对齐 + OPD Think-ON→OFF 上下文蒸馏（`build_pro_preference_and_bsd_pack`）**：
   - 将 CoT 阶段失败轨迹配对成**学生 DPO/KTO 样本**（`pro_dpo_kto_pairs.jsonl`，严禁用于 Teacher DPO）；
   - 导出 `pro_opd_bsd_seeds.jsonl`（`distill_mode="think_on_to_off_context_distill"`），在 `Coding/OPD` 中以 `Think-ON` 分布对齐 `Think-OFF` 学生 logits，确保不开启 thinking 时不掉点。

---

## 2. 多轮错误簇归因、50/30/20 增量 CPT 与单调账本（`iteration_ledger.json`）

| 错例类别 | 典型症状 | 对应动作 | 是否更新权重 |
|---|---|---|---|
| `missing_index_evidence` | 缺参考图、实体别名或新版条文 | 补齐索引/别名处方（`index_prescriptions.jsonl`），先固定权重复测 Recall | 否（先验索引） |
| `disposition_mapping_drift` | 实体识别正确，处置或模板版本映射漂移 | 更新模板路由映射，固定权重对照复测 | 否（先修路由） |
| `style_collision_false_positive` | 撞风格原创误判 / 同构商标误杀 | 同载体正负对照图文入增量 CPT | 是（CPT） |
| `insufficient_forced_judgment` | 缺关键证据时强判侵权 | 「证据存在/抹除」配对：领域入 CPT，拒答入异模板 SFT | 是（CPT + SFT） |

- **多轮版本账本（`iteration_ledger.json`）**：`error_cluster_miner.py` 接受 `--round-id`、`--parent-ckpt` 与 `--prev-ledger`，跨轮单调累计 `cumulative_excluded_template_families` 与 `cumulative_excluded_doc_families`，每轮强制断言 **`sft_overlap == 0`**，保证滚动升级长期可追溯且永不污染评测/SFT 家族。

---

## 3. 跨仓进阶闭环（Montessori × OPD × Datawheel）

- **Montessori-Instruct**：参考 $\mathcal{I}(x,y;\mathcal{D}_{\text{ref}})$ 与黑盒 API 记忆引导，精准剔除负影响伪 CoT。
- **OPD**：`bsd_api_bridge.py` + `kl_objectives.py`（ToDi）+ `depth_lora_router.py`（技能层/事实层正交分配）。
- **Datawheel**：针对高误杀难例通过 [`diagnose_and_refill_v6.py`](../../../../Datawheel/tools/diagnose_and_refill_v6.py) + Qwen-Image 2.1 定向补采后回流下一轮 Pro 账本。
