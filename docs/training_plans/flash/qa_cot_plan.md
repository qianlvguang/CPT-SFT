# QA→CoT v0.4「知识内化 + 双模防崩溃 + CPT 散文化安全管控」（`qa_cot_plan.md`）

> 版本：v0.4（2026-11-19 升级，取代 v0.3 RAG 引用型；原 `5k_v2` 的 4,715 条 RAG 引用型归档为离线 A/B 对照组，不参与主训）
> 实现脚本：[`training/flash/gen_cot_from_qa.py`](../../../training/flash/gen_cot_from_qa.py) · 提示词：[`training/flash/prompts/qa_cot_bridge_v04.md`](../../../training/flash/prompts/qa_cot_bridge_v04.md)

---

## 1. 源 QA 资产盘点与前置去重（`qa/qa_kb_raw.jsonl`）

| 指标 | 规模 | 处理策略 |
|---|---:|---|
| 原始总量 | 227,656 | 发现 **80,594（35.4%）完全/近似重复行** |
| **规范去重后唯一 QA** | **147,062** | `dedup_qa_records` 按归一化 `(task_type, question, stripped_answer)` 去重，优先保留 evidence 更完整行 |
| 6 类题型分布 | 全覆盖 | `fact` / `condition` / `insufficient`（19%）/ `audit` / `comparison` / `synthesis`（含 `synonsense` 拼写归一） |

---

## 2. 从 v0.3 到 v0.4 的核心范式跃迁：「知识内化（Knowledge Internalization）」

### 2.1 v0.3 病灶
v0.3 在 `user` 轮放入 `[S1]..[Sn]` 证据束，并在 `<think>` 与终答中强制 `[S1]` 引用。但真实线上推理部署 **没有外部 RAG 检索系统**：若继续按 v0.3 蒸馏，模型学到的是“先翻材料、没见材料不敢断”的条件反射，实体/IP 知识留在 prompt 外壳中，**未压入参数权重**。

### 2.2 按知识可变性二分入路（核心设计）

| 知识性质 | 典型内容 | 放置位置与通道 | 目标可迁移能力 |
|---|---|---|---|
| **规则 / 判定标准 / 流程**<br>（可变、可矛盾、可追加） | 「若仅属常规无序装饰条纹或具备合法备案则豁免」「补正未完成前不开审」 | **`user` 轮「本次审核给定规则 `[R1]..[Rm]`」** | 上线后换新规即可照章遵循，防 rule 幻觉与版本错乱 |
| **实体 / IP / 形象 / 符号知识**<br>（稳定客观、高频复现） | 「示例电鼠P = 黄毛红圆形电气囊黑耳尖锯齿闪电尾」「爱尔兰国旗 = 绿白橙 1:2」 | **从 `user` 轮剥离**；仅在 distill 阶段供 Teacher 对齐，经 **`reasoning` 第一人称直出** 压入参数 | 零检索情况下直接调用参数化领域本体与视觉锚点 |
| **干扰项同法则分流** | 易混淆规则 / 易混淆实体 | 混入 rule 干扰放入 `user` 对照区；**易混淆实体放入 `reasoning` 对比辨析句**（*“需与某白色电鼠区分”*） | 提高抗噪音排查力与同类目防误杀泛化 |

---

## 3. v0.4 八道程序质检门（替换 RAG 引用门）

| 门控 | v0.4 校验逻辑（`verify_cot_v04_8gates`） | 失败处置 |
|---|---|---|
| **门 1：终答一致与格式纯净** | `reasoning` 非空、`final_answer == clean_gold_answer`、无 `<think>` 嵌套/死板标题、终答**零 `[Sx]`** | 重试 / 降级 |
| **门 2：防 RAG 泄漏与规则精度** | 严禁 `[S\d+]`、`根据材料`、`来源[`、`检索结果` 等 RAG 依赖腔调；不捏造 `[Rx]` | 重试 / 降级 |
| **门 3：分层实体锚点覆盖门** | 提取稳定实体锚点 `gold_entity_anchors`，`reasoning` 第一人称叙述覆盖率 **`>= 0.65`**；同时核验给定 `rule` 无遗漏 | 重试 / 降级 |
| **门 4：双轨干扰项同句极性排除** | 干扰规则 `[Rk]` 与易混淆实体名均须在含否定/区分词（`排除/不同于/区分/不适用`）的同一句群内显式排错 | 重试 / 降级 |
| **门 5：不可判定反事实保持** | `insufficient` / `answerable=False` 必含缺失槽位声明 +「若…则…」双分支推演 | 重试 / 降级 |
| **门 6：防同义反复与摘录上限** | 4-gram 重合度 `<= 0.55`，原文连续摘录占比 `<= 45%` | 重试 / 降级 |
| **门 7：动态长度硬下限 / 软上限** | `fact >= 180` / `mid >= 350` / `heavy >= 500`（硬门）；`300/600/900` 记软警告 | 重试 / 降级 |
| **门 8：风格三禁黑名单** | 严禁后见之明倒推词、寒暄套话与主观臆测 | 重试 / 降级 |

> **人工前置门槛（Smoke 60 Gate）**：先产出 `--smoke-sample 60` 分层人工核验（覆盖 6 题型 + 纯实体 + 混合规则实体 + 干扰项对比），全绿后再扩量全量 `147k`。

---

## 4. 高危风险评估一：CPT / Mid-Training 阶段喂 QA 的 5 项潜在风险与管线管控

若将带 ChatML 对话外壳和 `<think>` 开关的 QA 数据直接喂给 `megatron pt`（全序列自回归 loss），会引发五重系统性退化：

| 风险 | 机理 | 本管线硬管控措施（`gen_cot_from_qa.py` + `build_mm_wiki_cpt.py`） |
|---|---|---|
| **1. Prompt/System 模板污染** | `pt` 无 Loss Mask，模型被迫背诵 `system` 与 `[R1]` 格式模版，挤占实体记忆带宽并偏移对话角色先验 | **`qa_midtrain_prose.jsonl` + `sanitize_qa_for_cpt`**：进 CPT 前剥离所有 `system`/`user` ChatML 边界，转为单条百科式散文（`ch_qa_knowledge_prose`）；`allow_raw_chatml_in_cpt=false` 默认强阻断（`raw_chatml_in_cpt_count == 0`） |
| **2. `<think>` 开关损坏** | `<think>` 在 `pt` 阶段被无差别记忆且无真实思考切换决策语义，后续 SFT 开关控制失灵 | **`sanitize_qa_for_cpt` 彻底移除 `<think>` / `</think>` 标签**，`reasoning` 转为连续陈述式「本体推演知识」段落 |
| **3. 可变规则化石化与旧规覆盖新规** | 固定在 CPT 中的历史条款会被视为绝对真理，上线遇新规则时拒绝改口 | **`split_evidence_by_mutability` + `convert_cot_to_midtrain_prose`**：剥离 `[R1]` 动态规则外壳，Mid-Training 散文仅承接稳定实体与推演结构；动态规则仅走 SFT `user` 轮训练条件遵循 |
| **4. 过早收敛 / 呆子化 & SFT 损失失效** | 若 CPT 阶段先吞大量 QA 原文，训练 loss 迅速坍缩至问答对，泛化崩坏且 SFT 验证失真 | 1. **QA 散文 Token 上限硬锁 `<= 15%`**（`max_qa_prose_share=0.15`）；<br>2. **家族零泄漏**（`--exclude-families-from sft_unified_val.jsonl` 等全量剔除 SFT/评测 `doc_family_id`，断言 `eval_family_overlap == 0`） |
| **5. 问答腔污染图文续写本能** | 在图文混编阶段混入过多"问题/结论"体裁会损害原生续写分布 | 仅保留高质量去重百科散文作为 `qa_prose_anti_collapse` 辅助锚，主体仍由 30% 知识文档+多图包 + 35% 真实图库回放 + 15% CF 构成 |

---

## 5. 高危风险评估二：后续实际使用不开启 Thinking（`enable_thinking=False`）会否崩溃？

### 5.1 为什么存在显著崩溃风险？
如果在 v0.4 中，所有关键 IP 实体特征与排错逻辑 **仅写在 `<think>...</think>` 内部**，且混合 SFT 中的直答样本（`enable_thinking=False`）仅靠“CoT 质检失败降级的残次品（`direct_fallback`）”填充，那么在生产环境低延迟请求开启 `enable_thinking=False` 时：
1. 模型直接跳过 `<think>`，失去触发已内化实体记忆的“热身脚手架”，出现严重知识召回跌落与乱猜；
2. 模型把“非思考模式”与“失败样本的低质量短答”错误绑定，触发系统性分布塌陷（**Thinking-OFF Collapse**）。

### 5.2 四道防线避免 Thinking=OFF 崩溃（已落地于管线）

1. **高质直答双胞胎（`build_clean_direct_twin_row` / `qa_direct_clean_twins.jsonl`）**：
   - 从 **已通过 8 道门的优质 CoT 样本** 中同源派生无 `<think>` 直答双胞胎：在终答正文第一行显式前置一句高浓缩实体/规则锚点句（例如 *“核心特征锚点：黄色体毛、红色圆形电气囊、黑色耳尖、闪电尾巴。\n[裁决结论]”*），确保 `Direct` 桶中 **`>= 80%` 为优质直答双胞胎**（`max_fallback_share_in_direct <= 0.20`），彻底切断“降级垃圾垄断直答桶”的毒害；推荐默认 **`7:3`（CoT : Clean-Direct）** 混配。
2. **百科散文前置入参（CPT 层打底）**：
   - 上述知识同时存在于 `v2_packed_cpt_docs` + `ch_mm_doc` + 散文化 `ch_qa_knowledge_prose` 中，即便跳过 `<think>`，基础参数层本身也保有高密度实体表征。
3. **双模 Canary 评估硬门控（`enhance_flash_to_pro.py` `evaluate_canary_and_wise_ft`）**：
   - 每个检查点（Lite / Flash / Pro）必须同时评测 `enable_thinking=True` 与 `enable_thinking=False` 两档指标；若 `think_off_vs_base_drop_pp > 2.0pp` 或 `think_on_off_gap_pp > 8.0pp`，自动标记 **`think_off_collapse_risk = true`** 并触发下一防线。
4. **Pro 阶段 Context Distillation（`OPD` Think-ON→OFF 蒸馏）**：
   - `enhance_flash_to_pro.py` 导出 `pro_opd_bsd_seeds.jsonl`（`distill_mode="think_on_to_off_context_distill"`），在 `OPD` 中以 `Think-ON` Teacher 的推理后概率分布指导 `Think-OFF` Student 分布对齐，实现低延迟免思考模式与满血思考模式几乎同齐（≥95% 能力保留）。
