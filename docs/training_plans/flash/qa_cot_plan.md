# QA→CoT 逆向桥接与质检方案（`qa_cot_plan.md` v0.3）

> 版本：v0.3（对应 2026-11-18 定稿口径，接入 Phase B Flash 全量线与三方对照 SFT）
> 实现脚本：[`training/flash/gen_cot_from_qa.py`](../../../training/flash/gen_cot_from_qa.py) · 提示词：[`training/flash/prompts/qa_cot_bridge_v03.md`](../../../training/flash/prompts/qa_cot_bridge_v03.md)

---

## 1. 源 QA 资产盘点（`qa/qa_kb_raw.jsonl`，227,656 条）

| 题型 | 条数 | 说明 |
|---|---:|---|
| `fact` | 46,894 | 单一/明确事实、实体属性与条文要点 |
| `condition` | 46,781 | 触发条件、前置顺序、但书与例外 |
| `insufficient` | 43,748 | 最高价值切片：边界模糊、缺失核心信息需拒答/澄清 |
| `audit` | 35,067 | 电商合规与无图审核规则推演 |
| `comparison` | 32,101 | 跨实体/版本/属性多维比对 |
| `synthesis` | 23,059 | 多源综合归纳与规则优先级仲裁 |
| `synonsense`（拼写残行） | 6 | 管线入口自动归一映射为 `synthesis`（保留 `raw_task_type` 审计） |
| **合计** | **227,656** | `answerable=True` 187,023 / `False` 40,633；多证据 113,764（50%）；原产出自 `qwen3.8-flash-next` |

---

## 2. v0.3 三项核心钦定

1. **严格冷酷高精度风格全链贯穿**：
   - `reasoning`（`<think>` 内）采用紧凑、客观的第一人称核验流，禁止 Markdown 死板段头（如 `### 步骤一`）和机械模板拼接；
   - `content`（终答）维持严谨机构化裁决；
   - 全链执行**风格三禁**（禁倒推话术、禁客套寒暄、禁主观臆测）。
2. **Teacher 选型 = 现役部署 `qwen3.8-flash`**：
   - 弃用异构 Gemini 路线，直接调度现役 `qwen3.8-flash`（规避原产出的 `qwen3.8-flash-next` 同源盲区固化）；
   - 开启原生 thinking 流，**`thinking_budget >= 2048`**。
3. **动态篇幅门控：下限硬门 + 上限软门**：
   - **轻量级 (`fact`)**：`180 ~ 300` 字（硬下限 `>= 180` 字，软上限 `300` 字）；
   - **中量级 (`condition` / `comparison` / `insufficient`)**：`350 ~ 600` 字（硬下限 `>= 350` 字，软上限 `600` 字）；
   - **重量级 (`synthesis` / `audit`)**：`500 ~ 900` 字（硬下限 `>= 500` 字，软上限 `900` 字）。
   - 超出软上限仅记 `soft_warnings` 供抽检观测，不因多步细致排查而误杀高质样本；低于硬下限则判定偷懒跳步并拒收/重试。

---

## 3. 逆向桥接（Bridging）防偏置与三类题型深化

### 3.1 破除后见之明偏差（Anti-Hindsight）+ 干扰项注入重映射
- 从同 `cate_id` / 邻近规则库混入 1~3 条高相似但条件/版本不符的硬负例切片；
- **随机打乱 `S1..Sn` 顺序并同步正则重映射 `gold_answer`、`gold_citations`、`gold_evidence` 中的 `[S_old] -> [S_new]`**，彻底消灭“干扰项永远靠后”的位置捷径；
- Teacher Prompt 中显式提供 `distractors_hint` 要求在 `reasoning` 中逐条核验并排除干扰项，同时严禁出现“根据参考答案”等倒推措辞。

### 3.2 三类高价值题型认知动作清单（Checklist）
- **`insufficient`（43,748 条）**：强制执行「**缺失槽位定位 + 反事实双分支推演**」（*"当前材料未披露 X；若后续核实为条件 A，则适用 [S1]…；若为情况 B，则触发 [S2]…；故现阶段无法闭合判定"*）。
- **`audit`（35,067 条）**：QA 对无图，严禁脑补未给出的画面细节；按「**客体/场景要素映射 → 命中主规则 → 排查但书与豁免条款 → 风险边界界定**」展开。
- **`synthesis`（23,059 条）**：在多源对齐中显式加入「**效力层级与时效性仲裁**」（新规覆盖旧规、类目专项细则优先于平台通用规则）。

---

## 4. 八道程序质检门与失败降级路由

| 门控 | 检验规则 | 失败处理 |
|---|---|---|
| **门 1：终答一致性** | `reasoning` 非空，且 `final_answer` 与重映射后的 `gold_answer` 一致 | 重试 / 降级 |
| **门 2：引用精度** | `reasoning_citations ⊆ all_context_citations`，且终答不得误引 `distractor_keys` | 重试 / 降级 |
| **门 3：引用召回** | `gold_citations ⊆ reasoning_citations`（不跳步），且已混入的干扰项须在 `reasoning` 中显式核验排除 | 重试 / 降级 |
| **门 4：摘录子串与纯净度** | 所有 `evidence.quote` 为原文精确子串且 SHA-256 一致；`reasoning` 无嵌套 `<think>` 标签或死板 Markdown 段头 | 重试 / 降级 |
| **门 5：不可判定反事实保持** | `answerable=False` 或 `insufficient` 须包含缺失槽位说明与「若…则…」反事实条件分支 | 重试 / 降级 |
| **门 6：防同义反复** | `reasoning` 与 `content` 的 4-gram 重合率 $\le 0.55$，且逐字摘录原文占比 $\le 45\%$ | 重试 / 降级 |
| **门 7：动态长度门** | 满足 `180 / 350 / 500` 硬下限；超出 `300 / 600 / 900` 记软警告 | 低于硬下限重试 / 降级 |
| **门 8：风格三禁词表** | 拦截倒推禁词、寒暄客套禁词与主观臆测禁词 | 重试 / 降级 |

- **严格降级策略（零模板污染）**：重试仍未过检的样本**严禁**用程序机械拼接 Quote 兜底；直接置 `reasoning = ""` 且设 **`enable_thinking = false`**，路由回 `qa_direct_fallback.jsonl`（直答桶）以维持 Qwen3.5 混合思考切换能力；失败尝试中的不合格 CoT 同步归档至 `qa_dpo_rejected_cots.jsonl`，供 Pro 阶段或偏好对齐复用。

---

## 5. 接入格局（Lite 增益收口后）

在 `Phase A` Lite 增益确认且基线锁定后，执行三方干净对照（推荐 **`CoT : Direct = 8 : 2`** 混合起跑）：

1. **`ood_only`**：`CPT + 外部 OOD SFT` 基线臂；
2. **`ood_plus_qa_direct`**：`CPT + OOD SFT + QA-Direct`（`enable_thinking=false`）；
3. **`ood_plus_qa_cot`**：`CPT + OOD SFT + QA-CoT (80%) + QA-Direct (20%)`。

落地执行流程：`gen_cot_from_qa.py` → `--smoke-sample 50` 人工分层抽检 → 全量扩量 → [`refactor_v1_v4_pipeline.py`](../../../training/flash/refactor_v1_v4_pipeline.py) `--sft-arm ood_plus_qa_cot` 导出训练。
