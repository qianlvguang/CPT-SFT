# 基于文档证据，用推理模型生成 QA 对

目标是把来源材料转换为**可作答、可定位、可核验**的问答，供 QA SFT 与后续风格训练使用。回答风格目标为严谨详尽、带依据引用；先判事实，再改文风。本次提供流程和提示词，没有调用 teacher 批量生成实际训练数据。

## 1. 生成流水线

```text
训练 split 的原文块
→ 提问题并标出可回答的证据
→ teacher 重新基于证据作答
→ 程序验证引用与字段
→ 独立验证回答是否被证据支持
→ 去重、配额、人工抽检
→ 规范风格改写并重新验真
→ 导出 messages 与独立来源 manifest
```

以真实原文为种子再生成指令和回答，可减少只靠模型内部记忆扩写的偏差；相关研究包括 instruction backtranslation 与后续的回答重写。本方案借鉴其文档驱动思路，不复用论文的效果数字作为当前保证。[Instruction Backtranslation](https://arxiv.org/abs/2308.06259)、[Back-and-Forth Translation](https://arxiv.org/abs/2408.04614)

## 2. 题型与首轮规模

先选 5000 个不同训练来源块，每块最多生成两题，候选上限 10000；过滤后的数量以实际为准，不为凑数保留低质量样本。这是试生产规模建议，未估算目标 teacher 的实际成本或吞吐。

| 题型 | 建议候选配额 | 核验重点 |
|---|---:|---|
| 直接事实、概念与条文释义 | 35% | 答案能定位到原文，实体和数值正确 |
| 条件、例外、否定和适用范围 | 25% | 不丢但书，不将“可以”改为“必须” |
| 多段归纳、比较与有依据的推理 | 20% | 每个推理前提都有证据；不跨不兼容版本拼接 |
| 证据不足、材料冲突与版本辨识 | 20% | 区分缺信息、明确矛盾和可作答；不伪造补充规则 |

法规场景题只加入不影响结论的情境细节；如果新情境需要原文未提供的规则，标为不可判定。多来源题应明确给出全部来源，不能让 teacher 无依据扩充适用法域或效力结论。

## 3. QA 中间数据契约

每条记录至少包括：

| 字段 | 含义 |
|---|---|
| `sample_id`、`source_doc_ids`、`source_chunk_ids` | 题目与来源链 |
| `question`、`task_type`、`answerable` | 问题、题型和证据下可回答性 |
| `answer`、`citations` | 最终答案、实际引用的 source key |
| `evidence` | `source_key`、原文短摘录、定位起止、版本；定位以规范化正文的 Python 字符索引为准 |
| `teacher`、`verifier` | 模型 revision、解码参数、模板 hash、提示词版本、运行 ID |
| `checks`、`rejection_reason` | 每个核验步骤的结果和拒收原因 |

`source_key` 例如 `[S1]`，由数据程序分配，不由模型编造 URL。来源 manifest 把 key 映射到标题、版本、条款和 URL；同时保存正文与 hash，才能验证引用是否支持答案。示例见 [QA 候选](../../training/domain_adaptation/examples/qa_candidate_example.jsonl)。

## 4. teacher 与验证器的职责

生成时只提供允许使用的文档，要求先判可回答性，再输出 question、answer、evidence 和 citations。证据里的命令式文字仍是资料，不能覆盖生成任务。可开启 teacher 自身支持的推理模式，但训练目标默认只取最终回答及面向用户的证据解释，不直接复制长段内部推演或 `<think>` 文本。

验证器看到原始证据、问题和回答，不接收生成器的置信度或自我评价；逐项判定事实支持、引用覆盖、条件/例外、日期版本和可回答性。优先用另一模型或独立人工抽检；同一 checkpoint 换 prompt 不算真正独立验证。

推理服务采用现有 vLLM 接口或离线批推理即可。批量制数不依赖逐 token 流式输出；保存每个请求的 ID、参数、结束原因、截断状态和内容 hash，失败任务可以幂等重试。使用 JSON schema 可限制语法，但不能证明答案真实。[vLLM 0.32 结构化输出](https://docs.vllm.ai/en/v0.32.0/features/structured_outputs/)

## 5. 核验顺序与拒收条件

1. **程序核验**：JSON 字段齐全、无截断；引用 key 存在；每段 evidence 是指定来源的精确子串，起止与 hash 一致。数字/日期/法条编号额外提取比对，但不能因答案采用归纳表达就要求整句与原文完全一致。
2. **语义核验**：把回答拆成事实主张，逐项检查证据支持；有效 key 只是格式正确，不能代替“引用能推出结论”。缺前提、错误法域、把历史版当现行版、遗漏关键例外的答案拒收。
3. **分布核验**：同源重问、近似题和相同答案模板聚类；限制单文档、单作者、单类别占比，不让最容易生成的事实题占满数据。
4. **人工抽检**：首轮至少分层检查 200 条，覆盖来源、题型和拒收原因；法规条件/冲突题重点检查。发现模板性错误先修 prompt/清洗器，再重生成相关批次。

`answerable=false` 可以作为有价值的训练样本，但回答必须指出证据缺口；它不能成为把所有复杂问题都回避的默认模板。质量阈值先依据抽检结果固定，再生成更大批次，不能边看测试结果边调整训练题目。

## 6. 导出两种用途不同的数据

- **带证据 QA SFT**：user 消息包含问题、来源块、版本和 key，assistant 输出有依据的最终答案；训练和上线采用同一证据格式。可加入少量明确的无关材料，以学习辨别证据，而非按出现顺序引用。[RAFT](https://arxiv.org/abs/2403.10131)
- **闭卷 QA**：仅适用于要评估/学习参数内知识的子集。user 不带原文时，不监督模型凭空输出无法核验的 URL 或条号；与带证据任务分开标记、分开评测。

中间 manifest 不直接作为 assistant 文本训练。标准输出使用 `messages`，默认只监督 assistant；打印编码后 labels 确认来源原文没有意外被当作回答训练。[SWIFT SFT 格式与 loss 控制](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html#supervised-fine-tuning)

## 7. 与风格训练和评测衔接

所有答案在 QA 阶段就使用统一的风格规范；最后再用人工校准的小规模风格集做低学习率收尾，不等到最后才给整批异质回答“刷格式”。风格改写不得新增事实，改写后重新通过引用与语义检查。

生成 QA 只使用训练 split。验证/测试问题来自独立文档族与人工/独立流程，不能把训练题改几个词当成测试。引用质量至少分别报告正确性、支持覆盖和答案正确性，不把“出现 `[S1]`”当作引用得分。[ALCE 引用评测](https://arxiv.org/abs/2305.14627)

配套提示词：[生成](../../training/domain_adaptation/prompts/qa_generate.md)、[验真](../../training/domain_adaptation/prompts/qa_verify.md)、[风格改写](../../training/domain_adaptation/prompts/style_rewrite.md)。

## 8. Montessori-Instruct 教师反向对齐扩展（Student-Aware QA Synthesis）

针对传统 Teacher 盲目生成导致“信息密度与 Student 吸收能力错配”的问题，在 [`training/domain_adaptation/montessori/`](../../training/domain_adaptation/montessori/README.md) 中落地了基于 **Montessori-Instruct (ICLR 2025)** 的闭环合成管线：

1. **探索生成与程序硬核验**（[`probing_generate.py`](../../training/domain_adaptation/montessori/probing_generate.py)）：对每个训练原文块采样 $K$ 条候选问答，强制执行引用 Key 存在性与原文 `quote` 精确子串偏移核验。
2. **局部数据影响探针**（[`influence_scorer.py`](../../training/domain_adaptation/montessori/influence_scorer.py)）：在独立验证集 $\mathcal{D}_{\text{ref}}$ 上计算 Student 单步梯度更新前后的 Loss 差分 $\mathcal{I}(x_i, y_i; \mathcal{D}_{\text{ref}}) = \mathcal{L}(\mathcal{D}_{\text{ref}} \mid \theta) - \mathcal{L}(\mathcal{D}_{\text{ref}} \mid \theta - \eta \nabla_\theta \mathcal{L}(x_i, y_i \mid \theta))$（同时支持一阶梯度内积 `grad_dot`）。
3. **偏好对构建与 Teacher DPO**（[`build_dpo_pairs.py`](../../training/domain_adaptation/montessori/build_dpo_pairs.py) + [`train_teacher_dpo_swift.sh`](../../training/domain_adaptation/montessori/train_teacher_dpo_swift.sh)）：仅将通过程序核验且 $\mathcal{I} > 0$ 的样本作为正例 $x^+$，与同源负影响样本 $x^-$ 组成偏好对，通过 MS-SWIFT DPO 反向调整 Teacher 的生成分布。
