# Phase B：Flash 去冗余 3 步骨干（CoT v0.4 + 生图 v2 + 单阶段 Mid-Training） (`training/flash/`)

在 Phase A Lite 增益确认后，执行去冗余后的 **Flash 三步骨干流水线**（砍掉两阶段 `no_save_optim` 断点、独立风格 SFT 与 Teacher DPO）：
- **Step 1（数据准备）**：CoT v0.4 知识内化（227k→147k 去重 + 规则进 user/实体进 reasoning + Clean Direct Twins 防 Thinking=OFF 崩溃 + CPT 散文剥离 `<think>`）· 生图 v2 按域分区多图单 Doc 打包（`doc_text_repeat=1`）· V1~V4 去重合并；
- **Step 2（单阶段 WSD Mid-Training）**：30% 域文档+多图 / 35% 真实图库 / 15% CF+干扰对比 / 20% 去 `<think>` 百科散文与回放（硬上限：QA 散文 `<=15%`、合成图 `<=25%`、零 raw ChatML、`eval_family_overlap==0`），B300 `MBS=2/GBS=16/8192` + `WSD linear 8e-6`；
- **Step 3（单轮双模统一 SFT）**：统一消费 V1~V4 + `7:3` Think-ON/Clean-Direct 混合包 + 风格样例（`MAX_LENGTH=8192`），结束后强制执行 `Think-ON` / `Think-OFF` 双模验收。

配套方案见 [`docs/training_plans/flash/qa_cot_plan.md`](../../docs/training_plans/flash/qa_cot_plan.md) 与 [`docs/training_plans/flash/image_gen_v2_plan.md`](../../docs/training_plans/flash/image_gen_v2_plan.md)。

## 核心组件

| 文件 | 职责 |
|---|---|
| [`gen_cot_from_qa.py`](gen_cot_from_qa.py) | QA→CoT v0.4 生成与 8 道内化门控（`--cot-mode internalized` 默认 + 兼容 v0.3 `rag_citation`）：227k→147k 规范去重、规则/实体可变性二分、实体锚点覆盖门 `>=0.65`、防 RAG 泄漏、输出 `qa_direct_clean_twins.jsonl`（`>=80%` 直答桶）与 CPT 安全散文 `qa_midtrain_prose.jsonl` |
| [`build_v2_image_pipeline.py`](build_v2_image_pipeline.py) | ImageGen v2 按域分区路由器：`logo/audit/doc` 停、`dex` `mm=True` 走 I2I、`flag/geo` 禁纯 T2I（矢量+Edit）、`move/ability` 节流、`kb_sports` 递归展开、`gen_kb_v1` 3 路回收、`doc_text_repeat=1` Top 2~4 多图单 Doc 打包及错图转 `ch_v2_cf_pair` |
| [`build_mm_wiki_cpt.py`](build_mm_wiki_cpt.py) | Mid-Training 打包器（`--mode midtrain` 为 Flash 主线，`cpt_2stage` 兼容旧 2 阶段）：执行 30/35/15/20 混合、`sanitize_qa_for_cpt`、`<=15%` QA 散文、`<=25%` 合成图与 `doc_family_id` 零泄漏 |
| [`refactor_v1_v4_pipeline.py`](refactor_v1_v4_pipeline.py) | 历史 V1～V4 三元去重分流与三方 SFT 实验臂构造器 |
| [`run_flash_pipeline.sh`](run_flash_pipeline.sh) | Phase B 三步端到端编排脚本 |

## 本地无 GPU 端到端预演

```bash
DEMO=1 DRY_RUN=1 bash training/flash/run_flash_pipeline.sh
python3 training/flash/gen_cot_from_qa.py --demo --smoke-sample 6 --output-dir /tmp/cot_demo
python3 training/flash/build_v2_image_pipeline.py --demo --output-dir /tmp/v2_img_demo
```
