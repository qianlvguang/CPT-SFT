# Phase B：Flash 去冗余骨干、QA→CoT v0.4 与生图 v2 方案

本目录汇总全量骨干阶段（**Flash 3 步流程**）的全部 Mid-Training、生图 v2 与 CoT v0.4 知识内化方案：

| 文档 | 核心内容 |
|---|---|
| [`qa_cot_plan.md`](qa_cot_plan.md) | **QA→CoT v0.4 详案**：227k→147k 规范去重、规则(`user`) vs 实体(`reasoning`) 二分知识内化、8 道内化门、`Clean Direct Twins`（防 Thinking=OFF 崩溃）与 `ch_qa_knowledge_prose` CPT 散文化 5 风险管控 |
| [`image_gen_v2_plan.md`](image_gen_v2_plan.md) | **生图 v2 规范**：8,692 实体按域分区（`logo/audit/doc` 停、`dex` I2I、`flag` 禁纯 T2I、`move` 节流、西甲递归展开）、`gen_kb_v1` 3 路回收、`doc_text_repeat=1` Top 2~4 多图单 Doc 与合成图 `<=25%` 红线 |
| [`b300_multimodal_wiki_rag_cpt_plan.md`](b300_multimodal_wiki_rag_cpt_plan.md) | B300 27B vs Flash-Next 可行性、TP4/DP2 GQA 整除约束、单阶段 Instruct-Aware WSD（30/35/15/20 混合、`MAX_LENGTH=8192`） |
| [`unified_v1_v4_cpt_qa_pipeline.md`](unified_v1_v4_cpt_qa_pipeline.md) | 历史 V1～V4 三元键去重分流进 CPT/SFT 及双模统一 SFT |
| [`qa_synthesis_plan.md`](qa_synthesis_plan.md) · [`answer_style_sft_plan.md`](answer_style_sft_plan.md) | 源文档驱动 QA 合成与机构化风格基线（已融入单轮统一 SFT，砍掉独立风格 SFT） |

可执行代码入口位于 [`training/flash/`](../../../training/flash/README.md)。
