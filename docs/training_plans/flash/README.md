# Phase B：Flash 全量 WSD 与 QA→CoT v0.3 方案

本目录汇总全量覆盖阶段（**Flash**）的全部多模态 CPT、V1～V4 重构与 QA/CoT 合成方案：

| 文档 | 核心内容 |
|---|---|
| [`qa_cot_plan.md`](qa_cot_plan.md) | **QA→CoT v0.3 详案**：227,656 条 `qa_kb_raw.jsonl` 题型台账、`qwen3.8-flash`（thinking ≥ 2048）、冷酷核验风格三禁、八道程序质检门、动态长度（180/350/500 硬下限 + 300/600/900 软上限）、8:2 混配与三方对照设计 |
| [`b300_multimodal_wiki_rag_cpt_plan.md`](b300_multimodal_wiki_rag_cpt_plan.md) | B300 27B vs Flash-Next 可行性、TP4/DP2 GQA 整除约束、6 通道（+`v2.3 sch` 并入）WSD 两阶段课程与 Qwen-Image 2.1 反事实规范 |
| [`unified_v1_v4_cpt_qa_pipeline.md`](unified_v1_v4_cpt_qa_pipeline.md) | 历史 V1～V4 三元键去重分流进 CPT/SFT 及监督 Token 重平衡 |
| [`qa_synthesis_plan.md`](qa_synthesis_plan.md) | 源文档驱动 QA 合成、程序偏移/Hash 核验与独立语义验证 |
| [`answer_style_sft_plan.md`](answer_style_sft_plan.md) | 严谨详尽、带 `[S1]` 依据引用的机构化风格规范与 2000 条风格收尾 |

可执行代码入口位于 [`training/flash/`](../../../training/flash/README.md)。
