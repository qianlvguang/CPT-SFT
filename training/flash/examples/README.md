# 示例数据与独立配置模板索引 (`training/flash/examples/`)

本目录集中存放**自编教学/联调样例**与**独立配置模板**（均非真实业务全量训练集）：

## 1. Phase B Flash 联调样例

| 文件 | 用途 | 消费脚本 |
|---|---|---|
| [cpt_mm_6ch_example.jsonl](cpt_mm_6ch_example.jsonl) | 6 通道多模态 CPT 示例（纯文本索引桩、标准图元、Qwen-Image 合成、爬取反事实、Flash-Next RAFT、定位回放） | [build_mm_wiki_cpt.py](../build_mm_wiki_cpt.py) |
| [v1_v4_legacy_sample.jsonl](v1_v4_legacy_sample.jsonl) | V1～V4 历史数据接入样例（含 V4 同 `sample_id` 多图样本与 `enable_thinking=false`） | [refactor_v1_v4_pipeline.py](../refactor_v1_v4_pipeline.py)、[run_flash_pipeline.sh](../run_flash_pipeline.sh) |
| [qa_kb_raw_example.jsonl](qa_kb_raw_example.jsonl) | QA→CoT v0.3 源数据示例（覆盖 6 类题型、`synonsense` 拼写残行、`answerable=false` 与同域干扰项） | [gen_cot_from_qa.py](../gen_cot_from_qa.py) |
| [source_documents_example.jsonl](source_documents_example.jsonl) | 领域原文登记样例（教学合成规则） | [`../../pro/montessori/probing_generate.py`](../../pro/montessori/probing_generate.py) |
| [qa_candidate_example.jsonl](qa_candidate_example.jsonl) | QA 候选与引文定位中间数据结构 | [`../../pro/montessori/influence_scorer.py`](../../pro/montessori/influence_scorer.py) |
| [style_sft_example.jsonl](style_sft_example.jsonl) | 严谨详尽、带 `[S1]` 引用风格 SFT 对话样例 | [run_flash_pipeline.sh](../run_flash_pipeline.sh)、[`../../pro/montessori/run_montessori_pipeline.sh`](../../pro/montessori/run_montessori_pipeline.sh) |
| [sample_provenance_example.jsonl](sample_provenance_example.jsonl) | 训练样本到原始文档的审计溯源示例 | 审计校验 |

> Lite 阶段 `sch + kb` 演示样例位于 [`../../lite/examples/lite_sch_kb_sample.jsonl`](../../lite/examples/lite_sch_kb_sample.jsonl)，Pro 阶段错例簇演示样例位于 [`../../pro/examples/pro_error_clusters_sample.jsonl`](../../pro/examples/pro_error_clusters_sample.jsonl)。

## 2. 独立原生配置模板（`configs/`）

> **注意**：主线启动脚本（`training/cpt_27b/run_cpt_27b_b300.sh` / `training/flash/run_flash_pipeline.sh`）直接通过环境变量拼接显式参数，**不会隐式读取本目录 JSON**；以下文件用于查阅超参对照或手动传入 `megatron` / `swift` 命令。

| 文件 | 对应阶段 |
|---|---|
| [cpt_27b_mm_rag_wsd_megatron.json](configs/cpt_27b_mm_rag_wsd_megatron.json) | B300×8 `megatron pt` 27B 全参 WSD CPT 参考配置 |
| [cpt_27b_mm_rag_swift_lora.json](configs/cpt_27b_mm_rag_swift_lora.json) | `swift pt` 27B BF16 LoRA CPT 参考配置 |
| [sft_27b_unified_v1_v4_qa_b300.json](configs/sft_27b_unified_v1_v4_qa_b300.json) | CPT 产物接续 V1～V4 + 新 QA/CoT 统一 SFT 参考配置 |

## 3. 早期概念与跨阶段设计示意（历史参考）

| 文件 | 说明 |
|---|---|
| [cpt_example.jsonl](cpt_example.jsonl) | 2026-11-08 早期单条纯文本 `assistant` CPT 格式示例（已被 6 通道多模态样例 [cpt_mm_6ch_example.jsonl](cpt_mm_6ch_example.jsonl) 扩展替代） |
| [recipe_example.json](recipe_example.json) | 2026-11-08 跨阶段设计配比总览示意，**不可直接传给任何训练器** |
