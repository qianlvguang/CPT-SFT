# 示例数据与独立配置模板索引 (`training/flash/examples/`)

本目录集中存放**自编教学/联调样例**与**独立配置模板**（均非真实业务全量训练集）：

## 1. Phase B Flash 联调样例

| 文件 | 用途 | 消费脚本 |
|---|---|---|
| [qa_kb_v04_example.jsonl](qa_kb_v04_example.jsonl) | **QA→CoT v0.4 知识内化示例**（7 行：含规则/实体混排、纯实体、纯规则、`insufficient`、`synonsense` 拼写与 1 条刻意重复行测试去重） | [gen_cot_from_qa.py](../gen_cot_from_qa.py) |
| [v2_image_entities_sample.jsonl](v2_image_entities_sample.jsonl) | **生图 v2 实体按域分区示例**（12 实体：覆盖停跑域、动漫 `mm=True/False`、招式节流、品牌解剖、国旗 Edit-Only 与西甲俱乐部父子递归展开） | [build_v2_image_pipeline.py](../build_v2_image_pipeline.py) |
| [v2_image_v1_audit_sample.jsonl](v2_image_v1_audit_sample.jsonl) | `gen_kb_v1` 三路回收示例（Clean 晋升 / 畸变转 CF 负例 / 冗余清理） | [build_v2_image_pipeline.py](../build_v2_image_pipeline.py) |
| [cpt_mm_6ch_example.jsonl](cpt_mm_6ch_example.jsonl) | 6 通道多模态 CPT 示例（基础回放与图文对照） | [build_mm_wiki_cpt.py](../build_mm_wiki_cpt.py) |
| [v1_v4_legacy_sample.jsonl](v1_v4_legacy_sample.jsonl) | V1～V4 历史数据接入样例 | [refactor_v1_v4_pipeline.py](../refactor_v1_v4_pipeline.py) |
| [qa_kb_raw_example.jsonl](qa_kb_raw_example.jsonl) | QA→CoT v0.3 RAG 引用模式兼容示例 | [gen_cot_from_qa.py](../gen_cot_from_qa.py) `--cot-mode rag_citation` |
| [style_sft_example.jsonl](style_sft_example.jsonl) | 严谨详尽风格 SFT 对话样例（并入统一 SFT 池） | [run_flash_pipeline.sh](../run_flash_pipeline.sh) |
| [source_documents_example.jsonl](source_documents_example.jsonl) · [qa_candidate_example.jsonl](qa_candidate_example.jsonl) · [sample_provenance_example.jsonl](sample_provenance_example.jsonl) | 领域原文登记、候选与溯源样例 | [`../../pro/montessori/`](../../pro/montessori/README.md) |

## 2. 独立原生配置模板（`configs/`）

| 文件 | 对应阶段 |
|---|---|
| [cpt_27b_mm_rag_wsd_megatron.json](configs/cpt_27b_mm_rag_wsd_megatron.json) | B300×8 `megatron pt` 27B WSD CPT 参考配置 |
| [cpt_27b_mm_rag_swift_lora.json](configs/cpt_27b_mm_rag_swift_lora.json) | `swift pt` 27B BF16 LoRA CPT 参考配置 |
| [sft_27b_unified_v1_v4_qa_b300.json](configs/sft_27b_unified_v1_v4_qa_b300.json) | 统一双模 SFT 参考配置 |

## 3. 早期概念示意

| 文件 | 说明 |
|---|---|
| [cpt_example.jsonl](cpt_example.jsonl) · [recipe_example.json](recipe_example.json) | 早期纯文本示意与跨阶段概念配比，不可直接作为全量输入 |
