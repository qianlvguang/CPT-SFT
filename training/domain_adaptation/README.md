# 领域 CPT、QA 合成与风格 SFT 的方案文件

此目录保存新增工作线的建议配置、提示词和自编示例。**尚未实现批量语料处理、teacher 调用或训练启动器**，不将示例当成可用训练集。

| 内容 | 文件 |
|---|---|
| 总流程与验证边界 | [方案总览](../../docs/training_plans/README.md) |
| 单条建议配方 | [recipe_example.json](recipe_example.json) |
| 进阶训练与管理设计 | [27B QAT/QLoRA、WSD、新 Megatron 及原生配置](../../docs/training_plans/qwen35_27b_advanced_training_and_management.md) |
| 原文登记示例 | [source_documents_example.jsonl](examples/source_documents_example.jsonl) |
| SWIFT PT 格式示例 | [cpt_example.jsonl](examples/cpt_example.jsonl) |
| QA 中间数据 | [qa_candidate_example.jsonl](examples/qa_candidate_example.jsonl) |
| 风格 SFT 格式示例 | [style_sft_example.jsonl](examples/style_sft_example.jsonl) |
| 来源映射示例 | [sample_provenance_example.jsonl](examples/sample_provenance_example.jsonl) |
| 生成与验证提示词 | [QA 生成](prompts/qa_generate.md)、[QA 验真](prompts/qa_verify.md)、[风格改写](prompts/style_rewrite.md) |
| Montessori-Instruct 合成管线 | [montessori/README.md](montessori/README.md)（探索采样 + 单步 Loss 差分探针 + Teacher DPO 对齐） |

示例正文是“教学合成规则”，并非真实法规；示例中没有伪造真实法律出处。JSONL 每行是一条记录，训练文本与审计元数据分别存放。正式执行前需补充真实来源、冻结数据 manifest、核对精确软件版本，并完成数据与 label 预检。
