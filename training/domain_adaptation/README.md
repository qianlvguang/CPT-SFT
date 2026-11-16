# B300 统一主线：多模态 CPT、V1～V4 融合与 Montessori QA-SFT

本目录是当前 **B300 27B 多模态 Wiki+RAG CPT → V1～V4 重平衡 + 引用 QA SFT** 的统一工程与方案落盘入口（旧 H200 `4.3.2` 独立三阶段 SFT 已归档至 [`../qwen35_27b/`](../qwen35_27b/README.md)）。

## 目录组件

| 类别 | 文件 | 说明 |
|---|---|---|
| **方案文档** | [B300 多模态 Wiki+RAG CPT](../../docs/training_plans/b300_multimodal_wiki_rag_cpt_plan.md)、[V1～V4 统一重构方案](../../docs/training_plans/unified_v1_v4_cpt_qa_pipeline.md) | 可行性评估、6 通道配比、WSD 两阶段课程与 V1～V4 分流设计 |
| **数据构建与分流** | [refactor_v1_v4_pipeline.py](refactor_v1_v4_pipeline.py) | 历史 V1～V4 三元键去重与分流（V1/V3/V4 进 CPT 回放/RAFT，Token 重平衡进统一 SFT，抽取 Qwen-Image-Edit 难例） |
| | [build_mm_wiki_cpt.py](build_mm_wiki_cpt.py) | 6 通道多模态 CPT 分片构建器（`<image>` 前置因果校验、文档族零泄漏、WSD Stage 2 顽固长尾重采样与 SHA256 清单） |
| **训练与编排入口** | [train_qwen35_27b_cpt_b300.sh](train_qwen35_27b_cpt_b300.sh) | B300 27B 多模态 WSD/Cosine CPT 启动器（`megatron pt` / `swift pt`、`lora` / `full`、`SMOKE=1`、`DRY_RUN=1`） |
| | [run_unified_b300_pipeline.sh](run_unified_b300_pipeline.sh) | 5 步统一端到端编排（V1～V4 分流 → 6 通道 CPT 构建 → WSD 两阶段 CPT → HF 导出 → 统一 QA+V1～V4 SFT） |
| **QA 合成与偏好子管线** | [montessori/README.md](montessori/README.md) | 基于 Montessori-Instruct 的探索采样、单步 Loss 差分探针、闭源 API 对比引导与 DPO/KTO/GRPO 导出 |
| **提示词模板** | [QA 生成](prompts/qa_generate.md)、[QA 验真](prompts/qa_verify.md)、[风格改写](prompts/style_rewrite.md) | 离线构造与核验带依据 QA 的提示词 |
| **示例与独立配置模板** | [examples/README.md](examples/README.md) | 统一归档主线联调 JSONL、独立参考 JSON 配置（`examples/configs/`）与早期 `recipe_example.json` / `cpt_example.jsonl` |

## 本地无 GPU 一键预演

```bash
# 从仓库根目录执行：完整跑通 Step 1~2 真实数据构造并打印 Step 3~5 可执行命令
DEMO=1 DRY_RUN=1 bash training/domain_adaptation/run_unified_b300_pipeline.sh
```

> 正式上线训练前仍须补充真实语料与 `Flash-Next` RAG 检索库、冻结数据 `manifest.json`、核对目标集群精确依赖版本，并以 `SMOKE=1` 完成 10 步显存与 `labels` 预检。
