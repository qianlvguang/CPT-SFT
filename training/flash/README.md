# Phase B：Flash 全量 WSD 线与 QA→CoT v0.3 (`training/flash/`)

在 Phase A Lite 增益确认后，承接 **全量模板覆盖、两阶段 WSD CPT（`bundle v2.3` 并入 `sch` 通道）、QA→CoT v0.3 思维链合成（`qwen3.8-flash` + 八道程序门）与三方对照统一 SFT**。

配套方案见 [`docs/training_plans/flash/README.md`](../../docs/training_plans/flash/README.md) 与 [`docs/training_plans/flash/qa_cot_plan.md`](../../docs/training_plans/flash/qa_cot_plan.md)。

## 核心组件

| 文件 | 职责 |
|---|---|
| [`build_mm_wiki_cpt.py`](build_mm_wiki_cpt.py) | 全量 WSD 两阶段 CPT 构建器（兼容 6 通道 Wiki/RAG/Logo + `sch iso` `ch_ip_grounding`/`ch_ip_negative` + 16K 长度守卫与 `--max-row-repeats 3`） |
| [`gen_cot_from_qa.py`](gen_cot_from_qa.py) | QA→CoT v0.3 生成器与八道质检门（`synonsense->synthesis` 清洗、干扰项混排与 `[S_old]->[S_new]` 重映射、冷酷核验风格、`180/350/500` 硬下限、失败样本清空 `reasoning=""` 降级回直答桶、8:2 混配导出） |
| [`refactor_v1_v4_pipeline.py`](refactor_v1_v4_pipeline.py) | 历史 V1～V4 三元去重分流与三方 SFT 实验臂构造器（`ood_only` / `ood_plus_qa_direct` / `ood_plus_qa_cot`） |
| [`run_flash_pipeline.sh`](run_flash_pipeline.sh) | Phase B 端到端编排脚本 |
| [`prompts/`](prompts/qa_cot_bridge_v03.md) & [`examples/`](examples/README.md) | `qa_cot_bridge_v03.md`、QA 生成/验真/改写提示词与 B300 原生配置参考 |

## 本地无 GPU 端到端预演

```bash
DEMO=1 DRY_RUN=1 bash training/flash/run_flash_pipeline.sh
```
