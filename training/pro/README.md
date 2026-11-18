# Phase C：Pro 错例簇闭环演进与 Montessori 对齐 (`training/pro/`)

面向 Flash 验收基线之后的**长周期滚动迭代**，对应方案见 [`docs/training_plans/pro/pro_closed_loop_plan.md`](../../docs/training_plans/pro/pro_closed_loop_plan.md) 与 [`docs/training_plans/online_ip_template_cpt.md`](../../docs/training_plans/online_ip_template_cpt.md)。

## 核心组件

| 文件 / 目录 | 职责 |
|---|---|
| [`error_cluster_miner.py`](error_cluster_miner.py) | 线上/评测错例四类缺口归因（`missing_index_evidence` / `style_collision_false_positive` / `insufficient_forced_judgment` / `disposition_mapping_drift`）、固定权重索引补全门控、**50% 新核验缺口 + 30% 已验收模板 + 20% V 系列回放** 增量 CPT 构建与 `sft_overlap == 0` 模板族隔离 |
| [`montessori/`](montessori/README.md) | Montessori-Instruct 探索采样、单步梯度影响函数 $\mathcal{I}(x,y;\mathcal{D}_{\text{ref}})$、黑盒 API ICDPO 引导及 DPO/KTO/GRPO 多格式偏好导出 |
| [`run_pro_pipeline.sh`](run_pro_pipeline.sh) | Phase C 端到端编排（错例聚类 → B300 Pro 增量 CPT → Montessori 对齐） |
| [`examples/pro_error_clusters_sample.jsonl`](examples/pro_error_clusters_sample.jsonl) | 四类错例簇与隔离校验演示数据 |

## 本地无 GPU 端到端预演

```bash
DEMO=1 DRY_RUN=1 bash training/pro/run_pro_pipeline.sh
```
