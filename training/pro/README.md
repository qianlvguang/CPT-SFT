# Phase C：Pro 同源程序增强、Think-OFF 蒸馏与多轮演化 (`training/pro/`)

面向 Flash 验收基线之后的**同源质量跃升与长周期滚动迭代**，对应方案见 [`docs/training_plans/pro/pro_closed_loop_plan.md`](../../docs/training_plans/pro/pro_closed_loop_plan.md) 与 [`docs/training_plans/online_ip_template_cpt.md`](../../docs/training_plans/online_ip_template_cpt.md)。

## 核心组件

| 文件 / 目录 | 职责 |
|---|---|
| [`enhance_flash_to_pro.py`](enhance_flash_to_pro.py) | **基于与 Flash 同一份原始/Mid-Training 池的 4 项程序升级**：<br>1. 信息密度与长尾精炼（`curate_high_value_midtrain`）；<br>2. 视觉多图 ↔ CoT v0.4 跨模态锚点对齐审计（`alignment_audit.json`）；<br>3. 双模 Canary 监控 + WiSE-FT 权重插值（通用掉点 `>2.0pp` 触发 `0.8*CPT + 0.2*Base`）与 **Thinking-OFF 崩溃检测**；<br>4. 失败 CoT 复用为**学生 DPO/KTO 对**（非 Teacher DPO）与 `OPD` **Think-ON→OFF 上下文蒸馏**种子（`pro_opd_bsd_seeds.jsonl`） |
| [`error_cluster_miner.py`](error_cluster_miner.py) | 四类错例归因（固定权重先验索引/路由）、50/30/20 增量 CPT 与**多轮单调版本账本（`iteration_ledger.json`）**——支持 `--round-id` / `--parent-ckpt` / `--prev-ledger`，累计追踪已用 `template_family_id` / `doc_family_id` 并断言跨轮 `sft_overlap == 0` |
| [`montessori/`](montessori/README.md) | Montessori-Instruct 探索采样、单步梯度影响函数与 API ICDPO/OPD 对齐 |
| [`run_pro_pipeline.sh`](run_pro_pipeline.sh) | Phase C 端到端五步编排（同源接入 → 4 项程序增强 → `pro` Mid-Training → 多轮 `error_cluster_miner` + `pro_inc` → Montessori/OPD） |

## 本地无 GPU 端到端预演

```bash
DEMO=1 DRY_RUN=1 bash training/pro/run_pro_pipeline.sh
python3 training/pro/enhance_flash_to_pro.py --demo --output-dir /tmp/pro_enh_demo
```
