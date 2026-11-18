# Phase C：Pro 错例簇闭环演进方案

本目录归档并拓展 Flash 基线验收之后的 **Pro 长期滚动迭代、Montessori 影响对齐与高阶训练/量化管理方案**：

- [Pro 错例簇闭环演进与跨仓蒸馏方案](pro_closed_loop_plan.md)：四类错例归因（固定权重先验索引/路由）、50/30/20 增量 CPT、`sft_overlap=0` 隔离与 Montessori / OPD / Datawheel 闭环。
- [27B 进阶训练与管理（QAT、QLoRA、WSD 与堆栈调研）](qwen35_27b_advanced_training_and_management.md)：NF4 QLoRA、NVFP4 QAT/QAD 精度恢复及堆栈核对记录。

可执行代码入口位于 [`training/pro/`](../../../training/pro/README.md)。
