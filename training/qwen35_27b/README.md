# V1～V4 唯一推荐训练路线

入口：[train_s1_s3_recommended.sh](train_s1_s3_recommended.sh)。部署时把它与 [Python 调度器](train_s1_s3_recommended.py)、[插件](recommended_pipeline_plugin.py) 放在同一目录。历史脚本与 sweep 仅作为实验记录保留；新入口不调用它们。

这是基于现有 S1/S1-L/S2 记录选择的推荐配方，尚未在新数据组合上实测，不能保证是所有可能配方中的最佳。

## 配方

| 阶段 | 完整数据 | epochs | 视觉栈 | LLM LoRA | LLM 学习率 |
|---|---|---:|---|---|---|
| S1 | V1～V4 mini，800,000 行 | 2 | ViT r16/a32；merger r32/a64，两者 lr=1e-5 | r64/a128 | 3e-5 → 3e-6 |
| S2 | V1～V4 medium，540,000 行 | 2 | 保留并冻结 S1 的 ViT/merger LoRA | 加载并继续训练 r64/a128 | 2e-5 → 2e-6 |
| S3 | V1～V4 final，1,097,530 行；另加 V1 回放 120,000 行 | 1 | 保留并冻结 S1 的 ViT/merger LoRA | 加载并继续训练 r64/a128 | 1e-5 → 1e-6 |

原始数据共 2,437,530 行；按实际文件行数训练，不截取 mini/medium/final。这里的“所有数据”指这 12 个已整理训练文件；raw、caption 总池、全量版属于上游/重复池，不再次拼接。V5 不参与。

S1 采用历史 S1-L 视觉 LoRA 路线。S2 保留历史的 2 epochs 上限，支持从至少完成一遍数据的存档中选档，避免只凭终点决定父模型。S3 final 的 V1 原始占比约 3.65%，而 V3/V4 合计约 75.38%；从 V1 mini+medium 的 480,000 行池中，以 seed=42 无放回均匀抽取 120,000 行回放，把 S3 中 V1 的行占比提升到约 13.15%。这是防遗忘的经验选择，不是已验证的最优比例；行占比也不等于监督 token 占比。

底座始终为同一份 Qwen3.5-27B；S2/S3 用 `--adapters` 加载上一阶段选定的累计 LoRA，`--finetune true` 重置 optimizer/调度器/iteration，不恢复训练状态。LLM LoRA 延续已有参数，不是重新随机初始化。这与历史的“合并后新建 fresh LoRA”不是同一种参数化：同一份 LLM adapter 的秩保持64，多次 fresh LoRA 的累计增量秩可能更高，效果必须重新评测。

中间所有存档均 `--merge_lora false`，只保存 adapter；最终产物也是底座 + 一份 S3 累计 adapter，可在真正部署需要完整模型时再合并一次。H200×8，TP4/DP2，GBS32 packed windows，长度8192，warmup3%，cosine，dropout0.05。S2/S3 冻结视觉 LoRA 时同时关闭其 dropout；S3 先完整训练一遍，控制进一步漂移。

**冻结视觉的实现不能只改 CLI。** 在 LoRA 模式下，`freeze_vit/aligner=true` 会阻止挂载相应 adapter，不能用于“保留旧视觉 LoRA 的前向但停止更新”。本路线三阶段都先构建相同的视觉异构 LoRA、加载旧权重，再由插件在 optimizer 创建前冻结视觉及全部 base 参数。日志中的 `visual_frozen=True` 和实际 trainable 参数是验收依据；启动参数中的 freeze 标志是挂载阶段的设置。

## 启动

以下命令在 `training/qwen35_27b/` 目录执行。

本地直接查看方案，不要求远端环境和数据：

```bash
bash train_s1_s3_recommended.sh plan
```

远端默认使用 `/data/autovision-cfs/autovision-notebook` 下的 `tpz_xayah` 环境、`algo4/models/Qwen3.5-27B` 底座和 `algo4/tpz/test` 数据目录。

```bash
bash train_s1_s3_recommended.sh preflight
bash train_s1_s3_recommended.sh smoke
nohup bash train_s1_s3_recommended.sh run > recommended_pipeline.log 2>&1 &
```

可以通过环境变量设置部署位置：`WORK_DIR`、`VENV_ACTIVATE`、`BASE_MODEL`、`TEST_DIR`、`OUTPUT_ROOT`。这些只改变资源位置，不开放训练配方 sweep。`SKIP_ACTIVATE=1` 用于已经激活环境的场景。

smoke 使用单独的 `_smoke` 输出目录，三阶段各 10 iterations，并省略 `num_train_epochs`。它使用真实完整数据，首次编码仍可能花较长时间。smoke 仅验证能否运行，不能确认训练效果；真实运行前检查日志中的 rank_pattern、实际 trainable 模块及 loss_V1～V4。三阶段都应挂载相同视觉异构 LoRA；S2/S3 日志必须有 `visual_frozen=True`，trainable 仅为 LLM LoRA，存档必须仍包含视觉 LoRA。

## 评测与选档

已有评测脚本可以用 `EVAL_SCRIPT` 接入；同一条训练路线，不产生不同超参分支。调度器串行调用：

```text
bash "$EVAL_SCRIPT" ADAPTER_DIR STAGE_NUMBER METRICS_JSON_PATH
```

底座路径由环境变量 `PIPELINE_BASE_MODEL` 传给评测脚本，调用推理时需要同时使用底座和 adapter，而不是把 adapter 目录当成完整模型。脚本必须用固定的 eval_v3 5658 口径完成推理并生成：

```json
{"dataset_id": "eval_v3_5658", "accuracy": 0.85, "f1": 0.80}
```

```bash
EVAL_SCRIPT=/path/to/eval_selected_checkpoint.sh bash train_s1_s3_recommended.sh run
```

S1/S2 只在已覆盖至少一遍数据的候选中按 accuracy 排序，再以 F1、较早步数打破平局。S3 只有 1 epoch，因此最终产物使用终点；中间档保留用于观察趋势。评测失败会停止衔接，不用缺失或混口径结果选档。调度器只能校验返回值，评测脚本需自行保证真实使用对应的固定数据集、解码配置和统计口径。

没有 `EVAL_SCRIPT` 也能完整运行，但各阶段使用最终 adapter 档，结果明确标记 `final_unscored`；这时输出是推荐配方训练的终点，不能称为“实测最佳 checkpoint”。未提供真实远端评测入口，因此交付没有假设或伪造一个评测实现。

## 数据与产物保护

- 启动前逐批检查 12 个文件的 schema、variant、行级重复以及同一 V 内 mini/medium/final 交集；行级键保留 sample_id、cate_id 和完整 images，不按二元键丢弃多图样本。不同 V 之间可能对应不同监督目标，不做跨 V 去重。V4 final 必须 enable_thinking=false。
- 开启 strict；插件不允许忽略超长错误，并检查预处理批次未丢行。新阶段签名使用独立的持久缓存，避免沿用历史过滤后的缓存。若编码超过8192或存在异常样本，停止并暴露问题，不默默删行或截断回答；因此真实完整数据是否都能在当前长度下编码，必须由远端 smoke 确认。相关默认行为核对了 [ms-swift 数据/模板参数文档](https://swift.readthedocs.io/en/v4.3/Instruction/Command-line-parameters.html)。
- SHA256 覆盖完整训练文件、回放文件、模型内容、调度器、插件和评测脚本；版本进入阶段签名。底座和父档校验会顺序读取全部权重，有额外 I/O。
- 只有训练进程成功退出、选档成功、adapter 配置及 safetensors 头/shape/dtype/数据区检查通过、A/B 矩阵完整且同时包含视觉与 LLM 增量，才原子发布 `complete.json`。存在配置文件或存档目录不代表完成。底座另行检查分片索引和全部权重。
- 同一个输出根目录只允许一个进程持锁运行；硬故障可能留下 `.pipeline.lock`，先核查 owner.json 对应任务确已停止，再处理残留锁。
- 未完成的阶段不会自动覆盖、移动或删除；更换 `OUTPUT_ROOT` 重新训练。已完整完成且指纹一致的阶段可以复用。该版本不保存优化器/RNG，不支持阶段内断点续训。
- 每约 1/4 epoch 保存一档，所有候选保留，不自动轮替。按历史 packing 密度估计，全流程约 20～23 份累计 adapter；历史 adapter 约1 GB/档，增量存档可能降到数十 GB，常规空间预检按3 GiB/档加额外余量计算。实际大小取决于导出权重的 dtype/模块，须通过 smoke 确认。底座只保留已有的一份，不再每档导出约51 GB完整权重。另需考虑编码缓存、预检 SQLite 和 NFS 独立配额；常规磁盘空间检查不能代替配额检查。
- 不合并主要节省存储和导出 I/O。训练仍需驻留底座、激活和优化器；冻结视觉 adapter 会减少相应梯度/优化器状态，但不应预期仅关闭 merge 就大幅降低训练显存。
- 新输出中包括 data_manifest.json、回放数据及其来源记录、各阶段 launch_manifest.json/train.log/complete.json，最终 result.json 同时记录底座与选中的 S3 累计 adapter 路径。

环境严格要求历史的 ms-swift 4.3.2、Megatron-Core 0.18.0，并检查其余训练依赖存在。冻结/LoRA 参数语义核对了 [ms-swift 4.3 官方文档](https://swift.readthedocs.io/en/v4.3/Megatron-SWIFT/Command-line-parameters.html)。4.3.2 [训练器源码](https://github.com/modelscope/ms-swift/blob/v4.3.2/swift/megatron/trainers/base.py)明确只支持一份 adapter，并在 optimizer 创建前加载 adapter；因此采用单 adapter 累计训练和加载后冻结，而不是未经支持的多 adapter 叠加。新环境升级后应先适配 API，再更换输出根目录。

## 本地验证边界

验证项目包括 `bash -n`、三个新增 Python 文件的 `py_compile`、本地 plan、标准库测试及文档本地链接检查。测试覆盖不完整/截断权重拒绝、分片索引检查、原地换权重、至少一遍数据后的选档、smoke 参数、三阶段累计 adapter 衔接与完成结果复用、训练失败停止后续阶段，以及视觉 LoRA 挂载后冻结的逻辑。

调度测试使用小型模拟权重并替代训练/数据读取；本机没有 pyarrow、远端 parquet 或 H200，尚未执行真实 parquet 预检、回放生成、GPU smoke 和效果评测。这些必须在远端环境按上述命令验证；本地测试通过不代表真实训练已通过。
