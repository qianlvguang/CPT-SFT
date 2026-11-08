# S1 / S1-L / S2 完整训练日志（含 S2-L 计划与 S3 数据现状）

> 2026-09-28 编制。信息源：三个 run 的 `args.json`（实际发射参数 ground truth）、`logging.jsonl`（训练日志）、
> `runs/*.tfevents.*`（TensorBoard）、`mergekit_config.yml`（合并配置）、本地/远端训练脚本、
> `data-analysis/2026-09-28_s1_ckpt_task_trend/BIG_COMPARE.md`（评测结果）。
> 用途：提交给 S3 生成智能体作为完整上下文。

---

## 〇、文件路径速查表

| 资产 | 路径 |
|---|---|
| 底座模型 | `/data/autovision-cfs/autovision-notebook/algo4/models/Qwen3.5-27B` |
| S1 run 目录 | `/data/autovision-cfs/autovision-notebook/algo4/tpz/ft/sft-qwen3_5-27b-vis-double` |
| S1-L run 目录 | `/data/autovision-cfs/autovision-notebook/algo4/tpz/ft/sft-qwen3_5-27b-vis-double-L` |
| S2 run 目录 | `/data/autovision-cfs/autovision-notebook/algo4/tpz/ft/sft-qwen3_5-27b-vis-tripple` |
| S1 启动脚本 | `/data/autovision-cfs/autovision-notebook/sft/training/qwen3_5_27b/train_sft_qwen3_5_27b_megatron_lora_h200_vis_double.sh`（28,243 B，md5 `3d54df5aa61f6375e4d67d90a739a609`） |
| S1-L 启动脚本 | `.../train_sft_qwen3_5_27b_megatron_lora_h200_vis_double_L.sh`（32,371 B，md5 `70f1d6c3e2edfb39f30c14f243fc7ffd`） |
| S2/S2-L 脚本 | `.../train_sft_qwen3_5_27b_megatron_lora_h200_vis_tripple.sh`（27,884 B，md5 `6cda9c0e1b910b0f772f0d54ffb6146d`；**已被改为 S2-L 新版**，S2 旧版发射内容以 args.json 为准） |
| 训练数据目录 | `/data/autovision-cfs/autovision-notebook/algo4/tpz/test`（下称 TEST_DIR） |
| 评测集 | `TEST_DIR/test_oms.parquet`（全量 9450）+ eval_v3 5658 子集 |
| 评测结果 | 本地 `data-analysis/2026-09-28_s1_ckpt_task_trend/BIG_COMPARE.md` + `oms_new_big_compare.json`（23 run）+ `subset5658_acc.json`（13 run）+ `s2_acc.json` |
| 评测 rollouts | `/data/autovision-cfs/autovision-notebook/package/verl/output/oms-new/double-vis-s{1,s1-L,s2}-*-sglang` |
| 插件（自生成） | `.../sft/training/qwen3_5_27b/_vis_double_plugin.py`（4,537 B）/ `_vis_double_L_plugin.py`（5,762 B）/ `_vis_tripple_plugin.py`（3,185 B） |
| swift 数据缓存 | `MODELSCOPE_CACHE_HOME=.../logos-remote/sft/cache`（钉 NFS，pod 重建不丢） |

**TensorBoard 事件文件（loss 曲线，共 4 个）**：

| run | 文件 | 大小 |
|---|---|---|
| S1 | `ft/sft-qwen3_5-27b-vis-double/runs/events.out.tfevents.1790159604.udf-pod-7933-0-60ec51eb71c2e47f.461.0` | 607,638 B |
| S1-L 第一段 | `ft/sft-qwen3_5-27b-vis-double-L/runs/events.out.tfevents.1790160564.udf-pod-7936-0-ecc7491e24e66fe1.461.0` | 407,116 B |
| S1-L 第二段 | `ft/sft-qwen3_5-27b-vis-double-L/runs/events.out.tfevents.1790254331.udf-pod-7936-0-fa2ecd59eb2195b1.461.0` | 607,624 B |
| S2 | `ft/sft-qwen3_5-27b-vis-tripple/runs/events.out.tfevents.1790342371.udf-pod-7937-0-b5d1630feea1a702.459.0` | 437,536 B |

另有 `logos-remote/tensorboard_log/Qwen3_5-35B-A3B-SFT/`（3 个 88-1638 B 小文件）——那是 2026-09-16/17 **A3B LoRA 冒烟实验**的遗留，与 S1 系**无关**。每个 run 还自带 swift 渲染的 PNG 曲线（`images/train_loss.png`、`train_loss_V1..V5.png`、`train_grad_norm.png`、`train_learning_rate.png`、`train_memory(GiB).png`、`train_train_speed(s_it).png`）和 `logging.jsonl`（逐 5-iters JSON 日志，loss 曲线数据源）。

**本地副本（2026-09-28 下载，md5 与远端逐文件对账一致）**：
- `data-analysis/2026-09-28_s1_ckpt_task_trend/tb/`：`tb_s1_events.tfevents` / `tb_s1l_events_part1.tfevents` / `tb_s1l_events_part2.tfevents` / `tb_s2_events.tfevents`（EventAccumulator 验证可解析：各 10 tags，train/loss 点数 1053/681/1053/737，last_step 5260/3400/5260/3680）
- `data-analysis/2026-09-28_s1_ckpt_task_trend/scripts/`：三个训练脚本（`*_vis_double.sh` md5 `3d54df5a...` / `*_vis_double_L.sh` md5 `70f1d6c3...` / `*_vis_tripple.sh` md5 `6cda9c0e...`——注意 tripple 为 S2-L 新版，S2 旧版发射参数以 args.json 为准）
- 传输方式：远端 tar.gz（576,998 B，md5 `74a38b3e0142310d944aebb15e888218`）→ base64 13 chunks → cell 输出回传 → 本地拼装，chunk/targz/文件三层 md5 全校验通过

---

## 一、家族谱系与训练环境

```
Qwen3.5-27B（裸底座，dense VLM）
├── S1  (vis-double)   —— 860k mini 数据；ViT LoRA r16 + merger 全参 + LLM LoRA r64；2ep 5263 iters
│   └── S2 (vis-tripple) —— 从 checkpoint-5263-merged 续训；600k medium；冻结视觉栈；仅 LLM LoRA；2ep 3683 iters
└── S1-L (vis-double-L) —— S1 的对照臂：同底座同数据，merger 挂 LoRA（非全参）；2ep 5263 iters
    └── S2-L (vis-tripple-L) —— 计划中：从 S1-L checkpoint-4608-merged 续训；新改版 tripple 脚本已就位
```

- **框架**：ms-swift 4.3.2 + Megatron-Core 0.18.0（mcore_bridge 1.5.2），LoRA 走 `--tuner_type lora` + `merge_lora=true`
- **venv**：`tpz_xayah`（torch 2.13.0 + peft 0.19.1；tripple 脚本 `source tpz_xayah/bin/activate`）
- **硬件**：H200 ×8（141 GB/卡）。并行：TP=4 / PP=1 / CP=1 / EP=1（27B dense）/ DP=2，sequence_parallel=true
- **显存**：S1 115.54 GiB/卡（merger 全参+ViT LoRA 最重）；S1-L 115.18；S2 112.21（视觉栈冻结最轻）
- **pod**：S1 = udf-pod-7933-0；S1-L = udf-pod-7936-0；S2 = udf-pod-7937-0（S1 与 S1-L 是两台机器**并行**发射，时间差仅 16 分钟）
- **通道约定**：GBS=32 计的是 **packed windows**（MAX_LENGTH=8192 打包后），不是 raw samples

---

## 二、S1（vis-double）—— 视觉感知阶段

**目标**：module-heterogeneous adapters——感知/域适应打底：ViT 挂小 LoRA、aligner(merger) 全参微调、LLM 挂大 LoRA。

### 2.1 训练数据（5×mini，共 860,000 rows）

| 文件（TEST_DIR/） | rows | bytes | 说明 |
|---|---|---|---|
| `train_caption_ultra_x_v1mini.parquet` | 360,000 | 197,796,354 | 7 cols |
| `train_caption_ultra_x_v2mini.parquet` | 200,000 | 105,400,420 | 7 cols |
| `train_caption_ultra_x_v3mini.parquet` | 120,000 | 93,253,879 | 7 cols |
| `train_caption_ultra_x_v4mini.parquet` | 120,000 | 71,257,872 | 7 cols |
| `train_caption_ultra_x_v5mini.parquet` | 60,000 | 149,708,873 | **6 cols**；md5 `20537b5c5b1a843495ca398b8b2c9d62`；governance QA，fenced-json {"res","reason"} by design，194 条纯文本无图行是有意的；variant=5 列 2026-09-23 注入 |

packed 后 84,217 windows，8104.29±123.56 tok/window（min 556 / max 8192）。

### 2.2 args.json 实际参数（16,599 B，ground truth）

| 参数 | 值 |
|---|---|
| model | `algo4/models/Qwen3.5-27B`（裸底座） |
| tuner / target_modules | lora / `['all-linear']` |
| LLM LoRA | rank 64 / alpha 128 / dropout 0.05，lr **3e-5**（peak）→ min 3e-6（floor），cosine，warmup 3% |
| ViT LoRA | plugin rank_pattern `r'visual\..*'` → **r16/a32**，vit_lr **1e-5** |
| aligner（merger） | **全参**（`freeze_aligner=True` + `modules_to_save=['visual.visual.merger']`），aligner_lr **5e-6** |
| freeze_vit / freeze_llm | False / False（ViT 挂 LoRA、LLM 挂 LoRA） |
| epochs / iters | 2 epochs → **5263 iters**（GBS=32 packed windows） |
| save | save_steps 512，save_total_limit 6，merge_lora=true（每档落 adapter + merged 双份，merged ~51GB/档 12-shard + model_mtp.safetensors） |
| 其他 | seed 42，packing=true（内部强制 padding_free=true），recompute selective/uniform/1 层，flash attn，weight_decay 0.01，clip 1.0，MAX_LENGTH 8192，enable_channel_loss=True（V1..V5 仅记录，不影响 loss/吞吐） |
| external_plugins | `_vis_double_plugin.py`（patch A: variant→'V%d' channel；patch B: ViT rank_pattern 注入） |

### 2.3 训练过程

- 启动 ≈ 2026-09-23 18:33（北京时间，TB 时间戳 1790159604 推算）；总耗时 **23h38m52s**，末段 16.185 s/it，显存 115.54 GiB/卡
- 首行 loss 1.1887（V1 0.9368 / V2 2.1538 / V3 1.1852 / V4 2.7357 / V5 0.7954），lr 1.9e-7（warmup 起点）
- 末段 loss ≈ **0.6065**（V1 0.4517 / V2 1.1265 / V3 0.5616 / V4 1.3887 / V5 0.5520），lr 已到 floor 3e-6
- logging.jsonl 1054 行（每 5 iters 一条），348,510 B

### 2.4 产出

- **checkpoint-3072 / 3584 / 4096 / 4608 / 5120 / 5263**（各含 adapter ~1GB + `-merged` 完整模型 12-shard；3072/4608/5263 的 merged 含 model_mtp.safetensors）
- **checkpoint-1536 评测时存在，现已被 save_total_limit=6 轮替磨盘吃掉**（5658 acc 0.8427 有评测记录，权重已不在盘上）
- merge-double/：**mergekit Model Stock** 合并 3072+4608+5263 三档 merged（base=Qwen3.5-27B，dtype bfloat16，config 见 `merge-double/mergekit_config.yml`，另有大量注释掉的历史候选模型路径）

### 2.5 评测（oms 9450 全量 × 108 task + eval_v3 5658 子集双口径）

| ckpt | 5658 子集 acc | 全量 acc |
|---|---|---|
| S1-1536（已轮替） | 0.8427 | — |
| **S1-3072（S1 选定）** | **0.8457** | 0.8219 |
| S1-4608 | 0.8443 | — |
| S1-5263（S2 的父档） | 0.8434 | 0.8238 |

S1 系全量区间 0.8206-0.8222；F1(rollout0) S1-3072 = 0.7926。

---

## 三、S1-L（vis-double-L）—— aligner-LoRA 对照臂

**与 S1 的全部差异**（其余参数逐项相同，同底座、同 860k mini 数据、同 5263 iters）：

| 参数 | S1 | S1-L |
|---|---|---|
| freeze_aligner | True（排除 merger 出 LoRA 目标） | **False（merger 挂 LoRA）** |
| modules_to_save | `['visual.visual.merger']`（全参） | **`[]`（空，无全参模块）** |
| aligner_lr | 5e-6（全参 LR） | **1e-5（LoRA LR）** |
| external_plugins | _vis_double_plugin.py | _vis_double_L_plugin.py |

设计意图：检验 merger 全参 vs LoRA 的差异（S1 的 merger fp32 optimizer ~3GB/卡 在 S1-L 消失，其余静态显存几乎不变）。**注意：S1-L 与 S1 数据完全相同**（旧结论「S1-L 用 medium 配比」有误，args.json 铁证两者都是 5×mini；medium 是 S2 才用的）。

### 3.1 训练过程（一次中断 + 完全重跑）

- 与 S1 并行发射（TB1 时间戳 1790160564，晚 S1 960 s，pod-7936-0 另一台机器）
- **第一段在 ~iter 3405 中断**（TB1 407,116 B ≈ 3521 iters 的 events；logging 第一段 681 行 × 5 = 3405 iters——行数/字节对账推断）
- **完全重跑**（非 resume）：TB2 时间戳 1790254331（09-24 20:51），607,624 B ≈ 完整 5263 iters；logging.jsonl 共 1735 行 = 681（第一段）+ 1054（完整第二段）✓ 对账闭环
- 第二段有效耗时 **23h31m42s**（16.103 s/it，与 S1 同速），显存 115.18 GiB/卡；末段 loss ≈ 0.6067（V1 0.4519 / V2 1.1266 / V3 0.5620 / V4 1.3887 / V5 0.5494）
- 完成时间 ≈ 09-25 20:22；S2 于 09-25 21:19 发射（衔接紧凑）

### 3.2 产出与评测

- checkpoint-3072 / 3584 / 4096 / 4608 / 5120 / 5263（同 S1 结构）+ merge-double-L/（Model Stock 3072+4608+5263）

| ckpt | 5658 子集 acc | 全量 acc |
|---|---|---|
| S1-L-3072 | 0.8512 | — |
| **S1-L-4608（S1-L 选定，S2-L 的入口）** | **0.852775** | 0.8290 |
| S1-L-5263 | 0.8499 | — |

S1-L 系全量 0.8262-0.8290；F1 S1-L-4608 = 0.7975。S1-L 同档全面压制 S1 +0.43~+0.71pp（全量），misleading 簇 0.650→0.746 的修复是 S1-L 独有（merger LoRA 路线的贡献）。

---

## 四、S2（vis-tripple）—— 知识阶段（基于 S1-5263-merged）

**目标**：stage-2 知识 SFT——冻结整个视觉栈（ViT + aligner），只训一个全新 LLM LoRA。

### 4.1 关键设定

- **model = `ft/sft-qwen3_5-27b-vis-double/checkpoint-5263-merged`**（args.json 铁证；父子关系 S1-5263 → S2）
- fresh LoRA 进入（RESUME=0 + `--finetune true` 新建 LoRA/optimizer/scheduler）；**禁止用 S1 脚本的 RESUME=1 进 stage 2**（会恢复 S1 的 iteration/optimizer/RNG 状态并保留 S1 LoRA）
- freeze_vit=True / freeze_aligner=True / modules_to_save=[]（`${MODULES_TO_SAVE-}` 无冒号写法防空串回填）/ 无 vit_lr/aligner_lr
- patch B（ViT rank_pattern）移除——视觉栈冻结后全局 r64/a128 只落在 LLM
- LLM LoRA r64/a128/dropout 0.05，lr **2e-5** → min 2e-6，cosine，warmup 3%

### 4.2 训练数据（600k medium mix，S1-unused pools）

| 文件（TEST_DIR/） | rows | bytes |
|---|---|---|
| `train_caption_ultra_x_v1medium.parquet` | 120,000 | 92,568,736 |
| `train_caption_ultra_x_v2medium.parquet` | 90,000 | 65,890,835 |
| `train_caption_ultra_x_v3medium.parquet` | 150,000 | 151,868,590 |
| `train_caption_ultra_x_v4medium.parquet` | 180,000 | 76,085,164 |
| `train_rough_templete.parquet` | 60,000 | 139,931,396 |

合计 600,000 rows ✓（packed 58,936 windows，8114.23±123.92 tok/window，min 235 / max 8192）。
注意：V5 通道实际用的文件是 **train_rough_templete.parquet**（60k，md5 `1c4cb74286133e4dddaeb314566a3973`，6 cols），不是 v5mini；external_plan 的配比是 V1 120k / V2 90k / V3 150k / V4 180k / V5 60k。V5 属 rule binary judgment 的 S1-seen REPLAY 性质——**禁止从 V5 切验证集**。

### 4.3 训练过程与产出

- 启动 ≈ 2026-09-25 21:19（TB 1790342371，pod-7937-0）；总耗时 **13h33m51s**（13.270 s/it，比 S1 快——视觉栈冻结），显存 112.21 GiB/卡
- 首行 loss 0.9432（V1 0.4642 / V2 1.1652 / V3 0.5962 / V4 1.5725 / V5 0.6051）——起点明显低于 S1 首行 1.1887（继承父档的知识）
- 末段 loss ≈ **0.7090**（V1 0.3875 / V2 1.1631 / V3 0.5456 / V4 1.3093 / V5 0.5968），lr floor 2e-6
- save_steps 612（= iters_per_epoch 1836 / 3），save_total_limit 6，2 epochs → **3683 iters**
- logging.jsonl 738 行，243,196 B
- **checkpoint-1224 / 1836 / 2448 / 3060 / 3672 / 3683**（前 5 个是 612 间隔档，3683 是终点档；3672=612×6 定期档与 3683 终点档并存）
- merge-tripple/：Model Stock 合并 **2448+3060 两档**（base=Qwen3.5-27B，bfloat16）

### 4.4 评测（五档双口径）

| ckpt | 全量 acc | 5658 子集 acc |
|---|---|---|
| S2-1224 | 0.823810 | 0.8480 |
| S2-1836 | 0.830370 | 0.8507 |
| S2-2448 | 0.830794（全量峰值） | 0.8519 |
| **S2-3060（S2 选定）** | 0.830688 | **0.854366（子集峰值）** |
| S2-3683 | 0.829418 | 0.8492 |

F1(rollout0)：S2-3060 = **0.8044**（三系最高）。1836 步即近全量饱和。

**S2 vs 父档 S1-5263 继承分析**（S2 的正确比法）：

- corr(S2-x, S1-5263) = 0.940 / 0.907 / 0.920 / 0.905 / 0.861（1224→3683，越训越偏）——「S1-5263 + 局部重排」的父子关系成立
- 总量增益：S2-3060 vs 父档 全量 +0.85pp / 5658 +1.10pp
- 继承增益 top（全量）：misleading_subject +13.2pp、IP_berserk +9.4、Flag inconsistent +8.9、Flag&Emblem +7.9、FHSA +7.7、sunglass_label +6.8
- 继承退化 top：inconsistent quantity pro -30.0pp、PillOrganizer -19.6、ip-Rilakkuma -15.2、PPE -13.3、Object recognition -9.3
- 簇级：text-compliance 0.744→0.788-0.795（+4.6pp 系内最高）、flag-hard 0.524→0.588-0.628（+6.4~10.4pp，逼近 libra 0.654）、misleading/apparel 持平、vision-strong -0.5~-2.2pp（小让渡换前两簇）

---

## 五、三系横评结论（master 钦定口径：ckpt 选档以 5658 子集为准）

| 指标 | S1-3072 | S1-L-4608 | **S2-3060** | libra（锚点参照） |
|---|---|---|---|---|
| 5658 acc | 0.8457 | 0.852775 | **0.854366** | 0.884942 |
| 全量 acc | 0.8219 | 0.8290 | **0.8307** | 0.8638 |
| F1 | 0.7926 | 0.7975 | **0.8044** | — |
| 严格赢 libra（5658 task 数） | 24/108 | 33/108 | **33/108** | — |

- **最终推荐 ckpt = S2-3060**（5658 + F1 双料冠军；12-ckpt 5658 排名：s2_3060 #1 / s1l_4608 #2 / s2_2448 #3 / s1l_3072 #4 / s2_1836 #5）
- 三系簇互补：S1 收 apparel+vision-strong；S1-L 收 misleading+vision-strong（0.895）；S2 收 text-compliance+flag-hard
- 对 libra 遗留缺口（5658）：misleading 簇 -19pp（S2 未解，S1-L 收一半）、apparel -12pp（全系未解）、text-compliance -9.5pp（S2 收一半）
- 评测方法：sglang 推理 + `vlm_rollout_filter` 产出（S1/S1-L 读 summary per_task_acc，S2/libra 由 parquet 现算）；转录校验 micro 与 summary.json mean_pass_rate 全一致 + 双口径交叉验证 delta<6e-7
- 详细分析+图：本地 `BIG_COMPARE.md`（三系版，fig10/11/12）；旧 23-run 大横评归档 `archive_23run/BIG_COMPARE_23run.md`

---

## 六、S2-L 计划（vis-triple-L，新改版脚本已就位未发射）

`train_sft_qwen3_5_27b_megatron_lora_h200_vis_tripple.sh` 已改版（即上文速查表中的 md5 `6cda9c0e...` 版本）：

- **MODEL_PATH 默认 = `ft/sft-qwen3_5-27b-vis-double-L/checkpoint-4608-merged`**（S1-L-4608——S1-L 系 5658 最佳档）
- **OUTPUT_DIR = `ft/sft-qwen3_5-27b-vis-tripple-L`**
- 其余超参沿用 S2 配方：冻结视觉栈、fresh LLM LoRA r64/a128、lr 2e-5/floor 2e-6/warmup 3%/cosine、2 epochs、GBS 32 packed、SAVE_STEPS=612（SMOKE 后按实际 iters_per_epoch/3 校正）、save_total_limit 6、MERGE_LORA=true、RESUME=0 fresh 进入
- 数据：600k 配比（脚本注释指向 `TEST_DIR/train_caption_s2_manifest.json`；**注意 TEST_DIR 当前没有 `train_caption_s2_x_*` 文件**——09-24 女仆间交接提过 s2_x_v{1,2,5}（一号）/v3 15万（三号）/v4 18万（二号）接口，实际落位文件待确认；S2 旧版实际使用的是 medium×4 + rough_templete）
- 预计 iters_per_epoch ~1836（600k @ GBS32 packed ~10.2 rows/window），2ep ~3672+终点档
- 验证清单（launch log grep）：`[vis_tripple_plugin] variant->channel patch installed`、无 `rank_pattern`、trainable 仅 LLM LoRA、`loss_V1..V5`、`MODULES_TO_SAVE/VIT_LR/ALIGNER_LR` 全空

---

## 七、S3 final 训练数据现状（二号·小研 2026-09-28 20:19 交接）

四件 final parquet 已全部落成于 `TEST_DIR/`（schema 与各自 medium 版一致；行级 `(sample_id, cate_id, image)` 唯一；与各自 mini/medium 零交集）：

| 文件 | rows | bytes | md5 |
|---|---|---|---|
| `train_caption_ultra_x_v1final.parquet` | 40,113 | 31,811,074 | — |
| `train_caption_ultra_x_v2final.parquet` | 230,100 | 161,706,623 | — |
| `train_caption_ultra_x_v3final.parquet` | 423,370 | 412,576,344 | — |
| `train_caption_ultra_x_v4final.parquet` | 403,947 | 239,057,530 | `cf3edbe28c76993ec22a2872a6da3d8a`（variant=4，enable_thinking=False） |

- **v4final gate 对账**：ok 724,936 − needs_audit 26,070 − parse_fail 1,093 − mini/medium 已用 293,826 = 403,947 ✓；v4 推理覆盖率 99.999%
- **v4 补推**：todo 路 21,573 行（gemini-3.7-flash，ok 21,570）+ refused 路 1,426 行（换 gemini-3-pro，ok 1,424），均已 append 进 `TEST_DIR/v4_caption_all.jsonl`（727,791 行，4,680,218,143 B；raw 源 `train_caption_ultra_x_v4raw.jsonl` 366,077,007 B）
- **血泪教训（硬规则）**：v4 同一 (task,sid) 下可挂多张不同图（48,248 个多行 key）——**行级键 `(sample_id, cate_id, image)` 才唯一，禁止按 (task,sid) 去重**（第一版 176,448 就是这么错的，会毁一半数据）
- 构建产物：`v123final_build_report.json` / `v4final_build_report.json`（logos-remote）；构建/校验脚本 `build_final_v123.py`、`build_final_v4*.py`、`verify_final_*.py`（待归档批示）
- 数据量参考（每行 assistant token 密度，09-24 接口留言）：s2 接口 V1 632.2 tok/row / V2 232.7 / V5 83.1

---

## 八、血泪教训与硬规则（S3 智能体必读）

1. **轮替磨盘**：`save_total_limit=6` 在每次新存档后 ~2 分钟内吃掉最旧一对 ckpt——需要评测/救援的档**存档瞬间就要拷走**（2560 讣告：147G+66G 存活仅 ~2 分钟）。S1-1536 即被吃掉（有评测无权重）。
2. **SMOKE 必须省略 `--num_train_epochs`**：`megatron_args.py:934` 只要 num_train_epochs 非 None 就覆盖 train_iters。
3. **stage 2 进入方式**：必须 fresh（MODEL_PATH=父档 merged + RESUME=0 + `--finetune true`），禁止 RESUME=1。
4. **V5/rough_templete 是 S1-seen REPLAY**：禁切验证集；in-run val 全程 OFF（split_dataset_ratio=0）。
5. **空串参数陷阱**：`MODULES_TO_SAVE=${MODULES_TO_SAVE-}`（无冒号）——`${VAR:-default}` 会把显式空串回填成 default。
6. **首启动 encode 慢（小时级）**：`MODELSCOPE_CACHE_HOME` 已钉 NFS（`logos-remote/sft/cache`），重启动/重建 pod 免重编码。
7. **variant→channel 列差异**：v1-v4 mini/medium 是 7 cols，v5mini/rough_templete 是 6 cols——plugin 对无 variant 列的文件静默跳过 channel 翻译（不影响训练 loss，只影响 V5 channel 指标归组）。
8. **GBS 计 packed windows**：860k rows → 84,217 windows（S1）；600k → 58,936（S2）。iters_per_epoch 估算用 ~10.2 rows/window。
9. **NFS 大坑**（AGENT.md §5）：配额撞顶进程 D-state 卡死不报 ENOSPC；重建文件写新临时文件 + os.replace；多进程写同一输出绝对禁止。
10. **swift resume 的 elapsed_time 是逻辑累计**（train_speed×iters），不是 wall clock；判断中断史要看 TB 事件文件个数 + logging.jsonl 行数对账。
11. **训练参数以 args.json 为准**：脚本会被后续改版覆盖（本例 tripple.sh 已改 S2-L 版），args.json 是发射时刻的快照。
12. **远端 root 的 `vis_double.sh`（27,908 B）是旧版部署副本**，权威版本在 `sft/training/qwen3_5_27b/`（28,243 B）；本地 `sft/training/qwen3_5_27b/` 副本与远端 md5 不同（28,243 B / f2f7a90f... 与 06240d0f...），同步时注意。

---

## 附：训练数据全量清单（TEST_DIR 相关，2026-09-28 盘点）

| 文件 | rows | bytes | 用途 |
|---|---|---|---|
| train_caption_ultra_x_v1mini..v5mini | 360k/200k/120k/120k/60k | 197.8M/105.4M/93.3M/71.3M/149.7M | S1 + S1-L |
| train_caption_ultra_x_v1medium..v4medium | 120k/90k/150k/180k | 92.6M/65.9M/151.9M/76.1M | S2 |
| train_rough_templete | 60,000 | 139,931,396 | S2 V5 |
| train_mixed_templete | 60,186 | 140,024,434 | （5 cols，未用于 S 系） |
| train_caption_ultra_x_v1/v2/v3final | 40,113/230,100/423,370 | 31.8M/161.7M/412.6M | S3 候选 |
| train_caption_ultra_x_v4final | 403,947 | 239,057,530 | S3 候选 |
| train_caption_ultra_x_v1/v2/v3.parquet（全量版） | — | 292.1M/258.2M/499.0M | full-scale 池（S1 注释：留给后续 stage） |
| train_caption_ultra_x_v4raw.jsonl | — | 366,077,007 | v4 raw 源 |
| v4_caption_all.jsonl | 727,791 行 | 4,680,218,143 | v4 caption 总池 |
