# CPT-27B 战役 Runbook（Lite 增益对照线）

> 2026-11-18 17:15 CST · 实测台账核对版
> 数字全部 pod 实测复核（manifest / md5 / cuobjdump，2026-11-18 17:0x 拉取），无凭记忆项。
> 配套文档：[QA→CoT 详案 v0.3](../flash/qa_cot_plan.md) · [三档总纲](../online_ip_template_cpt.md) · [本地 Lite 脚本说明](../../../training/lite/README.md)

---

## 0. 一页速览

| 项 | 值 |
|---|---|
| 目标 | Qwen3.5-27B 域适应 CPT（电商侵权 / IP 认知域） |
| 战场 | **AutoVision 8×B300**（`sm_103` Blackwell Ultra，268.6 GiB/卡）；pod 只做数据+发射链，禁训练 |
| 当前线 | **lite 增益对照实验**：`CPT(lite) + 外部 OOD SFT` vs `base + 同一套 OOD SFT`，双臂比增益（SFT 数据外部提供） |
| 训练数据 | `bundle_lite_v3_all·val0`：**12,085 行 / 9,549,679 tok / 18.21 步/epoch**（GBS32×16384） |
| 配方 | 2 epochs ≈ **36 步**，constant LR `1e-5` + warmup 6%，TP4 DP2 MBS4 GBS32 `flash` 16384 |
| 底座 | `algo4/models/Qwen3.5-27B` |
| 发射命令 | `bash /data/autovision-cfs/autovision-notebook/CPT-SFT/training/cpt_27b/run_cpt_27b_b300.sh lite --model /data/autovision-cfs/autovision-notebook/algo4/models/Qwen3.5-27B --launch` |
| 当前状态 | 环境两枚地雷全排：APEX fusion **已修**；TE `sm_103` **已修**（混搭移植，pod 卡 5 B300 smoke 全绿，见 §6.2，待重发确认） |

---

## 1. 阶段设计

```text
Phase A（当前）  lite 增益对照
  CPT(lite, 9.5M tok ×2ep) ──┬── 外部 OOD SFT ──→ 评测臂 A
  base（不做 CPT）────────────┴── 同一套 OOD SFT ──→ 评测臂 B
  结论 = A vs B 增益差。训练内 val 砍掉（val0），评测靠双臂外部对照。

Phase B（增益确认后）  全量线（Flash WSD 两段）
  stage1 恒 LR 大包 → stage2 linear 退火（STAGE2_DECAY_ITERS 以 stage1
  启动日志实打 train_iters 为准校正）。数据 = bundle v2.x（cpt_kb_mm 全量
  15,137 行）+ sch 并入（v2.3）+ QA→CoT v0.3 终化数据。

Phase C（长期演进）  Pro 错例闭环
  四类错例簇归因（固定权重补索引优先）→ 50/30/20 增量 CPT + 异模板 SFT + Montessori。
```

- **lite = 平铺 constant 恒 LR**（无 WSD）：epochs 型小包不值两段切；WSD 是全量线设计。
- **lite 砍 val**：12,085 行小包里 val 无统计意义还占数据；增益评测靠外部 OOD SFT 双臂对照。
- **数据零重复**：bundle 内零重复，重复由训练侧 `num_epochs` 控制（不违反“不注水稀缺通道”铁律）。

---

## 2. 数据来源与配比（`bundle_lite_v3_all·val0`）

### 2.1 三个源头（`manifest.json` 实测，built_at 2026-11-18 06:40:11）

| 源 | 文件 | 行数 | 校验 |
|---|---|---|---|
| kb 全量池 | `pt/cpt_kb_mm.jsonl` | 15,137 | sha256 `e2cfe739…f177d` |
| kb 补充 | `pt/cpt_kb_supplement.jsonl` | 33 | sha256 `60eb1a33…e0ad` |
| **sch iso 改造** | `pt/cpt_sch_mm.jsonl` | **11,502** | sha256 `54e93973…3bfd`（md5 `f90e7a01`） |

**选取规则**（manifest 原文）：`level in T1_FULL(52) OR rel-bound to 30 template id variants OR sch_tpl（iso 改造源全选）`；`T1_FULL 52 = a_tier 24 + generic 5 + disney_movies 23`；`ft_mentions_excluded: true`。

### 2.2 汇总账

| 维度 | 数字 |
|---|---|
| 总行数 | **12,085**（`wc -l` 实测 = manifest `rows`） |
| sch | 11,502（iso 改造源 11,513 行 → 改造产出 11,502） |
| kb 池 | 550 |
| kb 补充 | 33 |
| 带图行 | **11,574** |
| mm 行 | 72 |
| token（train） | **9,549,679** |
| 步/epoch | 18.21（GBS32×16384 口径） |
| val | 0（`val_ratio 0.0`，`zero_leakage_assert passed`，seed 42） |

### 2.3 通道配比（行数，加和 = 12,085）

| 通道 | 行数 | 说明 |
|---|---|---|
| `ch_ip_grounding` | 3,352 | `desc` + 点名段（品牌/IP 实体锚定） |
| `ch_ip_negative` | 5,985 | 显式「元素：无。」负样本（防误杀） |
| `ch5_flash_rag_raft` | 2,165 | kb title-scan 命中 `elements` 门禁升格三段式 + S2 干扰 |
| `kb text` | 440 | `kb_ip 342 / kb_audit 36 / kb_entity 35 / kb_ip_supp 21 / kb_logo 3 / kb_bc 2 / kb_doc 1` |
| `kb mm` | 72 | `kb_ip_mm 42 / kb_ip_supp_mm 12 / kb_entity_mm 10 / kb_audit_mm 5 / kb_logo_mm 3` |
| `kb rel` | 71 | `kb_rel_entity_audit 34 / kb_rel_logo_audit 20 / kb_rel_ip_audit 9 / kb_rel_audit_ip 8` |

### 2.4 `sch iso` 改造要点（[`build_cpt_from_sch.py`](../../../training/lite/build_cpt_from_sch.py)）

- **CPT 格式，不是 SFT 格式**：问句剥除、单 `assistant` 背书行、`<image>` 置顶。
- `desc + elements` 原子字段自拼（`polished` 版文本融合改写会丢失点名段头，不可用）。
- **通道判别以 `elements` 实际内容为准，`label` 字段降级为审计字段（`audit_raw_label`）**——`sch` 源 7,778 行 `label=True` 里有 2,462 行 `elements=「无」`，`label` 不可信。
- `sch` 96 模板全量进 lite（早期 30 模板口径 2,673 行作废）。
- 30 模板中 9 个无 `sch` 数据（067/132/150/408/268/296/398/467/481），靠 `kb + 补充` 覆盖。
- 图检：10,853 唯一图 PIL verify 全检，0 缺 / 0 零字节 / 0 不可解码。

### 2.5 文件台账（`algo4/tpz/data/v-CPT/pt/`）

| 文件 | md5 |
|---|---|
| `bundle_lite_v3_all/lite_train.jsonl` | `3265cfe8fa8afbf1cf945d5b2e8802c7` |
| `bundle_lite_v3_all/lite_val.jsonl` | 空壳 `d41d8cd98f00b204e9800998ecf8427e` |
| `bundle_lite_v3_all/manifest.json` | `b8a401fc` |
| `bundle_lite_compat/stage1_train.jsonl` | symlink 指向 `bundle_lite_v3_all/lite_train.jsonl` |

---

## 3. 运行配方（v2.0 校准版 —— 解决 B300 `7 steps` 崩溃）

> **v1.0 `tqdm 0/7` 根因复盘**：`bundle_lite_v3_all` 中 11,502/12,085（95.2%）均为 `sch` 短行（约 50 字 + 缩略图 ≈ **150 tok/行**，远低于离线高估的 790 tok/行），1 epoch 真实总 token 仅 `~1.83M`；叠加 `GBS=32 × MAX_LENGTH=16384 = 524,288 tok/step`，2 epochs 仅得 `(1.83M × 2) / 524,288 = 7.0 steps`！同时 16K 打包过度吞并短行、若降成 4K 又会截断 4K–8K KB 文本。
> **v2.0 修复**：1. Lite 默认改由 [`training/lite/slice_flash_to_lite.py`](../../../training/lite/slice_flash_to_lite.py) 从 Flash 4 桶池按 `doc_family_id` 整家族按比例切片 **15%**（`val0`，不含原始 `<think>` ChatML）；2. B300 Lite 收窄为 **`MBS=1 / GBS=4 / MAX_LENGTH=8192`**（`32,768 tok/step` → 稳定 **~100–150 steps**，兼顾 8K 完整文本留存）。

| 参数 | v1.0（已废弃，`7 steps`） | v2.0（当前校准版） | 备注 |
|---|---|---|---|
| 数据源 | `bundle_lite_v3_all` (95% `sch`) | `bundle_lite/lite_train.jsonl` (15% Flash 4 桶切片) | `bundle_lite_compat/stage1_train.jsonl` 软链兼容；`LITE_MODE=sch_legacy` 仍保留 |
| epochs / SAVE | `2` / `SAVE_STEPS=128` | **`2`** / **`SAVE_STEPS=40`** | 投影步数 `~100–150` 步，`~40` 步留 1 档中间 checkpoint |
| LR | `constant 1e-5` | **`constant 8e-6`** + warmup `6%` | `TP4 × DP2`，冻结 ViT、保留 Aligner (`2e-6`) |
| MBS / GBS / seq | `4 / 32 / 16384` (524K/step) | **`1 / 4 / 8192`** (**32.8K/step**) | Flash/Pro 统一用 `2 / 16 / 8192`（131K/step） |
| attention / fusion | `flash` / `--gradient_accumulation_fusion false` | 同左 | APEX 地雷关闭、TE `sm_103` 混搭不变 |

**B300 显存账**：27B 全参 TP4 DP2 + dist-opt ≈ 67.5 GB/卡 → 余 ~201–207 GB/卡激活；MBS4×16384 经 TP4 分摊 = 每卡 16K tok/micro-batch，余量充裕。

---

## 4. 发射链脚本台账（`CPT-SFT/training/`）

| 文件 | 角色 |
|---|---|
| [`training/lite/slice_flash_to_lite.py`](../../../training/lite/slice_flash_to_lite.py) | Flash 4 桶池整家族 `15%` 按比例分层切片 + B300 `MBS=1/GBS=4/8192` 步数自检（`<60` 步告警） |
| [`training/cpt_27b/run_cpt_27b_b300.sh`](../../../training/cpt_27b/run_cpt_27b_b300.sh) | wrapper：支持 `lite` / `flash` / `pro` / `pro_inc` / 兼容 `stage1`/`stage2`；缺省 dry-run，`--launch` 才实发 |
| [`training/cpt_27b/train_cpt_qwen3_5_27b_megatron.sh`](../../../training/cpt_27b/train_cpt_qwen3_5_27b_megatron.sh) | 薄壳（含 `--gradient_accumulation_fusion false`；纪律：`train_iters` 偏差 >10% 杀掉重发登记） |
| [`training/lib/_core.sh`](../../../training/lib/_core.sh) | 公共层（日志、命令打印、`DRY_RUN` / `--launch` 门禁） |
| [`training/lib/hw_b300.sh`](../../../training/lib/hw_b300.sh) | B300 硬件层（默认 `DEF_MICRO_BATCH_SIZE=2, DEF_GLOBAL_BATCH_SIZE=16, DEF_MAX_LENGTH=8192`、`TP_SIZE` 整除校验、TE `sm_103` 预检） |

---

## 5. 环境矩阵与两枚实弹排险

### 5.1 Pod NFS venv 矩阵（2026-11-18 实测）

| venv | torch | swift | TE pybind | TE lib arch | B300 可训 |
|---|---|---|---|---|---|
| `tpz` | 2.13.0+cu130 | 4.7.0.dev0 | `b0d4ad31` | `sm_90 + sm_120` | ❌ |
| **`tpz_caitlin`（已修复）** | 2.13.0+cu130 | 4.7.0.dev0 | `b0d4ad31`（老版回植） | `e6808132`（`sm_75/80/89/90/100/103/120`） | ✅（卡 5 smoke 全绿） |
| `tpz_xayah` | 2.13.0+cu130 | 4.3.2 | `dc3deb16`（对 2.14 编） | `e6808132`（全 arch） | lib✓ pybind✗ |

### 5.2 两枚实弹地雷排险总结

1. **已修：APEX `gradient_accumulation_fusion`**：纯 py apex 包骗过 swift 自动降级导致缺 `fused_weight_gradient_mlp_cuda`。薄壳已默认关闭 `--gradient_accumulation_fusion false`（可用环境变量覆盖）。
2. **已修：TE `sm_103` 内核缺失（`rmsnorm_fwd_cuda_kernel.cu:76`）**：
   - `tpz_caitlin` 原 TE 仅编译 `sm_90 + sm_120`，缺少 B300 `sm_103` cubin；
   - `tpz_xayah` 的 TE lib 含 `sm_103` 但 pybind 针对 torch 2.14 编译（`import transformer_engine.pytorch` 报错 COW 符号缺失；**仅顶层 `import transformer_engine` 是假绿**）；
   - **已落地混搭**：将 `tpz_xayah` 多 arch TE lib 移植入 `tpz_caitlin` 并回植 torch 2.13 兼容的 `b0d4ad31` pybind，在 pod 卡 5（B300 `sm_103`）完成 `LayerNormLinear` + `LayerNormMLP` BF16 前向 `SMOKE_OK×2`。

### 5.3 重发验证清单

1. 执行 `run_cpt_27b_b300.sh lite --launch`（默认走 `MBS=1 / GBS=4 / 8192`）；
2. **核对启动日志 `train_iters >= 80`（目标 `100–150` 步）**——若仍出现个位数 steps 立即停机检查 `GBS` 或切片总量；
3. 双臂 OOD SFT 后同时评测 `enable_thinking=true` 与 `enable_thinking=false` 两档指标，确保无 Thinking-OFF 塌陷。
