# 生图 v2 按域分区产线与多图打包规范（`image_gen_v2_plan.md`）

> 版本：v2.0（2026-11-19，替代无差别 8.7 万张 `gen_kb_v1`；实现脚本：[`training/flash/build_v2_image_pipeline.py`](../../../training/flash/build_v2_image_pipeline.py)）
> 底层引擎：Qwen-Image-2.1（31G，vLLM-Omni `/v1/images/generations` & `/v1/images/edits`，遵守 `Datawheel/docs/QWEN_IMAGE_2_1_PIPELINE.md` 物理锚定与白名单合成门禁）

---

## 1. 为什么停掉 v1 并重构为 v2？

- **v1 病灶**：对 `kb_entities`（8,692）× 10 场景无差别纯文本生图，导致早期配额被 `logo`/`audit`/`brand` 商品图占满，最终产出 9,749 张（14GB）时真正急需的 wiki IP / 符号图远未跑完。而全库其实已经拥有 **2,627,519 张唯一实拍商品图（`v` 系列，100% 台账命中免验）**，`audit`/`logo` 根本无需再用扩散模型低效重写。
- **单个 Doc 硬生成 10 张并拆 10 行的致命风险**：按 1 doc → 10 行铺量会把文本语料机械重复 10 遍（~580k tokens 严重过拟合），且大量扩散近似图会把 8×B300 的 CPT 阶段退化为过拟合低质分布的“呆子模型”。

---

## 2. 8,692 实体全谱按域分区决策表

| 域（规模） | v2 处置 | 路由与生成规则（`route_entity_domain`） | 质检与防翻车红线 |
|---|---:|---|---|
| **`logo` (3,426)** | **停** | 停止生成——文字/标牌自体可辨识，不需要扩散补图 | — |
| **`audit` (423)** | **停** | 停止生成——直接由 263 万张真实 `v` 系列商品图库供给 | — |
| **`doc/overview/lace/long` (43)** | **停** | 停止生成——非视觉实体段落 | — |
| **动漫 IP (2,305)**：`ip 1,220` + `dex 1,025` + `bc 45` + `figure 15` | **生成** | **`mm=True`（如 `dex`）强制走 I2I 参考图保真路线**；`mm=False` 走角色/场景 T2I；每实体生成 ≤4 张 | 参考图特征锁定，防止人物瞳色/配饰漂移 |
| **招式/特性 (1,292)**：`move 919` + `ability 373` | **节流降 75%** | **跳过抽象光效/能量波纹**（易引发通用纹理误报）；仅保留 `has_concrete_prop=True` 具体实体道具（≤2 张/实体） | `throttled_abstract_effect` |
| **品牌 (640)**：`brand 123` + `craft 338` + `entity 179` | **生成（元素解剖）** | 重要代表产品 + **局部设计元素特写**（Adidas 三道斜杠、Coke 弧线瓶、BMW 双肾），优先参考图 Edit 路线，不做通用泛目录图 | **强上 VLM + OCR + 条数/要素计数检验**；不合格错图直接转入 CF 负例 |
| **符号 (563)**：`flag 250` + `geo 313` | **严格 Edit-Only** | **严禁 `flag`/`geo` 走纯 T2I！** 必须基于标准矢量底图 + Qwen-Image-Edit 生成正面标准/飘扬/徽章贴图场景 | 无参考矢量底图标为 `pending_vector_asset`，绝不空转纯 T2I |
| **体育递归展开（`kb_sports`）** | **生成** | **西甲式递归展开**：联赛父 doc → 子俱乐部 → 队徽（`crest`）+ 主客场球衣（`kit`），继承父 `doc_family_id` 防止分裂泄漏 | 同品牌解剖门禁 |

---

## 3. `gen_kb_v1`（9,749 张停跑遗产）三路精确回收

1. **Clean 合格 `brand`/`craft`/`anime`**（VLM 分 `>= 0.85` 且 OCR/要素数正确）→ 并入 v2 正样本包（`recycle_to_v2_positive`）；
2. **畸变残次 `logo`/`brand`**（OCR 错误/条数错位如 4 道杠、且存在对照原图）→ 与官方参照图组成 **双图反事实难负例**（`ch_v1_recycled_neg`，`recycle_as_cf_negative`）；
3. **冗余 `audit`/低质无参考废片** → 直接清理剪枝（`prune_discard`）。

---

## 4. 多图单 Doc 打包与合成占比红线（防呆子化核心机制）

1. **Multi-Image-per-Doc 打包（`doc_text_repeat == 1`）**：
   - 将每个实体 VLM 评分排序的 **Top 2~4 张多视角/多场景图** + 参考基准图打进 **同一条 Mid-Training Doc 行**（`ch_v2_multi_img_doc` / `ch_sports_expanded`），文首对齐对应数量的 `<image>` 占位符。文本描述 **仅出现 1 次**，彻底杜绝同一条 Doc 复制 10 遍造成的文本过拟合。
2. **失败残次图零浪费转反事实（`ch_v2_cf_pair`）**：
   - VLM/OCR 校验不通过的生成图（如 4 道杠错误件），若存在原图参照，自动配对为 `[参考原图, 畸变生成图]` 双图反事实对比行，教模型精确识别要素数目与色序错误。
3. **全局合成图比例红线 `<= 25%`**：
   - [`build_mm_wiki_cpt.py`](../../../training/flash/build_mm_wiki_cpt.py) `--mode midtrain` 硬约束：整个 Mid-Training 混合集中合成图占比 `synth_image_share <= 0.25`，保证视觉表征锚定在真实商品图像域内。
