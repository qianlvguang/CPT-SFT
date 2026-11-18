#!/usr/bin/env python3
"""Warm Claude-Style Architecture Workflow Diagram Renderer for CPT-SFT.

Generates `docs/training_plans/assets/cpt_sft_lite_flash_pro_workflow.png`
at 3600x2280 resolution matching the warm cream/terracotta/sage/slate/amber/plum
visual design of `docs/training_plans/flash/assets/b300_27b_cpt_sft_pipeline.png`.

Requires Pillow (PIL). Run with an environment that provides Pillow, e.g.:
  /Users/nitro/Documents/Coding/Datawheel/.venv/bin/python3 tools/render_workflow_diagram.py
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "training_plans" / "assets" / "cpt_sft_lite_flash_pro_workflow.png"

WIDTH = 3600
HEIGHT = 2280

# Warm Claude Palette (matched to b300_27b_cpt_sft_pipeline.png)
BG_CREAM = (250, 248, 244)        # #FAF8F4
COL_BG = (243, 240, 233)          # #F3F0E9
COL_BORDER = (221, 214, 203)      # #DDD6CB
COL_HDR_BG = (235, 230, 221)      # #EBE6DD
DASH_BG = (247, 245, 240)         # #F7F5F0
DASH_BORDER = (205, 197, 185)     # #CDC5B9
CARD_BG = (255, 254, 251)         # #FFFEFB
CARD_BORDER = (224, 218, 209)     # #E0DAD1
CARD_SHADOW = (232, 227, 218)     # #E8E3DA
INK_MAIN = (31, 27, 24)           # #1F1B18
INK_SUB = (94, 86, 78)            # #5E564E
INK_MUTED = (128, 119, 110)       # #80776E
CODE_BG = (239, 235, 228)         # #EFEBE4
CODE_TEXT = (60, 53, 47)          # #3C352F
ARROW_ACCENT = (176, 86, 52)      # #B05634

THEMES = {
    "blue": {
        "bar": (74, 123, 167),
        "pill_bg": (235, 243, 250),
        "pill_border": (195, 218, 238),
        "pill_text": (45, 86, 124),
    },
    "green": {
        "bar": (61, 122, 90),
        "pill_bg": (234, 244, 238),
        "pill_border": (189, 222, 203),
        "pill_text": (38, 89, 63),
    },
    "terracotta": {
        "bar": (200, 90, 50),
        "pill_bg": (252, 238, 232),
        "pill_border": (241, 199, 182),
        "pill_text": (158, 62, 28),
    },
    "amber": {
        "bar": (184, 115, 36),
        "pill_bg": (253, 244, 231),
        "pill_border": (238, 209, 166),
        "pill_text": (138, 83, 20),
    },
    "purple": {
        "bar": (123, 82, 148),
        "pill_bg": (243, 238, 248),
        "pill_border": (216, 198, 232),
        "pill_text": (90, 54, 112),
    },
}


def load_fonts() -> dict[str, ImageFont.FreeTypeFont]:
    hiragino = "/System/Library/Fonts/Hiragino Sans GB.ttc"
    menlo = "/System/Library/Fonts/Menlo.ttc"
    if not Path(hiragino).exists():
        raise FileNotFoundError(f"Missing macOS CJK font: {hiragino}")
    return {
        "title": ImageFont.truetype(hiragino, 38, index=2),
        "subtitle": ImageFont.truetype(hiragino, 19, index=0),
        "col_num": ImageFont.truetype(hiragino, 19, index=2),
        "col_title": ImageFont.truetype(hiragino, 20, index=2),
        "col_tag": ImageFont.truetype(menlo, 14, index=0),
        "sub_title": ImageFont.truetype(hiragino, 16, index=2),
        "card_id": ImageFont.truetype(menlo, 14, index=1),
        "card_title": ImageFont.truetype(hiragino, 18, index=2),
        "card_badge": ImageFont.truetype(hiragino, 14, index=2),
        "card_body": ImageFont.truetype(hiragino, 15, index=0),
        "card_code_ascii": ImageFont.truetype(menlo, 14, index=0),
        "card_code_cjk": ImageFont.truetype(hiragino, 14, index=0),
        "legend": ImageFont.truetype(hiragino, 16, index=2),
        "edge_label": ImageFont.truetype(hiragino, 14, index=2),
    }


def _split_runs(text: str) -> list[tuple[bool, str]]:
    """Splits text into (is_ascii, segment) runs so Menlo only renders ASCII."""
    if not text:
        return []
    runs: list[tuple[bool, str]] = []
    cur_ascii = text[0].isascii()
    buf = [text[0]]
    for ch in text[1:]:
        is_asc = ch.isascii()
        if is_asc == cur_ascii:
            buf.append(ch)
        else:
            runs.append((cur_ascii, "".join(buf)))
            cur_ascii = is_asc
            buf = [ch]
    runs.append((cur_ascii, "".join(buf)))
    return runs


def mixed_text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    ascii_font: ImageFont.FreeTypeFont,
    cjk_font: ImageFont.FreeTypeFont,
) -> tuple[int, int]:
    total_w = 0
    max_h = 0
    for is_asc, seg in _split_runs(text):
        font = ascii_font if is_asc else cjk_font
        bbox = draw.textbbox((0, 0), seg, font=font)
        total_w += bbox[2] - bbox[0] + 1
        max_h = max(max_h, bbox[3] - bbox[1])
    return max(0, total_w - 1), max_h


def draw_mixed_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    ascii_font: ImageFont.FreeTypeFont,
    cjk_font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> None:
    x, y = xy
    for is_asc, seg in _split_runs(text):
        font = ascii_font if is_asc else cjk_font
        y_adj = y if is_asc else y + 1
        draw.text((x, y_adj), seg, font=font, fill=fill)
        bbox = draw.textbbox((0, 0), seg, font=font)
        x += (bbox[2] - bbox[0]) + 1


@dataclass
class CardSpec:
    cid: str
    theme: str
    title: str
    badge: str
    code_line: str
    lines: list[str]


@dataclass
class SubBoxSpec:
    title: str
    cards: list[CardSpec]


@dataclass
class ColumnSpec:
    num: str
    title: str
    tag: str
    subboxes: list[SubBoxSpec]


COLUMNS: list[ColumnSpec] = [
    ColumnSpec(
        num="I",
        title="源资产台账与归档",
        tag="v-CPT+Archive",
        subboxes=[
            SubBoxSpec(
                title="现役 PT & QA 资产池",
                cards=[
                    CardSpec(
                        cid="A1",
                        theme="blue",
                        title="pt/cpt_sch_mm.jsonl",
                        badge="96模板全选",
                        code_line="11,502 行 (原11,513) · md5 f90e7a01",
                        lines=[
                            "desc + elements 原子字段自拼 CPT 正文",
                            "label 不可信 (7,778 True 含 2,462 无)",
                        ],
                    ),
                    CardSpec(
                        cid="A2",
                        theme="blue",
                        title="cpt_kb_mm + 补充",
                        badge="T1_FULL 52",
                        code_line="15,137 + 33 行 → Lite 筛选 550+33",
                        lines=[
                            "a_tier 24 / generic 5 / disney 23 + 30模板",
                            "排除 ft_mentions · 覆盖 9 个无 sch 模板",
                        ],
                    ),
                    CardSpec(
                        cid="A3",
                        theme="blue",
                        title="cpt_ch2_m2 + v_mm",
                        badge="Flash全量池",
                        code_line="ch2: 9,420 双图 · v_mm: 22,000 行",
                        lines=[
                            "标准 Logo/国旗双图对比 + V1/V3/V4 回放",
                            "V4 挂检索证据自动升格 ch5_flash_rag_raft",
                        ],
                    ),
                    CardSpec(
                        cid="A4",
                        theme="purple",
                        title="qa/qa_kb_raw.jsonl",
                        badge="227,656 条",
                        code_line="md5 318fdb1e · 原产 qwen3.8-flash-next",
                        lines=[
                            "insufficient 43,748 · audit 35,067 · syn 23,059",
                            "多证据 50% · answerable False 40,633 · 残行 6",
                        ],
                    ),
                ],
            ),
            SubBoxSpec(
                title="外部对照集与历史归档隔离",
                cards=[
                    CardSpec(
                        cid="A5",
                        theme="amber",
                        title="外部 OOD SFT 评测池",
                        badge="多臂同源",
                        code_line="doc_family_id 与训练池严格零交叉",
                        lines=[
                            "Lite 双臂增益对照与 Flash 三方对照共用",
                            "密封人审金标不进入 CPT / SFT / 检索库",
                        ],
                    ),
                    CardSpec(
                        cid="A6",
                        theme="green",
                        title="archive/ H200 & QSA",
                        badge="主线清净",
                        code_line="archive/training/{qwen35_27b_h200,qsa_muon}",
                        lines=[
                            "旧 H200 4.3.2 三阶段线与 35B-A3B QSA 归档",
                            "保留 test_recommended_pipeline.py 9/9 回归",
                        ],
                    ),
                ],
            ),
        ],
    ),
    ColumnSpec(
        num="II",
        title="Phase A · Lite 增益",
        tag="training/lite",
        subboxes=[
            SubBoxSpec(
                title="sch iso 改造与通道守卫",
                cards=[
                    CardSpec(
                        cid="L1",
                        theme="green",
                        title="build_cpt_from_sch.py",
                        badge="sch iso",
                        code_line="剥问句 → 单 assistant + <image> 置顶",
                        lines=[
                            "禁用 polished_text 改写版，仅用 desc+elements",
                            "KB title-scan 命中且含 S2 升格 ch5 三段式",
                        ],
                    ),
                    CardSpec(
                        cid="L2",
                        theme="amber",
                        title="elements 通道判定",
                        badge="修2,462行",
                        code_line="label -> audit_raw_label (仅作审计)",
                        lines=[
                            "elements=空/无 → 强制入 ch_ip_negative",
                            "正文中显式落盘「元素：无。」防线上误杀",
                        ],
                    ),
                    CardSpec(
                        cid="L3",
                        theme="blue",
                        title="6 大通道汇总",
                        badge="12,085 行",
                        code_line="9,549,679 tok · 18.21 步/ep (GBS32×16K)",
                        lines=[
                            "grounding 3,352 · negative 5,985 · ch5 2,165",
                            "kb_text 440 · kb_mm 72 · kb_rel 71 (加和一致)",
                        ],
                    ),
                ],
            ),
            SubBoxSpec(
                title="bundle_lite_v3_all·val0 & 对照",
                cards=[
                    CardSpec(
                        cid="L4",
                        theme="green",
                        title="val0 打包 + 全量图检",
                        badge="10,853图 0缺",
                        code_line="bundle_lite_compat/stage1_train.jsonl",
                        lines=[
                            "包内零重复，val_ratio=0.0 (lite_val 空壳)",
                            "PIL verify 0 缺 / 0 零字节 / 0 不可解码",
                        ],
                    ),
                    CardSpec(
                        cid="L5",
                        theme="terracotta",
                        title="Lite CPT (2ep ≈ 36步)",
                        badge="constant 1e-5",
                        code_line="cpt-qwen3_5-27b-lite-v1.0.0/stage1",
                        lines=[
                            "平铺 constant 1e-5 + warmup 6% (不做 WSD)",
                            "27B 全参 TP4 DP2 MBS4 GBS32 len=16384",
                        ],
                    ),
                    CardSpec(
                        cid="L6",
                        theme="amber",
                        title="双臂外部 OOD 对照",
                        badge="Arm A vs B",
                        code_line="run_lite_pipeline.sh (CPT+OOD vs Base)",
                        lines=[
                            "在同一套外部 OOD SFT 与评测集上对比净增益",
                            "增益确认后锁定基座并解锁 Phase B 全量线",
                        ],
                    ),
                ],
            ),
        ],
    ),
    ColumnSpec(
        num="III",
        title="B300 权威发射底盘",
        tag="lib+cpt_27b",
        subboxes=[
            SubBoxSpec(
                title="统一脚本与 B300 拓扑",
                cards=[
                    CardSpec(
                        cid="B1",
                        theme="blue",
                        title="run_cpt_27b_b300.sh",
                        badge="四档Profile",
                        code_line="lite | stage1 | stage2 | pro --launch",
                        lines=[
                            "缺省仅 dry-run，传 --launch 才真正发射",
                            "强制 --model 必填校验并拦截 Flash-Next 误入",
                        ],
                    ),
                    CardSpec(
                        cid="B2",
                        theme="blue",
                        title="hw_b300.sh + _core.sh",
                        badge="8×268.6GiB",
                        code_line="TP4 DP2 (GQA KV=4禁TP8) · MBS4 GBS32",
                        lines=[
                            "27B+dist-opt ≈ 67.5GB/卡，余 ~207GB/卡激活",
                            "freeze_vit=true / aligner=false (2e-6) / 无MTP",
                        ],
                    ),
                ],
            ),
            SubBoxSpec(
                title="两枚实弹排险与启动核验",
                cards=[
                    CardSpec(
                        cid="B3",
                        theme="green",
                        title="地雷 1：APEX fusion",
                        badge="默认关闭",
                        code_line="--gradient_accumulation_fusion false",
                        lines=[
                            "绕过纯 py apex 骗过 swift 降级导致的",
                            "fused_weight_gradient_mlp_cuda 缺失崩溃",
                        ],
                    ),
                    CardSpec(
                        cid="B4",
                        theme="green",
                        title="地雷 2：TE sm_103 混搭",
                        badge="卡5全绿×2",
                        code_line="tpz_caitlin: 老pybind(t2.13)+xayah lib(sm103)",
                        lines=[
                            "验证必须 import transformer_engine.pytorch",
                            "LayerNormLinear & LayerNormMLP bf16 SMOKE_OK",
                        ],
                    ),
                    CardSpec(
                        cid="B5",
                        theme="amber",
                        title="重发启动纪律与预案",
                        badge="偏差>10%杀",
                        code_line="核对启动日志 train_iters≈36 (排 tqdm 0/7)",
                        lines=[
                            "偏差 >10% 立即停机登记排查 packing 口径",
                            "OOM 预案 env MICRO_BATCH_SIZE=2/1 降档",
                        ],
                    ),
                ],
            ),
        ],
    ),
    ColumnSpec(
        num="IV",
        title="Phase B · Flash WSD",
        tag="training/flash",
        subboxes=[
            SubBoxSpec(
                title="多源重构与 v2.3 全量建包",
                cards=[
                    CardSpec(
                        cid="F1",
                        theme="blue",
                        title="refactor_v1_v4_pipeline",
                        badge="三元去重",
                        code_line="(variant, sample_id, cate_id, images)",
                        lines=[
                            "V2 默认过滤防稀释；V4 final 禁开 thinking",
                            "V1/V3 难例抽种子交 Qwen-Image 2.1 生反事实图",
                        ],
                    ),
                    CardSpec(
                        cid="F2",
                        theme="green",
                        title="build_mm_wiki_cpt (v2.3)",
                        badge="16K+倍限",
                        code_line="--max-tokens 16384 --max-row-repeats 3",
                        lines=[
                            "合并 kb/m2/v 全量 + sch (grounding/negative)",
                            "481 文档家族零泄漏切分，生成 SHA256 清单",
                        ],
                    ),
                ],
            ),
            SubBoxSpec(
                title="两阶段 WSD 课程与 HF 导出",
                cards=[
                    CardSpec(
                        cid="F3",
                        theme="terracotta",
                        title="Stage 1 恒定 LR 训练",
                        badge="保优化器",
                        code_line="PHASE=stage1 · constant 1e-5 · warmup 6%",
                        lines=[
                            "NO_SAVE_OPTIM=false 保留 AdamW m_t, v_t",
                            "消除 Stage 2 衔接冷启动 Gradient Spike",
                        ],
                    ),
                    CardSpec(
                        cid="F4",
                        theme="amber",
                        title="闭卷探针 & Stage 2 选包",
                        badge="11M~15M tok",
                        code_line="is_unresolved_tail=true 聚焦顽固长尾",
                        lines=[
                            "严控单行重复 <=3×，防 ch_rel_jump 死记模板",
                            "以 Stage 1 实打日志步数校准退火步数",
                        ],
                    ),
                    CardSpec(
                        cid="F5",
                        theme="terracotta",
                        title="Stage 2 WSD → HF 底座",
                        badge="WSD linear",
                        code_line="--lr_decay_style WSD --lr_wsd_decay_style linear",
                        lines=[
                            "swift export --to_hf true 输出 BF16 底座",
                            "永久留存 Stage 1 退火前档与 Stage 2 末档",
                        ],
                    ),
                ],
            ),
        ],
    ),
    ColumnSpec(
        num="V",
        title="QA→CoT v0.3 & SFT",
        tag="gen_cot_from_qa",
        subboxes=[
            SubBoxSpec(
                title="逆向桥接与 qwen3.8-flash 蒸馏",
                cards=[
                    CardSpec(
                        cid="C1",
                        theme="purple",
                        title="题型清洗 + 干扰项重映射",
                        badge="去位置捷径",
                        code_line="synonsense(6)→synthesis · [S_old]→[S_new]",
                        lines=[
                            "混入 1~3 条同域干扰项后洗牌重排 S1..Sn",
                            "单轮正则同步重写答案/引文中的引用编号",
                        ],
                    ),
                    CardSpec(
                        cid="C2",
                        theme="purple",
                        title="qwen3.8-flash 桥接",
                        badge="budget≥2048",
                        code_line="prompts/qa_cot_bridge_v03.md (thinking ON)",
                        lines=[
                            "全链冷酷客观核验流，无 Markdown 死板段头",
                            "insufficient 双分支 / audit 豁免 / synthesis 仲裁",
                        ],
                    ),
                ],
            ),
            SubBoxSpec(
                title="八道门控 + 降级 + 三方 SFT",
                cards=[
                    CardSpec(
                        cid="C3",
                        theme="green",
                        title="八道程序质检门",
                        badge="8 Gates",
                        code_line="verify_cot_8gates() 精确拦截跳步与倒推",
                        lines=[
                            "1终答 · 2精度 · 3召回 · 4子串SHA256无段头",
                            "5反事实 · 6防复述(≤0.55,摘录≤45%) · 7长 · 8三禁",
                        ],
                    ),
                    CardSpec(
                        cid="C4",
                        theme="amber",
                        title="动态长度门 & 降级路由",
                        badge="禁拼接Quote",
                        code_line="fact≥180 / mid≥350 / heavy≥500 (软3/6/900)",
                        lines=[
                            "失败样本置 reasoning=\"\" + enable_thinking=false",
                            "落入 qa_direct_fallback 保持思考切换能力",
                        ],
                    ),
                    CardSpec(
                        cid="C5",
                        theme="terracotta",
                        title="三方对照统一 SFT",
                        badge="8:2 混配",
                        code_line="ood_only vs +QA-Direct vs +QA-CoT (8:2)",
                        lines=[
                            "smoke 50 分层人检后扩量接入统一 SFT",
                            "监控 loss_V1..V4 + wiki_rag_qa_cot/direct",
                        ],
                    ),
                ],
            ),
        ],
    ),
    ColumnSpec(
        num="VI",
        title="Phase C · Pro 闭环",
        tag="training/pro",
        subboxes=[
            SubBoxSpec(
                title="四维归因与 50/30/20 增量",
                cards=[
                    CardSpec(
                        cid="P1",
                        theme="amber",
                        title="error_cluster_miner.py",
                        badge="固定权重优先",
                        code_line="missing_index / mapping_drift → 不更新权重",
                        lines=[
                            "先补索引别名/修映射并固定权重复测 Recall",
                            "撞风格误杀 & 缺证据强判 verified 后进训练池",
                        ],
                    ),
                    CardSpec(
                        cid="P2",
                        theme="terracotta",
                        title="50/30/20 增量 + 隔离",
                        badge="sft_overlap=0",
                        code_line="cosine 5e-6 · 从已验收父 CPT 开分支",
                        lines=[
                            "50% 新核验缺口 + 30% 已验收模板 + 20% V回放",
                            "异模板 SFT 强排所有 CPT template_family_id",
                        ],
                    ),
                ],
            ),
            SubBoxSpec(
                title="跨仓闭环与四维金标验收",
                cards=[
                    CardSpec(
                        cid="P3",
                        theme="purple",
                        title="Montessori + 失败 CoT",
                        badge="DPO/KTO/GRPO",
                        code_line="qa_dpo_rejected_cots.jsonl + influence_scorer",
                        lines=[
                            "单步 Loss 差分探针 I(x,y;D_ref) 筛正例",
                            "黑盒 API ICDPO 进化提示 + 导 DPO/KTO/GRPO",
                        ],
                    ),
                    CardSpec(
                        cid="P4",
                        theme="purple",
                        title="OPD × Datawheel 飞轮",
                        badge="跨仓联动",
                        code_line="OPD bsd_api_bridge + ToDi + Datawheel V6",
                        lines=[
                            "API CoT 作特权上下文 z 行逐 Token 惩罚归因",
                            "Datawheel diagnose_and_refill_v6 定向补反事实",
                        ],
                    ),
                    CardSpec(
                        cid="P5",
                        theme="green",
                        title="线上回放与四维验收",
                        badge="目标 +2 pp",
                        code_line="固定召回 ACC vs 真实链路 ACC 拆分定位",
                        lines=[
                            "30/全量宏ACC · 原创误罚 · 二创召回 · 强判率",
                            "视觉聚焦成对测试 + 95%CI 下界 > 0 方可晋级",
                        ],
                    ),
                ],
            ),
        ],
    ),
]


def draw_dashed_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    radius: int = 12,
    dash: int = 9,
    gap: int = 7,
    width: int = 2,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], radius=radius, fill=fill)
    for x in range(x0 + radius, x1 - radius, dash + gap):
        ex = min(x + dash, x1 - radius)
        draw.line([(x, y0), (ex, y0)], fill=outline, width=width)
        draw.line([(x, y1), (ex, y1)], fill=outline, width=width)
    for y in range(y0 + radius, y1 - radius, dash + gap):
        ey = min(y + dash, y1 - radius)
        draw.line([(x0, y), (x0, ey)], fill=outline, width=width)
        draw.line([(x1, y), (x1, ey)], fill=outline, width=width)
    d = radius * 2
    draw.arc([x0, y0, x0 + d, y0 + d], 180, 270, fill=outline, width=width)
    draw.arc([x1 - d, y0, x1, y0 + d], 270, 360, fill=outline, width=width)
    draw.arc([x0, y1 - d, x0 + d, y1], 90, 180, fill=outline, width=width)
    draw.arc([x1 - d, y1 - d, x1, y1], 0, 90, fill=outline, width=width)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_pill(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    bg: tuple[int, int, int],
    border: tuple[int, int, int],
    fg: tuple[int, int, int],
    pad_x: int = 9,
    pad_y: int = 4,
    radius: int = 10,
    align_right: bool = False,
) -> tuple[int, int, int, int]:
    tw, th = text_size(draw, text, font)
    pw = tw + pad_x * 2
    ph = th + pad_y * 2
    x0 = x - pw if align_right else x
    y0 = y
    draw.rounded_rectangle([x0, y0, x0 + pw, y0 + ph], radius=radius, fill=bg, outline=border, width=2)
    draw.text((x0 + pad_x, y0 + pad_y - 1), text, font=font, fill=fg)
    return (x0, y0, x0 + pw, y0 + ph)


def draw_arrow_head(
    draw: ImageDraw.ImageDraw,
    tip: tuple[int, int],
    direction: str,
    color: tuple[int, int, int],
    size: int = 10,
) -> None:
    tx, ty = tip
    if direction == "right":
        pts = [(tx, ty), (tx - size, ty - size // 2 - 1), (tx - size, ty + size // 2 + 1)]
    elif direction == "left":
        pts = [(tx, ty), (tx + size, ty - size // 2 - 1), (tx + size, ty + size // 2 + 1)]
    elif direction == "down":
        pts = [(tx, ty), (tx - size // 2 - 1, ty - size), (tx + size // 2 + 1, ty - size)]
    else:  # up
        pts = [(tx, ty), (tx - size // 2 - 1, ty + size), (tx + size // 2 + 1, ty + size)]
    draw.polygon(pts, fill=color)


def draw_orth_connector(
    draw: ImageDraw.ImageDraw,
    points: Iterable[tuple[int, int]],
    *,
    color: tuple[int, int, int],
    end_dir: str,
    label: str | None = None,
    label_pos: tuple[int, int] | None = None,
    font: ImageFont.FreeTypeFont | None = None,
    width: int = 3,
    dashed: bool = False,
) -> None:
    pts = list(points)
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        if not dashed:
            draw.line([(ax, ay), (bx, by)], fill=color, width=width)
        else:
            if ax == bx:
                step = 14 if by >= ay else -14
                cur = ay
                while (cur < by) if step > 0 else (cur > by):
                    nxt = min(by, cur + 8) if step > 0 else max(by, cur - 8)
                    draw.line([(ax, cur), (bx, nxt)], fill=color, width=width)
                    cur += step
            else:
                step = 14 if bx >= ax else -14
                cur = ax
                while (cur < bx) if step > 0 else (cur > bx):
                    nxt = min(bx, cur + 8) if step > 0 else max(bx, cur - 8)
                    draw.line([(cur, ay), (nxt, by)], fill=color, width=width)
                    cur += step
    draw_arrow_head(draw, pts[-1], end_dir, color=color, size=11)
    if label and label_pos and font:
        lx, ly = label_pos
        tw, th = text_size(draw, label, font)
        draw.rounded_rectangle(
            [lx - tw // 2 - 9, ly - th // 2 - 4, lx + tw // 2 + 9, ly + th // 2 + 4],
            radius=8,
            fill=CARD_BG,
            outline=color,
            width=2,
        )
        draw.text((lx - tw // 2, ly - th // 2 - 1), label, font=font, fill=color)


def render_diagram(output_path: Path) -> None:
    fonts = load_fonts()
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_CREAM)
    draw = ImageDraw.Draw(img)

    # 1. Header Banner
    title = "8×B300 Qwen3.5-27B 域适应（Lite / Flash / Pro）与 QA→CoT v0.3 全链路流程图"
    subtitle = (
        "AutoVision 8×B300 (sm_103) · TP4/DP2/MBS4/GBS32/16K · "
        "Lite sch iso 12,085行 (9.55M tok) · QA→CoT 227,656条 (qwen3.8-flash + 八道门) · H200/QSA 已归档"
    )
    draw.text((46, 30), title, font=fonts["title"], fill=INK_MAIN)
    draw.text((48, 84), subtitle, font=fonts["subtitle"], fill=INK_SUB)

    legend_items = [
        ("数据源 / 台账", "blue"),
        ("守卫 / 排雷 / 验收", "green"),
        ("核心训练 / 底座", "terracotta"),
        ("审计 / 门控 / 降级", "amber"),
        ("QA→CoT / Pro闭环", "purple"),
    ]
    lx = WIDTH - 46
    for text, theme_key in reversed(legend_items):
        th = THEMES[theme_key]
        box = draw_pill(
            draw,
            lx,
            40,
            text,
            fonts["legend"],
            bg=th["pill_bg"],
            border=th["bar"],
            fg=th["pill_text"],
            pad_x=13,
            pad_y=6,
            radius=13,
            align_right=True,
        )
        lx = box[0] - 11

    draw.line([(46, 126), (WIDTH - 46, 126)], fill=COL_BORDER, width=2)

    # 2. Six-column Grid Geometry (leaving generous top/bottom rails for routing)
    num_cols = len(COLUMNS)
    margin_x = 42
    col_gap = 40
    total_w = WIDTH - margin_x * 2
    col_w = (total_w - col_gap * (num_cols - 1)) // num_cols
    col_top = 172
    col_bottom = HEIGHT - 102

    card_boxes: dict[str, tuple[int, int, int, int]] = {}
    col_centers_x: list[int] = []

    for col_idx, col in enumerate(COLUMNS):
        cx0 = margin_x + col_idx * (col_w + col_gap)
        cx1 = cx0 + col_w
        col_centers_x.append((cx0 + cx1) // 2)

        draw.rounded_rectangle(
            [cx0, col_top, cx1, col_bottom],
            radius=18,
            fill=COL_BG,
            outline=COL_BORDER,
            width=2,
        )

        hdr_h = 52
        draw.rounded_rectangle(
            [cx0, col_top, cx1, col_top + hdr_h],
            radius=18,
            fill=COL_HDR_BG,
            outline=COL_BORDER,
            width=2,
        )
        draw.rectangle([cx0 + 1, col_top + hdr_h - 14, cx1 - 1, col_top + hdr_h], fill=COL_HDR_BG)
        draw.line([(cx0, col_top + hdr_h), (cx1, col_top + hdr_h)], fill=COL_BORDER, width=2)

        num_box = draw_pill(
            draw,
            cx0 + 14,
            col_top + 11,
            col.num,
            fonts["col_num"],
            bg=INK_MAIN,
            border=INK_MAIN,
            fg=CARD_BG,
            pad_x=9,
            pad_y=3,
            radius=8,
        )
        draw.text((num_box[2] + 10, col_top + 14), col.title, font=fonts["col_title"], fill=INK_MAIN)
        tag_w, _ = text_size(draw, col.tag, fonts["col_tag"])
        draw.text((cx1 - 14 - tag_w, col_top + 18), col.tag, font=fonts["col_tag"], fill=INK_MUTED)

        avail_top = col_top + hdr_h + 16
        avail_bottom = col_bottom - 16
        total_cards = sum(len(sb.cards) for sb in col.subboxes)
        sb_gap = 16
        card_gap = 13
        sub_hdr_h = 33
        sub_pad_bottom = 13

        fixed_overhead = (
            len(col.subboxes) * (sub_hdr_h + sub_pad_bottom)
            + max(0, len(col.subboxes) - 1) * sb_gap
            + sum(max(0, len(sb.cards) - 1) for sb in col.subboxes) * card_gap
        )
        card_h = (avail_bottom - avail_top - fixed_overhead) // total_cards

        cur_y = avail_top
        for sb in col.subboxes:
            sb_h = sub_hdr_h + len(sb.cards) * card_h + max(0, len(sb.cards) - 1) * card_gap + sub_pad_bottom
            sb_x0, sb_x1 = cx0 + 14, cx1 - 14
            draw_dashed_rect(
                draw,
                (sb_x0, cur_y, sb_x1, cur_y + sb_h),
                fill=DASH_BG,
                outline=DASH_BORDER,
                radius=14,
            )
            draw.text((sb_x0 + 14, cur_y + 8), sb.title, font=fonts["sub_title"], fill=INK_SUB)

            card_y = cur_y + sub_hdr_h
            for card in sb.cards:
                kx0, kx1 = sb_x0 + 11, sb_x1 - 11
                ky0, ky1 = card_y, card_y + card_h
                card_boxes[card.cid] = (kx0, ky0, kx1, ky1)

                th = THEMES[card.theme]
                draw.rounded_rectangle(
                    [kx0 + 2, ky0 + 3, kx1 + 2, ky1 + 3],
                    radius=12,
                    fill=CARD_SHADOW,
                )
                draw.rounded_rectangle(
                    [kx0, ky0, kx1, ky1],
                    radius=12,
                    fill=CARD_BG,
                    outline=CARD_BORDER,
                    width=2,
                )
                draw.rounded_rectangle([kx0, ky0, kx0 + 8, ky1], radius=6, fill=th["bar"])
                draw.rectangle([kx0 + 4, ky0 + 1, kx0 + 8, ky1 - 1], fill=th["bar"])

                id_box = draw_pill(
                    draw,
                    kx0 + 16,
                    ky0 + 11,
                    card.cid,
                    fonts["card_id"],
                    bg=th["pill_bg"],
                    border=th["pill_border"],
                    fg=th["pill_text"],
                    pad_x=7,
                    pad_y=3,
                    radius=7,
                )
                draw.text(
                    (id_box[2] + 8, ky0 + 12),
                    card.title,
                    font=fonts["card_title"],
                    fill=INK_MAIN,
                )
                draw_pill(
                    draw,
                    kx1 - 10,
                    ky0 + 11,
                    card.badge,
                    fonts["card_badge"],
                    bg=th["pill_bg"],
                    border=th["pill_border"],
                    fg=th["pill_text"],
                    pad_x=8,
                    pad_y=3,
                    radius=8,
                    align_right=True,
                )

                # Code pill strip (mixed Menlo + Hiragino so zero tofu glyphs!)
                code_y = ky0 + 44
                cw, ch = mixed_text_size(
                    draw,
                    card.code_line,
                    fonts["card_code_ascii"],
                    fonts["card_code_cjk"],
                )
                code_pill_w = min(kx1 - kx0 - 28, cw + 18)
                draw.rounded_rectangle(
                    [kx0 + 16, code_y, kx0 + 16 + code_pill_w, code_y + ch + 10],
                    radius=7,
                    fill=CODE_BG,
                )
                draw_mixed_text(
                    draw,
                    (kx0 + 25, code_y + 4),
                    card.code_line,
                    fonts["card_code_ascii"],
                    fonts["card_code_cjk"],
                    fill=CODE_TEXT,
                )

                body_start_y = code_y + ch + 19
                rem_h = max(36, ky1 - 10 - body_start_y)
                line_step = min(30, rem_h // max(1, len(card.lines)))
                for li, line in enumerate(card.lines):
                    draw.text(
                        (kx0 + 18, body_start_y + li * line_step),
                        line,
                        font=fonts["card_body"],
                        fill=INK_SUB,
                    )

                card_y += card_h + card_gap
            cur_y += sb_h + sb_gap

    # 3. Clean Orthogonal Connectors across Gutters & Dedicated Top/Bottom Rails
    def right_mid(cid: str, dy: int = 0) -> tuple[int, int]:
        x0, y0, x1, y1 = card_boxes[cid]
        return x1, (y0 + y1) // 2 + dy

    def left_mid(cid: str, dy: int = 0) -> tuple[int, int]:
        x0, y0, x1, y1 = card_boxes[cid]
        return x0, (y0 + y1) // 2 + dy

    def bottom_mid(cid: str, dx: int = 0) -> tuple[int, int]:
        x0, y0, x1, y1 = card_boxes[cid]
        return (x0 + x1) // 2 + dx, y1

    def gutter_x(col_left_idx: int, offset: int = 0) -> int:
        cx1 = margin_x + col_left_idx * (col_w + col_gap) + col_w
        return cx1 + col_gap // 2 + offset

    g12 = gutter_x(0)
    g23 = gutter_x(1)
    g34 = gutter_x(2)
    g45 = gutter_x(3)
    g56 = gutter_x(4)

    # Col I -> Col II
    draw_orth_connector(
        draw,
        [right_mid("A1"), (g12, right_mid("A1")[1]), (g12, left_mid("L1")[1]), left_mid("L1")],
        color=THEMES["blue"]["bar"],
        end_dir="right",
    )
    draw_orth_connector(
        draw,
        [right_mid("A2"), (g12 - 7, right_mid("A2")[1]), (g12 - 7, left_mid("L3")[1]), left_mid("L3")],
        color=THEMES["blue"]["bar"],
        end_dir="right",
    )
    draw_orth_connector(
        draw,
        [right_mid("A5"), (g12, right_mid("A5")[1]), (g12, left_mid("L6")[1]), left_mid("L6")],
        color=THEMES["amber"]["bar"],
        end_dir="right",
    )

    # Col II -> Col III
    draw_orth_connector(
        draw,
        [right_mid("L4"), (g23, right_mid("L4")[1]), (g23, left_mid("B1")[1]), left_mid("B1")],
        color=THEMES["green"]["bar"],
        end_dir="right",
    )
    draw_orth_connector(
        draw,
        [right_mid("L5", -14), (g23 + 7, right_mid("L5", -14)[1]), (g23 + 7, left_mid("B2")[1]), left_mid("B2")],
        color=THEMES["terracotta"]["bar"],
        end_dir="right",
        dashed=True,
    )

    # Col III -> Col IV
    draw_orth_connector(
        draw,
        [right_mid("B1"), (g34, right_mid("B1")[1]), (g34, left_mid("F3")[1]), left_mid("F3")],
        color=THEMES["terracotta"]["bar"],
        end_dir="right",
    )
    draw_orth_connector(
        draw,
        [right_mid("B4"), (g34 + 7, right_mid("B4")[1]), (g34 + 7, left_mid("F5", -16)[1]), left_mid("F5", -16)],
        color=THEMES["green"]["bar"],
        end_dir="right",
    )

    # Col IV -> Col V
    draw_orth_connector(
        draw,
        [right_mid("F5"), (g45, right_mid("F5")[1]), (g45, left_mid("C5")[1]), left_mid("C5")],
        color=THEMES["terracotta"]["bar"],
        end_dir="right",
    )

    # Top rail: A4 (227,656 raw QA) -> C1 (QA->CoT v0.3) cleanly above col_top
    top_rail_y = 149
    draw_orth_connector(
        draw,
        [
            right_mid("A4", -30),
            (g12 + 8, right_mid("A4", -30)[1]),
            (g12 + 8, top_rail_y),
            (g45 - 6, top_rail_y),
            (g45 - 6, left_mid("C1")[1]),
            left_mid("C1"),
        ],
        color=THEMES["purple"]["bar"],
        end_dir="right",
        label="227,656 qa_kb_raw → CoT v0.3",
        label_pos=((g12 + g45) // 2, top_rail_y),
        font=fonts["edge_label"],
        dashed=True,
    )

    # Bottom rail 1: A3 (ch2 + v_mm) -> F1 (Flash v2.3) cleanly below col_bottom
    bot_rail_1 = HEIGHT - 70
    draw_orth_connector(
        draw,
        [
            right_mid("A3", 40),
            (g12 - 10, right_mid("A3", 40)[1]),
            (g12 - 10, bot_rail_1),
            (g34 - 8, bot_rail_1),
            (g34 - 8, left_mid("F1")[1]),
            left_mid("F1"),
        ],
        color=THEMES["blue"]["bar"],
        end_dir="right",
        label="ch2 + v_mm + V1~V4 → Flash 全量池",
        label_pos=((g12 + g34) // 2 - 140, bot_rail_1),
        font=fonts["edge_label"],
        dashed=True,
    )

    # Col V -> Col VI
    draw_orth_connector(
        draw,
        [right_mid("C4", 18), (g56 - 7, right_mid("C4", 18)[1]), (g56 - 7, left_mid("P3")[1]), left_mid("P3")],
        color=THEMES["purple"]["bar"],
        end_dir="right",
        dashed=True,
    )
    draw_orth_connector(
        draw,
        [right_mid("C5"), (g56 + 7, right_mid("C5")[1]), (g56 + 7, left_mid("P5")[1]), left_mid("P5")],
        color=THEMES["green"]["bar"],
        end_dir="right",
    )

    # Bottom rail 2: P2 -> gutter g56 -> bottom rail -> B5 (never passes over Col VI cards!)
    bot_rail_2 = HEIGHT - 36
    draw_orth_connector(
        draw,
        [
            left_mid("P2", 26),
            (g56 - 12, left_mid("P2", 26)[1]),
            (g56 - 12, bot_rail_2),
            (col_centers_x[2], bot_rail_2),
            bottom_mid("B5"),
        ],
        color=ARROW_ACCENT,
        end_dir="up",
        label="Pro 50/30/20 增量 CPT 回归 B300 发射链",
        label_pos=((col_centers_x[2] + g56) // 2, bot_rail_2),
        font=fonts["edge_label"],
        dashed=True,
    )

    footer_text = "CPT-SFT · B300 Qwen3.5-27B Lite / Flash / Pro + QA→CoT v0.3 Architecture · 2026-11-18"
    draw.text((46, HEIGHT - 38), footer_text, font=fonts["col_tag"], fill=INK_MUTED)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG", optimize=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    render_diagram(args.output)
    print(f"Rendered workflow diagram to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
