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
HEIGHT = 2320

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
        title="源资产与校准底盘",
        tag="Lake+B300",
        subboxes=[
            SubBoxSpec(
                title="现役原始资产总账 (三档同源共用)",
                cards=[
                    CardSpec(
                        cid="A1",
                        theme="blue",
                        title="KB 语料与实体图片库",
                        badge="263万+8,692",
                        code_line="kb/(362M) + entities(8,692) + v-bank",
                        lines=[
                            "chunk_kb/kb_gap/kb_sports 含西甲子展开",
                            "v系263万张唯一实拍图 100% 落袋免验",
                        ],
                    ),
                    CardSpec(
                        cid="A2",
                        theme="blue",
                        title="pt/ 58,092 + V1~V4",
                        badge="五大源池",
                        code_line="kb_mm 15k + v_mm 22k + sch 11.5k + ch2",
                        lines=[
                            "Lite/Flash/Pro 完全共用同套原始底池",
                            "V1~V4 去重进回放 · 4,715 条 v2 RAG 归档",
                        ],
                    ),
                    CardSpec(
                        cid="A3",
                        theme="purple",
                        title="qa_kb_raw 前置去重",
                        badge="227k->147k",
                        code_line="清理 80,594 重复 · 147,062 唯一 QA",
                        lines=[
                            "synonsense(6)->synthesis · insufficient 19%",
                            "独立 OOD 金标族零交叉 (eval_overlap=0)",
                        ],
                    ),
                ],
            ),
            SubBoxSpec(
                title="8×B300 校准发射底盘 & 排险",
                cards=[
                    CardSpec(
                        cid="A4",
                        theme="green",
                        title="hw_b300.sh 校准配方",
                        badge="8192 序列",
                        code_line="TP4 DP2(禁TP8) · TE sm_103 + no fusion",
                        lines=[
                            "Lite: MBS1/GBS4 (32.8K/step 修 7-step)",
                            "Flash/Pro: MBS2/GBS16 (131K/step 保8K文)",
                        ],
                    ),
                    CardSpec(
                        cid="A5",
                        theme="amber",
                        title="archive/ & 发射门禁",
                        badge="默认dry-run",
                        code_line="run_cpt_27b_b300.sh <lite|flash|pro>",
                        lines=[
                            "旧 H200 / QSA 已归档 (9/9 回归完整保留)",
                            "无 --launch 绝不执行 · 偏离预期步数停机",
                        ],
                    ),
                ],
            ),
        ],
    ),
    ColumnSpec(
        num="II",
        title="CoT v0.4 知识内化",
        tag="gen_cot",
        subboxes=[
            SubBoxSpec(
                title="可变性二分与双轨八门",
                cards=[
                    CardSpec(
                        cid="C1",
                        theme="purple",
                        title="rule vs entity 二分",
                        badge="核心改造",
                        code_line="split_evidence_by_mutability() 去[Sx]",
                        lines=[
                            "可变规则/流程 -> user 给定规则 [R1..Rm]",
                            "稳定实体/IP -> 剥离 user 经 reasoning 直出",
                        ],
                    ),
                    CardSpec(
                        cid="C2",
                        theme="purple",
                        title="v0.4 八道内化门控",
                        badge="budget>=2048",
                        code_line="qwen3.8-flash · verify_cot_v04_8gates",
                        lines=[
                            "门2 零[Sx]/根据材料 · 门3 锚点覆盖 >=0.65",
                            "门4 规则+实体同句否定辨析 · Smoke 60 人检",
                        ],
                    ),
                ],
            ),
            SubBoxSpec(
                title="Thinking-OFF 与 CPT 防护",
                cards=[
                    CardSpec(
                        cid="C3",
                        theme="green",
                        title="Clean Direct Twins",
                        badge=">=80% 直答",
                        code_line="qa_direct_clean_twins + 7:3 混配",
                        lines=[
                            "通过 CoT 派生浓缩锚点直答，防关思考崩溃",
                            "失败降级 <=20% 占直答，错迹存 DPO 负池",
                        ],
                    ),
                    CardSpec(
                        cid="C4",
                        theme="amber",
                        title="CPT 百科散文净化",
                        badge="零ChatML",
                        code_line="sanitize_qa_for_cpt() -> midtrain_prose",
                        lines=[
                            "进 CPT 前剥离 <think>/system/user 角色外壳",
                            "QA 散文硬锁 <=15% 且零 SFT/eval 族泄漏",
                        ],
                    ),
                ],
            ),
        ],
    ),
    ColumnSpec(
        num="III",
        title="生图 v2 按域多图",
        tag="image_v2",
        subboxes=[
            SubBoxSpec(
                title="8,692 实体按域分区路由",
                cards=[
                    CardSpec(
                        cid="V1",
                        theme="amber",
                        title="停跑与节流减半",
                        badge="8.7万->3万",
                        code_line="logo(3426)+audit(423)+doc(43) -> 停",
                        lines=[
                            "audit 用 263万 v-bank · move 抽象光效节流75%",
                            "kb_sports 西甲式联赛->俱乐部->队徽球衣展开",
                        ],
                    ),
                    CardSpec(
                        cid="V2",
                        theme="green",
                        title="I2I 保真 & Flag 红线",
                        badge="禁纯T2I",
                        code_line="dex(mm=T)->I2I · flag/geo->EditOnly",
                        lines=[
                            "flag 无矢量底图标为 pending 禁纯 T2I",
                            "brand 解剖特写上 VLM + OCR/条数计数校验",
                        ],
                    ),
                ],
            ),
            SubBoxSpec(
                title="v1 回收与多图单 Doc 打包",
                cards=[
                    CardSpec(
                        cid="V3",
                        theme="purple",
                        title="gen_kb_v1 三路回收",
                        badge="9,749张",
                        code_line="recycle_gen_kb_v1() 三分处置",
                        lines=[
                            "干净 brand 晋升正池 · 畸变配原图 -> v1_neg",
                            "其余冗余 audit 废片就地剪枝，零浪费",
                        ],
                    ),
                    CardSpec(
                        cid="V4",
                        theme="blue",
                        title="多图单 Doc + CF 负例",
                        badge="repeat==1",
                        code_line="Top 2~4 图 -> ch_v2_multi_img_doc",
                        lines=[
                            "禁止 1 doc 拆 10 行 (doc_text_repeat==1)",
                            "未过 OCR/计数错图配对转 ch_v2_cf_pair 负例",
                        ],
                    ),
                ],
            ),
        ],
    ),
    ColumnSpec(
        num="IV",
        title="Phase B · Flash 骨干",
        tag="run_flash",
        subboxes=[
            SubBoxSpec(
                title="单阶段 WSD Mid-Training (30/35/15/20)",
                cards=[
                    CardSpec(
                        cid="F1",
                        theme="blue",
                        title="去冗余统一建包",
                        badge="8K+2x限",
                        code_line="build_mm_wiki_cpt.py --mode midtrain",
                        lines=[
                            "30%域KB多图 + 35%实拍 + 15%CF + 20%散文",
                            "合成图<=25% · 零raw <think> · 零家族泄漏",
                        ],
                    ),
                    CardSpec(
                        cid="F2",
                        theme="terracotta",
                        title="单阶段 WSD 训演",
                        badge="MBS2/GBS16",
                        code_line="run_cpt_27b_b300.sh flash (WSD 8e-6)",
                        lines=[
                            "砍掉两阶段断点与独立风格 SFT，单程收敛",
                            "swift export --to_hf 输出统一 BF16 底座",
                        ],
                    ),
                ],
            ),
            SubBoxSpec(
                title="单轮双模 SFT & 验收",
                cards=[
                    CardSpec(
                        cid="F3",
                        theme="terracotta",
                        title="V1~V4 + 7:3 统一 SFT",
                        badge="单轮收敛",
                        code_line="refactor_v1_v4 + megatron sft (8192)",
                        lines=[
                            "Think-ON(70%) + Clean Direct Twins(30%)",
                            "砍掉独立风格 SFT 与 Teacher DPO 防冲突",
                        ],
                    ),
                    CardSpec(
                        cid="F4",
                        theme="green",
                        title="双模上线验收门",
                        badge="ON / OFF",
                        code_line="enable_thinking=true & false 双测",
                        lines=[
                            "同时验证带思考高精度与免思考低延迟零崩",
                            "通过后冻结为 Lite 对照与 Pro 增强基线",
                        ],
                    ),
                ],
            ),
        ],
    ),
    ColumnSpec(
        num="V",
        title="Phase A · Lite 切片",
        tag="training/lite",
        subboxes=[
            SubBoxSpec(
                title="同源按比例切片 (告别 7-step)",
                cards=[
                    CardSpec(
                        cid="L1",
                        theme="amber",
                        title="7-step 崩溃根因修复",
                        badge="MBS1/GBS4",
                        code_line="slice_flash_to_lite.py --ratio 0.15",
                        lines=[
                            "旧95% sch仅150tok/行，524K/step 7步刷完",
                            "现切Flash 15%，32.8K/step->100~150步",
                        ],
                    ),
                    CardSpec(
                        cid="L2",
                        theme="blue",
                        title="val0 整家族零分裂",
                        badge="compat链",
                        code_line="bundle_lite/lite_train.jsonl (val0)",
                        lines=[
                            "维持 30/35/15/20 与合成/QA 散文比例上限",
                            "兼容 LITE_MODE=sch_legacy 历史脚本回退",
                        ],
                    ),
                ],
            ),
            SubBoxSpec(
                title="快速对照与双模保留门",
                cards=[
                    CardSpec(
                        cid="L3",
                        theme="terracotta",
                        title="Lite 校准短跑 + OOD",
                        badge="8e-6 恒定",
                        code_line="run_cpt_27b_b300.sh lite -> Arm A/B",
                        lines=[
                            "SAVE_STEPS=40 留档，2ep constant 8e-6",
                            "CPT+OOD vs Base+OOD 隔离纯增益对照",
                        ],
                    ),
                    CardSpec(
                        cid="L4",
                        theme="green",
                        title="步数与防崩预警门",
                        badge="<60步警",
                        code_line="projected_steps + Think-ON/OFF 双检",
                        lines=[
                            "投影步数 <60 自动预警调小 GBS 或扩比例",
                            "确认净增益且无 Think-OFF 崩塌放行 Flash",
                        ],
                    ),
                ],
            ),
        ],
    ),
    ColumnSpec(
        num="VI",
        title="Phase C · Pro 演化",
        tag="training/pro",
        subboxes=[
            SubBoxSpec(
                title="同源 4 项程序增强 (零新原始数据)",
                cards=[
                    CardSpec(
                        cid="P1",
                        theme="purple",
                        title="密度筛选 + 跨模态对齐",
                        badge="enhance",
                        code_line="curate_high_value + alignment_audit",
                        lines=[
                            "提权长尾未闭合/多图CF，裁剪低质水词行",
                            "关联 v2 多图与 CoT v0.4 实体家族锚点",
                        ],
                    ),
                    CardSpec(
                        cid="P2",
                        theme="terracotta",
                        title="WiSE-FT & Think-OFF 门",
                        badge="0.8C+0.2B",
                        code_line="evaluate_canary_and_wise_ft() 6e-6",
                        lines=[
                            "通用掉>2pp 自动生成 0.8*CPT+0.2*Base",
                            "Think-OFF 掉>2pp 或差值>8pp 触 OPD 蒸馏",
                        ],
                    ),
                    CardSpec(
                        cid="P3",
                        theme="purple",
                        title="学生DPO + OPD 蒸馏",
                        badge="非Teacher",
                        code_line="pro_dpo_kto_pairs + pro_opd_bsd_seeds",
                        lines=[
                            "CoT 废迹转学生 DPO/KTO + Datawheel 补采",
                            "OPD Think-ON->OFF 蒸馏拉齐免思考性能",
                        ],
                    ),
                ],
            ),
            SubBoxSpec(
                title="多轮错例闭环与单调账本",
                cards=[
                    CardSpec(
                        cid="P4",
                        theme="amber",
                        title="error_cluster_miner",
                        badge="不动权重先",
                        code_line="missing_index/mapping_drift -> 补索引",
                        lines=[
                            "50%新核验缺口 + 30%已验收 + 20%V回放 5e-6",
                            "固定权重先验索引与路由，仅混淆类动权重",
                        ],
                    ),
                    CardSpec(
                        cid="P5",
                        theme="green",
                        title="多轮单调账本",
                        badge="sft_overlap=0",
                        code_line="iteration_ledger.json (--prev-ledger)",
                        lines=[
                            "跨 round_id 累计已用 doc/template 家族",
                            "全周期断言评测与异模板 SFT 绝对零泄漏",
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
    title = "8×B300 Qwen3.5-27B 同源 Lite / Flash / Pro + CoT v0.4 & 生图 v2 架构图"
    subtitle = (
        "AutoVision 8×B300 (8192) · Flash 3步骨干 (30/35/15/20, QA散文<=15%, 合成图<=25%) · "
        "Lite 15%切片 (MBS1/GBS4 修7-step) · Pro 同源4增强 + Think-OFF OPD蒸馏"
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
    col_bottom = HEIGHT - 140

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

    # Col I -> Col II: 147k dedup QA -> CoT v0.4 split
    draw_orth_connector(
        draw,
        [right_mid("A3"), (g12, right_mid("A3")[1]), (g12, left_mid("C1")[1]), left_mid("C1")],
        color=THEMES["purple"]["bar"],
        end_dir="right",
    )

    # Top rail: A1 (kb_entities 8,692 + kb_sports) -> V1 (ImageGen v2) cleanly above col_top
    top_rail_y = 148
    draw_orth_connector(
        draw,
        [
            right_mid("A1", -16),
            (g12 + 8, right_mid("A1", -16)[1]),
            (g12 + 8, top_rail_y),
            (g23 - 6, top_rail_y),
            (g23 - 6, left_mid("V1")[1]),
            left_mid("V1"),
        ],
        color=THEMES["amber"]["bar"],
        end_dir="right",
        label="entities+sports → 生图v2",
        label_pos=((g12 + g23) // 2 + 20, top_rail_y),
        font=fonts["edge_label"],
        dashed=True,
    )

    # Col II -> Col IV (top rail 2: C4 qa_midtrain_prose -> F1)
    draw_orth_connector(
        draw,
        [
            right_mid("C4"),
            (g23 - 8, right_mid("C4")[1]),
            (g23 - 8, top_rail_y),
            (g34 - 10, top_rail_y),
            (g34 - 10, left_mid("F1", 18)[1]),
            left_mid("F1", 18),
        ],
        color=THEMES["amber"]["bar"],
        end_dir="right",
        label="qa_prose<=15% (零<think>)",
        label_pos=((g23 + g34) // 2 + 10, top_rail_y),
        font=fonts["edge_label"],
        dashed=True,
    )

    # Col III -> Col IV: v2 Multi-Img & CF pairs -> Flash Mid-Training
    draw_orth_connector(
        draw,
        [right_mid("V4"), (g34, right_mid("V4")[1]), (g34, left_mid("F1", -14)[1]), left_mid("F1", -14)],
        color=THEMES["blue"]["bar"],
        end_dir="right",
    )
    # Separate bottom tracks so zero collinear overlap ever happens
    bot_rail_0 = HEIGHT - 122
    bot_rail_1 = HEIGHT - 90
    bot_rail_2 = HEIGHT - 56

    draw_orth_connector(
        draw,
        [right_mid("C3", 10), (g23 + 12, right_mid("C3", 10)[1]), (g23 + 12, bot_rail_0), (g34 - 6, bot_rail_0), (g34 - 6, left_mid("F3")[1]), left_mid("F3")],
        color=THEMES["green"]["bar"],
        end_dir="right",
        label="Clean Direct Twins -> 统一 SFT",
        label_pos=((g23 + g34) // 2 + 12, bot_rail_0),
        font=fonts["edge_label"],
        dashed=True,
    )

    # Bottom rail 1: A2 (pt 58k + v-bank) -> F2 cleanly below bot_rail_0
    draw_orth_connector(
        draw,
        [
            right_mid("A2", 20),
            (g12 - 10, right_mid("A2", 20)[1]),
            (g12 - 10, bot_rail_1),
            (g34 - 14, bot_rail_1),
            (g34 - 14, left_mid("F2")[1]),
            left_mid("F2"),
        ],
        color=THEMES["blue"]["bar"],
        end_dir="right",
        label="同源 pt 58k + 263万 v-bank → Flash WSD",
        label_pos=((g12 + g34) // 2 - 200, bot_rail_1),
        font=fonts["edge_label"],
        dashed=True,
    )

    # Col IV -> Col V: Flash Mid-Training pool -> Lite 15% Stratified Slice
    draw_orth_connector(
        draw,
        [right_mid("F1"), (g45, right_mid("F1")[1]), (g45, left_mid("L1")[1]), left_mid("L1")],
        color=THEMES["amber"]["bar"],
        end_dir="right",
    )

    # Col IV -> Col VI (top rail above Col V: Flash raw pool -> Pro enhance_flash_to_pro + F4 -> P4)
    draw_orth_connector(
        draw,
        [
            right_mid("F1", -24),
            (g45 + 10, right_mid("F1", -24)[1]),
            (g45 + 10, top_rail_y),
            (g56 - 6, top_rail_y),
            (g56 - 6, left_mid("P1")[1]),
            left_mid("P1"),
        ],
        color=THEMES["purple"]["bar"],
        end_dir="right",
        label="与 Flash 共用同套原始源池 → Pro 4项程序增强",
        label_pos=((g45 + g56) // 2, top_rail_y),
        font=fonts["edge_label"],
    )
    draw_orth_connector(
        draw,
        [
            right_mid("F4", 12),
            (g45 + 10, right_mid("F4", 12)[1]),
            (g45 + 10, bot_rail_1),
            (g56 - 8, bot_rail_1),
            (g56 - 8, left_mid("P4")[1]),
            left_mid("P4"),
        ],
        color=THEMES["terracotta"]["bar"],
        end_dir="right",
        label="线上错例与双模评测 → Pro 闭环",
        label_pos=((g45 + g56) // 2, bot_rail_1),
        font=fonts["edge_label"],
        dashed=True,
    )

    # Bottom rail 2: P2 -> gutter g56 -> bottom rail -> A5 (cleanly below Cols and above footer!)
    draw_orth_connector(
        draw,
        [
            left_mid("P2", 26),
            (g56 - 14, left_mid("P2", 26)[1]),
            (g56 - 14, bot_rail_2),
            (col_centers_x[0], bot_rail_2),
            bottom_mid("A5"),
        ],
        color=ARROW_ACCENT,
        end_dir="up",
        label="Pro WSD (pro) + 多轮 50/30/20 增量 (pro_inc) 回归 B300 统一发射底盘",
        label_pos=((col_centers_x[0] + g56) // 2, bot_rail_2),
        font=fonts["edge_label"],
        dashed=True,
    )

    footer_text = "CPT-SFT v2.0 · Single-Source Lite (15%) / Flash (3-Step) / Pro (4-Upgrade + Ledger) · CoT v0.4 & ImageGen v2 · 2026-11-19"
    draw.text((46, HEIGHT - 32), footer_text, font=fonts["col_tag"], fill=INK_MUTED)

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
