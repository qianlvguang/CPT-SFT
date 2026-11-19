#!/usr/bin/env python3
"""QA->CoT v0.4 Knowledge Internalization & v0.3 RAG-Citation Bridging Generator.

Implements `docs/training_plans/flash/qa_cot_plan.md` (v0.4 + v0.3 fallback):
  v0.4 ("Knowledge Internalization", default `--cot-mode internalized`):
    1. Pre-deduplicates `qa_kb_raw.jsonl` on canonical `(question, answer)`
       (matching the 227,656 -> 147,062 unique QA asset reduction).
    2. Normalizes typos (`synonsense` -> `synthesis`, `version` -> `condition`).
    3. Splits sources & distractors by knowledge mutability (`doc_kind` + lexical cues):
         - Mutable rules / thresholds / procedures -> `user` turn (`[R1]..[Rm]`)
         - Immutable Entity / IP / Flag / Brand facts -> removed from `user` turn
           and internalized via first-person `reasoning` (entity distractor discrimination
           embedded as contrastive analysis clauses).
    4. Runs Teacher (`qwen3.8-flash`, thinking >= 2048) or `--mode mock` with
       `prompts/qa_cot_bridge_v04.md`.
    5. Enforces 8 v0.4 programmatic quality gates (replacing G2/G3/G4 RAG citation gates
       with Layered Entity Anchor Coverage >= 0.65, Anti-RAG-Leak Ban, and Dual-Track
       Distractor Exclusion Polarity).

  v0.3 (`--cot-mode rag_citation`, preserved for A/B baseline comparison):
    Retains the original `[S1]..[Sn]` distractor shuffling, single-pass regex remapping,
    and citation precision/recall/quote-substring gates (`verify_cot_8gates`).

  Shared Post-Processing:
    - Failed items (after retries) clear `reasoning=""` and set `enable_thinking=false`
      into `qa_direct_fallback.jsonl` while exporting bad traces to `qa_dpo_rejected_cots.jsonl`.
    - Exports `qa_sft_mixed_8_2.jsonl` (`8:2` CoT:Direct mix) and stratified `--smoke-sample`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_PATH = THIS_DIR / "prompts" / "qa_cot_bridge_v03.md"
DEFAULT_PROMPT_V04_PATH = THIS_DIR / "prompts" / "qa_cot_bridge_v04.md"

CANONICAL_TASK_TYPES = ("fact", "condition", "comparison", "synthesis", "insufficient", "audit")
TASK_TYPE_ALIASES: dict[str, str] = {
    "synonsense": "synthesis",
    "version": "condition",
}

# Dynamic length budgets per v0.3/v0.4 spec: (hard_min_chars, soft_max_chars)
LENGTH_BUDGETS: dict[str, tuple[int, int]] = {
    "fact": (180, 300),
    "condition": (350, 600),
    "comparison": (350, 600),
    "insufficient": (350, 600),
    "synthesis": (500, 900),
    "audit": (500, 900),
}

DEFAULT_SYSTEM_PROMPT = (
    "你是电商合规与知识产权（IP）审核助手。严格基于给定材料进行高精度客观裁决："
    "明确列出命中规则、关键限定与适用边界，事实主张紧贴 [Sx] 引用；"
    "证据不足时明确指出缺失条件并拒绝强判，严禁编造来源、条号或链接。"
)

DEFAULT_V04_SYSTEM_PROMPT = (
    "你是电商合规与知识产权（IP）审核专家。请结合自身内化的品牌/IP/旗帜/设计要素本体知识，"
    "并严格遵从用户本次给定的动态审核规则与但书条件进行高精度客观裁决；"
    "若关键事实或授权状态缺失，须明确指出缺失条件并拒绝强判，严禁臆造未给规则。"
)

CITATION_TAG_RE = re.compile(r"\[(S\d+)\]")
RULE_TAG_RE = re.compile(r"\[(R\d+)\]")
THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)
RIGID_HEADER_RE = re.compile(
    r"(?m)^\s*(?:#{1,6}\s+|【(?:步骤|第[一二三四五六\d]+步|思考过程|结论)】)"
)
COUNTERFACTUAL_BRANCH_RE = re.compile(r"(?:若|如果|假如|假设).{2,60}(?:则|将|构成|适用|属于|触发|判定)")
MISSING_SLOT_RE = re.compile(r"(?:缺失|未提供|未披露|缺少|未规定|不足以|缺乏|无法确认|尚不明确)")

# v0.4 Mutability Classification & Anti-RAG Leak Guardrails
RAG_LEAK_RE = re.compile(
    r"(?:\[S\d+\]|根据(?:检索|上述)?材料|依据(?:上述|检索)材料|如上文(?:片段|来源)|来源\s*\[|检索结果显示|上下文(?:片段)?提到)"
)
EXCLUSION_CUE_RE = re.compile(r"(?:排除|不适用|不相符|不匹配|无关|无需适用|不同于|区分|并非)")
RULE_STRONG_RE = re.compile(
    r"(?:如果|若|当).{1,40}(?:则|应当|必须|禁止|构成|不予|放行|判定|拦截|下架|适用|优先于)"
    r"|(?:应当|必须|不得|禁止|豁免|不予|优先适用|受理细则|认定标准|处置边界|审核细则|通则|抽检标准|管理规范|第[一二三四五六七八九十\d]+条)"
)
ENTITY_STRONG_RE = re.compile(
    r"(?:角色设定|图鉴|本体|形象|视觉特征|色序|宽高比|纵横比|体毛|电气囊|耳尖|尾巴|三道杠|双肾|眼罩|锁扣|缝线|盾徽|麦穗|五角星|旗面|轮廓)"
)

# Gate 8 Style three-ban blacklists
ANTI_HINDSIGHT_RE = re.compile(
    r"(?:参考答案|标准答案|给定答案|已知结论|根据答案|题目给出的答案|如答案所述|为了得出该结论)"
)
ANTI_PLEASANTRY_RE = re.compile(
    r"(?:您好|很高兴为您|希望以上|祝您|亲，|随时提问|请问还有|感谢您的)"
)
ANTI_HEDGING_RE = re.compile(
    r"(?:大概也许|我个人认为|凭直觉|仅供参考|差不多吧|盲猜|随口一说)"
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_task_type(raw_task_type: Any) -> tuple[str, str]:
    raw_str = str(raw_task_type or "fact").strip().lower()
    mapped = TASK_TYPE_ALIASES.get(raw_str, raw_str)
    if mapped not in CANONICAL_TASK_TYPES:
        mapped = "synthesis"
    return mapped, raw_str


def _remap_citation_tags(text: str, key_map: dict[str, str]) -> str:
    if not text or not key_map:
        return text
    return CITATION_TAG_RE.sub(
        lambda m: f"[{key_map.get(m.group(1), m.group(1))}]",
        text,
    )


def _normalize_sources_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Extracts gold sources from either `sources` list or `evidence` fallback."""
    sources = record.get("sources")
    if isinstance(sources, list) and sources:
        out = []
        for idx, src in enumerate(sources, start=1):
            s = dict(src)
            s["source_key"] = str(s.get("source_key") or f"S{idx}")
            s["doc_id"] = str(s.get("doc_id") or f"doc_{idx}")
            s["title"] = str(s.get("title") or s["doc_id"])
            s["version"] = str(s.get("version") or "v1")
            s["locator"] = str(s.get("locator") or "全文")
            s["text"] = str(s.get("text") or "").strip()
            s["content_sha256"] = str(s.get("content_sha256") or sha256_text(s["text"]))
            s["is_distractor"] = False
            out.append(s)
        return out
    return []


def _extract_distractors(
    record: dict[str, Any],
    *,
    global_pool: list[dict[str, Any]],
    gold_doc_ids: set[str],
    rng: random.Random,
    max_distractors: int = 2,
) -> list[dict[str, Any]]:
    raw_d = record.get("distractors")
    if isinstance(raw_d, list) and raw_d:
        out: list[dict[str, Any]] = []
        for idx, d in enumerate(raw_d[:max_distractors], start=1):
            item = dict(d)
            item["doc_id"] = str(item.get("doc_id") or f"dist_{idx}")
            item["title"] = str(item.get("title") or item["doc_id"])
            item["version"] = str(item.get("version") or "2024-旧版")
            item["locator"] = str(item.get("locator") or "第1条")
            item["text"] = str(item.get("text") or "").strip()
            item["content_sha256"] = str(item.get("content_sha256") or sha256_text(item["text"]))
            item["is_distractor"] = True
            if item["text"]:
                out.append(item)
        return out

    cate = str(record.get("cate_id") or "")
    same_cate = [
        d for d in global_pool if d.get("doc_id") not in gold_doc_ids and d.get("cate_id") == cate
    ]
    others = [d for d in global_pool if d.get("doc_id") not in gold_doc_ids and d.get("cate_id") != cate]
    candidates = same_cate if same_cate else others
    if not candidates:
        return []
    picks = rng.sample(candidates, k=min(max_distractors, len(candidates)))
    out = []
    for d in picks:
        item = dict(d)
        item["is_distractor"] = True
        out.append(item)
    return out


def inject_distractors_and_remap(
    record: dict[str, Any],
    *,
    distractor_pool: list[dict[str, Any]] | None = None,
    inject_distractors: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Shuffles gold sources + distractors into S1..Sn and remaps answer/citations."""
    rng = random.Random(seed)
    gold_sources = _normalize_sources_from_record(record)
    gold_doc_ids = {s["doc_id"] for s in gold_sources}

    distractors: list[dict[str, Any]] = []
    if inject_distractors:
        distractors = _extract_distractors(
            record,
            global_pool=distractor_pool or [],
            gold_doc_ids=gold_doc_ids,
            rng=rng,
        )

    combined = list(gold_sources) + list(distractors)
    rng.shuffle(combined)

    key_map: dict[str, str] = {}
    remapped_sources: list[dict[str, Any]] = []
    source_map: dict[str, dict[str, Any]] = {}
    distractor_keys: list[str] = []

    for new_idx, src in enumerate(combined, start=1):
        new_key = f"S{new_idx}"
        old_key = str(src.get("source_key") or "")
        copied = dict(src)
        copied["source_key"] = new_key
        if not copied.get("is_distractor") and old_key:
            key_map[old_key] = new_key
        if copied.get("is_distractor"):
            distractor_keys.append(new_key)
        remapped_sources.append(copied)
        source_map[new_key] = copied

    raw_citations = [str(c).strip() for c in (record.get("citations") or []) if str(c).strip()]
    remapped_citations = [key_map.get(c, c) for c in raw_citations]
    remapped_answer = _remap_citation_tags(str(record.get("answer") or "").strip(), key_map)

    remapped_evidence: list[dict[str, Any]] = []
    for ev in record.get("evidence") or []:
        ev_copy = dict(ev)
        old_ev_key = str(ev_copy.get("source_key") or "").strip()
        new_ev_key = key_map.get(old_ev_key, old_ev_key)
        ev_copy["source_key"] = new_ev_key
        if new_ev_key in source_map:
            ev_copy["content_sha256"] = source_map[new_ev_key]["content_sha256"]
        remapped_evidence.append(ev_copy)

    bundle_lines = []
    for s in remapped_sources:
        bundle_lines.append(
            f"来源 [{s['source_key']}]\n"
            f"标题：{s['title']}\n"
            f"版本：{s['version']}\n"
            f"定位：{s['locator']}\n"
            f"正文：{s['text']}"
        )

    return {
        "source_bundle": "\n\n".join(bundle_lines),
        "source_map": source_map,
        "remapped_sources": remapped_sources,
        "gold_citations": remapped_citations,
        "distractor_keys": distractor_keys,
        "gold_answer": remapped_answer,
        "evidence": remapped_evidence,
        "key_mapping": key_map,
    }


def strip_rag_citations(text: str) -> str:
    """Removes `[S1]` style RAG citation tags and cleans orphaned punctuation spacing."""
    if not text:
        return ""
    cleaned = CITATION_TAG_RE.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"([。！？；])[，、]*([。！？；])", r"\1", cleaned)
    return cleaned.strip()


def dedup_qa_records(
    qa_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deduplicates raw QA records (matching 227,656 -> 147,062 unique reduction)."""
    best_by_sig: dict[str, dict[str, Any]] = {}
    ordered_sigs: list[str] = []
    for rec in qa_records:
        q_norm = re.sub(r"\s+", "", str(rec.get("question") or ""))
        a_norm = re.sub(r"\s+", "", strip_rag_citations(str(rec.get("answer") or "")))
        task_norm, _ = normalize_task_type(rec.get("task_type"))
        sig = sha256_text(f"{task_norm}||{q_norm}||{a_norm}")
        score = len(rec.get("sources") or []) * 10 + len(rec.get("distractors") or [])
        if sig not in best_by_sig:
            best_by_sig[sig] = rec
            ordered_sigs.append(sig)
        else:
            prev = best_by_sig[sig]
            prev_score = len(prev.get("sources") or []) * 10 + len(prev.get("distractors") or [])
            if score > prev_score:
                best_by_sig[sig] = rec
    unique_records = [best_by_sig[s] for s in ordered_sigs]
    return unique_records, {
        "raw_input_rows": len(qa_records),
        "unique_input_rows": len(unique_records),
        "dedup_dropped_rows": len(qa_records) - len(unique_records),
    }


def classify_slice_mutability(src: dict[str, Any]) -> str:
    """Binary mutability split: `'rule'` (mutable user rules) vs `'entity'` (stable IP/entity)."""
    explicit = str(src.get("doc_kind") or src.get("knowledge_type") or src.get("kind") or "").strip().lower()
    if explicit in {"entity", "ip", "anime", "brand", "flag", "symbol", "mascot", "dex", "craft"}:
        return "entity"
    if explicit in {"rule", "policy", "audit_rule", "proc", "law", "compliance_rule"}:
        return "rule"

    title = str(src.get("title") or "")
    version = str(src.get("version") or "")
    text = str(src.get("text") or "")
    if any(tok in title or tok in version for tok in ("图鉴", "本体", "要素解剖", "人物设定", "角色设定", "视觉规范", "旗帜规范")):
        return "entity"

    rule_hits = len(RULE_STRONG_RE.findall(f"{title} {text}"))
    entity_hits = len(ENTITY_STRONG_RE.findall(f"{title} {text}"))
    if rule_hits > entity_hits:
        return "rule"
    if entity_hits > 0:
        return "entity"
    return "rule"


def _clean_entity_subject(title: str) -> str:
    subj = re.sub(
        r"(?:角色设定图鉴|角色图鉴|人物本体特征|核心要素|视觉规范|图鉴|本体|规范|要素)$",
        "",
        title.strip(),
    ).strip()
    return subj or title.strip()


def extract_entity_anchors(src: dict[str, Any]) -> list[str]:
    """Extracts core entity anchors (explicit `anchors` preferred, lexical fallback)."""
    explicit = src.get("anchors")
    if isinstance(explicit, list) and explicit:
        cleaned = [str(a).strip() for a in explicit if str(a).strip()]
        if cleaned:
            return cleaned

    anchors: list[str] = []
    text = str(src.get("text") or "")
    title = str(src.get("title") or "")
    subj = _clean_entity_subject(title)
    if len(subj) >= 2:
        anchors.append(subj)
    for m in re.findall(r"\b\d+:\d+\b", text):
        if m not in anchors:
            anchors.append(m)
    for seg in re.split(r"[，。；：、（）\(\)\s]+", text):
        s = seg.strip()
        if 3 <= len(s) <= 12 and s not in anchors and not RULE_STRONG_RE.search(s):
            anchors.append(s)
        if len(anchors) >= 5:
            break
    return anchors[:6]


def split_evidence_by_mutability(
    record: dict[str, Any],
    *,
    distractor_pool: list[dict[str, Any]] | None = None,
    inject_distractors: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Splits sources & distractors into user-visible `[R1..Rm]` rules vs internalized entity KB."""
    rng = random.Random(seed)
    gold_sources = _normalize_sources_from_record(record)
    gold_doc_ids = {s["doc_id"] for s in gold_sources}

    distractors: list[dict[str, Any]] = []
    if inject_distractors:
        distractors = _extract_distractors(
            record,
            global_pool=distractor_pool or [],
            gold_doc_ids=gold_doc_ids,
            rng=rng,
        )

    rule_gold_raw: list[dict[str, Any]] = []
    entity_gold_raw: list[dict[str, Any]] = []
    for src in gold_sources:
        item = dict(src)
        kind = classify_slice_mutability(item)
        item["mutability"] = kind
        if kind == "rule":
            rule_gold_raw.append(item)
        else:
            item["anchors"] = extract_entity_anchors(item)
            entity_gold_raw.append(item)

    rule_dist_raw: list[dict[str, Any]] = []
    entity_dist_raw: list[dict[str, Any]] = []
    for dist in distractors:
        item = dict(dist)
        kind = classify_slice_mutability(item)
        item["mutability"] = kind
        if kind == "rule":
            rule_dist_raw.append(item)
        else:
            item["subject_name"] = _clean_entity_subject(str(item.get("title") or item.get("doc_id") or ""))
            entity_dist_raw.append(item)

    rule_combined = rule_gold_raw + rule_dist_raw
    rng.shuffle(rule_combined)

    rule_source_map: dict[str, dict[str, Any]] = {}
    rule_gold_keys: list[str] = []
    rule_distractor_keys: list[str] = []
    rule_lines: list[str] = []

    for r_idx, src in enumerate(rule_combined, start=1):
        r_key = f"R{r_idx}"
        copied = dict(src)
        copied["rule_key"] = r_key
        copied["source_key"] = r_key
        if copied.get("is_distractor"):
            rule_distractor_keys.append(r_key)
        else:
            rule_gold_keys.append(r_key)
        rule_source_map[r_key] = copied
        rule_lines.append(
            f"规则 [{r_key}]《{copied['title']}》（版本 {copied['version']} · {copied['locator']}）：\n{copied['text']}"
        )

    entity_source_map: dict[str, dict[str, Any]] = {}
    entity_kb_lines: list[str] = []
    gold_entity_anchors: list[str] = []
    seen_anchors: set[str] = set()
    for e_idx, src in enumerate(entity_gold_raw, start=1):
        e_key = f"E{e_idx}"
        copied = dict(src)
        copied["entity_key"] = e_key
        copied["subject_name"] = _clean_entity_subject(str(copied.get("title") or copied["doc_id"]))
        entity_source_map[e_key] = copied
        anchors_str = "、".join(copied.get("anchors") or [])
        entity_kb_lines.append(
            f"- 实体本体({copied['subject_name']}): {copied['text']} [关键锚点: {anchors_str}]"
        )
        for a in copied.get("anchors") or []:
            if a not in seen_anchors:
                seen_anchors.add(a)
                gold_entity_anchors.append(a)

    entity_distractor_subjects: list[str] = []
    entity_dist_lines: list[str] = []
    for ed_idx, d in enumerate(entity_dist_raw, start=1):
        copied = dict(d)
        subj = str(copied.get("subject_name") or copied.get("title") or f"干扰实体{ed_idx}")
        copied["entity_key"] = f"ED{ed_idx}"
        entity_source_map[copied["entity_key"]] = copied
        if subj not in entity_distractor_subjects:
            entity_distractor_subjects.append(subj)
        entity_dist_lines.append(f"- 易混淆实体({subj}): {copied['text']}")

    question = str(record.get("question") or "").strip()
    if rule_lines:
        user_given_rules = "\n\n".join(rule_lines)
        user_prompt = f"【本次审核给定规则】\n{user_given_rules}\n\n问题：{question}"
    else:
        user_given_rules = "（无额外给定规则）"
        user_prompt = f"问题：{question}"

    clean_gold_answer = strip_rag_citations(str(record.get("answer") or "").strip())
    return {
        "user_given_rules": user_given_rules,
        "user_prompt": user_prompt,
        "entity_kb_reference": "\n".join(entity_kb_lines) if entity_kb_lines else "（本题纯属规则核验，无额外实体本体）",
        "entity_distractors_hint": "\n".join(entity_dist_lines) if entity_dist_lines else "（无易混淆实体干扰项）",
        "rule_source_map": rule_source_map,
        "rule_gold_keys": rule_gold_keys,
        "rule_distractor_keys": rule_distractor_keys,
        "entity_sources": entity_gold_raw,
        "entity_distractors": entity_dist_raw,
        "entity_source_map": entity_source_map,
        "gold_entity_anchors": gold_entity_anchors,
        "entity_distractor_subjects": entity_distractor_subjects,
        "clean_gold_answer": clean_gold_answer,
        "split_counts": {
            "rule_gold": len(rule_gold_raw),
            "entity_gold": len(entity_gold_raw),
            "rule_distractors": len(rule_dist_raw),
            "entity_distractors": len(entity_dist_raw),
        },
    }


def _char_ngrams(text: str, n: int = 4) -> list[str]:
    cleaned = re.sub(r"\s+", "", text)
    if len(cleaned) < n:
        return [cleaned] if cleaned else []
    return [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]


def compute_ngram_overlap_ratio(text_a: str, text_b: str, n: int = 4) -> float:
    ng_a = set(_char_ngrams(text_a, n=n))
    ng_b = set(_char_ngrams(text_b, n=n))
    if not ng_a or not ng_b:
        return 0.0
    return len(ng_a & ng_b) / min(len(ng_a), len(ng_b))


def compute_quote_coverage_ratio(
    reasoning: str,
    source_map: dict[str, dict[str, Any]],
    min_span: int = 12,
) -> float:
    """Computes fraction of `reasoning` characters copied verbatim from sources."""
    if not reasoning:
        return 0.0
    r_clean = re.sub(r"\s+", "", reasoning)
    if len(r_clean) < min_span:
        return 0.0

    matched = [False] * len(r_clean)
    for src in source_map.values():
        s_clean = re.sub(r"\s+", "", str(src.get("text") or ""))
        if len(s_clean) < min_span:
            continue
        for idx in range(0, len(r_clean) - min_span + 1):
            window = r_clean[idx : idx + min_span]
            if window in s_clean:
                for k in range(idx, idx + min_span):
                    matched[k] = True
    return sum(matched) / len(r_clean)


def _normalize_text_for_match(text: str) -> str:
    return re.sub(r"\s+", "", text).strip()


def verify_cot_8gates(
    *,
    reasoning: str,
    final_answer: str,
    gold_answer: str,
    task_type: str,
    answerable: bool,
    gold_citations: list[str],
    distractor_keys: list[str],
    source_map: dict[str, dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Runs all 8 v0.3 programmatic gates and returns structured pass/fail report."""
    errors: list[str] = []
    soft_warnings: list[str] = []
    gate_results: dict[str, bool] = {}

    r_text = (reasoning or "").strip()
    a_text = (final_answer or "").strip()
    g_text = (gold_answer or "").strip()

    # Gate 1: Final answer consistency & non-empty reasoning
    g1_ok = bool(r_text) and bool(a_text) and (
        _normalize_text_for_match(a_text) == _normalize_text_for_match(g_text)
    )
    gate_results["gate1_answer_consistency"] = g1_ok
    if not g1_ok:
        errors.append("gate1_answer_consistency_failed")

    # Extract citation keys used in reasoning and answer
    reasoning_cites = sorted(set(CITATION_TAG_RE.findall(r_text)))
    answer_cites = sorted(set(CITATION_TAG_RE.findall(a_text)))
    all_ctx_keys = set(source_map.keys())
    dist_set = set(distractor_keys)
    gold_set = set(gold_citations)

    # Gate 2: Citation precision (no hallucinated keys; no distractor keys in final answer)
    hallucinated = (set(reasoning_cites) | set(answer_cites)) - all_ctx_keys
    answer_leaked_dist = set(answer_cites) & dist_set
    g2_ok = (not hallucinated) and (not answer_leaked_dist)
    gate_results["gate2_citation_precision"] = g2_ok
    if hallucinated:
        errors.append(f"gate2_hallucinated_citations:{sorted(hallucinated)}")
    if answer_leaked_dist:
        errors.append(f"gate2_distractor_in_final_answer:{sorted(answer_leaked_dist)}")

    # Gate 3: Citation recall (all gold citations covered in reasoning + distractors checked)
    missing_gold = gold_set - set(reasoning_cites)
    unchecked_dist = dist_set - set(reasoning_cites)
    g3_ok = (not missing_gold) and (not unchecked_dist)
    gate_results["gate3_citation_recall"] = g3_ok
    if missing_gold:
        errors.append(f"gate3_missing_gold_citations:{sorted(missing_gold)}")
    if unchecked_dist:
        errors.append(f"gate3_unchecked_distractors:{sorted(unchecked_dist)}")

    # Gate 4: Quote exact substring, SHA-256 integrity, and clean thinking formatting
    g4_errors: list[str] = []
    if not evidence:
        g4_errors.append("empty_evidence")
    for ev in evidence:
        s_key = str(ev.get("source_key") or "").strip()
        quote = str(ev.get("quote") or "").strip()
        if s_key not in source_map:
            g4_errors.append(f"unknown_ev_key:{s_key}")
            continue
        src = source_map[s_key]
        if not quote or quote not in src["text"]:
            g4_errors.append(f"non_substring_quote:{s_key}")
        expected_sha = sha256_text(src["text"])
        if ev.get("content_sha256") and ev["content_sha256"] != expected_sha:
            g4_errors.append(f"sha256_mismatch:{s_key}")
    if THINK_TAG_RE.search(r_text):
        g4_errors.append("nested_think_tag_in_reasoning")
    if RIGID_HEADER_RE.search(r_text):
        g4_errors.append("rigid_markdown_header_in_reasoning")
    g4_ok = len(g4_errors) == 0
    gate_results["gate4_quote_substring_and_format"] = g4_ok
    if not g4_ok:
        errors.extend(f"gate4_{e}" for e in g4_errors)

    # Gate 5: Insufficient / unanswerable counterfactual branch enforcement
    g5_ok = True
    if (not answerable) or task_type == "insufficient":
        has_missing_slot = bool(MISSING_SLOT_RE.search(r_text))
        has_counterfactual = bool(COUNTERFACTUAL_BRANCH_RE.search(r_text))
        g5_ok = has_missing_slot and has_counterfactual
        if not has_missing_slot:
            errors.append("gate5_insufficient_missing_slot_not_identified")
        if not has_counterfactual:
            errors.append("gate5_insufficient_missing_counterfactual_branch")
    gate_results["gate5_insufficient_counterfactual"] = g5_ok

    # Gate 6: Anti-tautology & verbatim quote cap (<= 45%)
    ngram_overlap = compute_ngram_overlap_ratio(r_text, a_text, n=4)
    quote_ratio = compute_quote_coverage_ratio(r_text, source_map, min_span=12)
    g6_ok = (ngram_overlap <= 0.55) and (quote_ratio <= 0.45)
    gate_results["gate6_anti_tautology"] = g6_ok
    if ngram_overlap > 0.55:
        errors.append(f"gate6_high_tautology_overlap:{ngram_overlap:.3f}")
    if quote_ratio > 0.45:
        errors.append(f"gate6_high_quote_ratio:{quote_ratio:.3f}")

    # Gate 7: Dynamic length hard lower bound + soft upper bound
    min_len, soft_max_len = LENGTH_BUDGETS.get(task_type, (350, 600))
    r_len = len(r_text)
    g7_ok = r_len >= min_len
    gate_results["gate7_dynamic_length_lower"] = g7_ok
    if not g7_ok:
        errors.append(f"gate7_below_min_length:{r_len}<{min_len}")
    if r_len > soft_max_len:
        soft_warnings.append(f"gate7_soft_upper_exceeded:{r_len}>{soft_max_len}")

    # Gate 8: Cold/high-precision style forbidden-word blacklist
    g8_hits: list[str] = []
    if ANTI_HINDSIGHT_RE.search(r_text) or ANTI_HINDSIGHT_RE.search(a_text):
        g8_hits.append("hindsight_phrase")
    if ANTI_PLEASANTRY_RE.search(r_text) or ANTI_PLEASANTRY_RE.search(a_text):
        g8_hits.append("pleasantry_phrase")
    if ANTI_HEDGING_RE.search(r_text) or ANTI_HEDGING_RE.search(a_text):
        g8_hits.append("hedging_phrase")
    g8_ok = len(g8_hits) == 0
    gate_results["gate8_style_forbidden_words"] = g8_ok
    if not g8_ok:
        errors.extend(f"gate8_{h}" for h in g8_hits)

    return {
        "passed": len(errors) == 0,
        "gates": gate_results,
        "errors": errors,
        "soft_warnings": soft_warnings,
        "metrics": {
            "reasoning_chars": r_len,
            "min_chars": min_len,
            "soft_max_chars": soft_max_len,
            "ngram_overlap": round(ngram_overlap, 4),
            "quote_ratio": round(quote_ratio, 4),
            "reasoning_citations": reasoning_cites,
            "answer_citations": answer_cites,
        },
    }


def _anchor_hit(anchor: str, reasoning_clean: str) -> bool:
    norm_a = re.sub(r"[，、；：（）\(\)\s]+", "", anchor)
    if not norm_a:
        return True
    if norm_a in reasoning_clean:
        return True
    # Allow matching split multi-token anchors (e.g. "红白绿" or 2-char stems)
    parts = [p for p in re.split(r"[、，/\s]+", anchor.strip()) if len(p) >= 1]
    if len(parts) >= 2 and all(p in reasoning_clean for p in parts):
        return True
    return False


def verify_cot_v04_8gates(
    *,
    reasoning: str,
    final_answer: str,
    clean_gold_answer: str,
    task_type: str,
    answerable: bool,
    gold_entity_anchors: list[str],
    entity_sources: list[dict[str, Any]],
    entity_distractor_subjects: list[str],
    rule_gold_keys: list[str],
    rule_distractor_keys: list[str],
    rule_source_map: dict[str, dict[str, Any]],
    entity_source_map: dict[str, dict[str, Any]],
    anchor_coverage_threshold: float = 0.65,
) -> dict[str, Any]:
    """Runs v0.4 internalized 8 quality gates (entity anchor coverage + anti-RAG leak)."""
    errors: list[str] = []
    soft_warnings: list[str] = []
    gate_results: dict[str, bool] = {}

    r_text = (reasoning or "").strip()
    a_text = (final_answer or "").strip()
    g_text = (clean_gold_answer or "").strip()
    r_clean = _normalize_text_for_match(r_text)

    # Gate 1: Final answer consistency, clean `<think>` / header formatting, zero `[Sx]`
    g1_errors: list[str] = []
    if not r_text or not a_text:
        g1_errors.append("empty_reasoning_or_answer")
    if _normalize_text_for_match(a_text) != _normalize_text_for_match(g_text):
        g1_errors.append("final_answer_mismatch")
    if CITATION_TAG_RE.search(a_text):
        g1_errors.append("rag_citation_tag_in_final_answer")
    if THINK_TAG_RE.search(r_text):
        g1_errors.append("nested_think_tag_in_reasoning")
    if RIGID_HEADER_RE.search(r_text):
        g1_errors.append("rigid_markdown_header_in_reasoning")
    g1_ok = len(g1_errors) == 0
    gate_results["gate1_answer_consistency"] = g1_ok
    if not g1_ok:
        errors.extend(f"gate1_{e}" for e in g1_errors)

    # Gate 2: Anti-RAG leak & rule-reference precision (zero `[Sx]` or "根据材料")
    r_rag_leaks = RAG_LEAK_RE.findall(r_text)
    a_rag_leaks = RAG_LEAK_RE.findall(a_text)
    reasoning_rules = sorted(set(RULE_TAG_RE.findall(r_text)))
    answer_rules = sorted(set(RULE_TAG_RE.findall(a_text)))
    valid_rule_keys = set(rule_source_map.keys())
    hallucinated_rules = (set(reasoning_rules) | set(answer_rules)) - valid_rule_keys
    answer_leaked_rule_dist = set(answer_rules) & set(rule_distractor_keys)
    g2_ok = (
        not r_rag_leaks
        and not a_rag_leaks
        and not hallucinated_rules
        and not answer_leaked_rule_dist
    )
    gate_results["gate2_anti_rag_leak"] = g2_ok
    if r_rag_leaks or a_rag_leaks:
        errors.append(f"gate2_rag_leak_detected:{r_rag_leaks or a_rag_leaks}")
    if hallucinated_rules:
        errors.append(f"gate2_hallucinated_rule_keys:{sorted(hallucinated_rules)}")
    if answer_leaked_rule_dist:
        errors.append(f"gate2_distractor_rule_in_answer:{sorted(answer_leaked_rule_dist)}")

    # Gate 3: Layered entity anchor coverage >= threshold + mutable rule coverage
    matched_anchors = [a for a in gold_entity_anchors if _anchor_hit(a, r_clean)]
    anchor_cov = (
        len(matched_anchors) / len(gold_entity_anchors)
        if gold_entity_anchors
        else 1.0
    )
    missing_rules: list[str] = []
    for rk in rule_gold_keys:
        r_title = _clean_entity_subject(str(rule_source_map[rk].get("title") or ""))
        if rk not in r_text and (not r_title or _normalize_text_for_match(r_title) not in r_clean):
            missing_rules.append(rk)
    g3_ok = (anchor_cov >= anchor_coverage_threshold) and (not missing_rules)
    gate_results["gate3_entity_anchor_coverage"] = g3_ok
    if anchor_cov < anchor_coverage_threshold:
        errors.append(
            f"gate3_low_entity_anchor_coverage:{anchor_cov:.2f}<{anchor_coverage_threshold:.2f}"
        )
    if missing_rules:
        errors.append(f"gate3_missing_user_rules:{missing_rules}")

    # Gate 4: Dual-track distractor discrimination (rule exclusion + entity comparison in same clause)
    g4_errors: list[str] = []
    clauses = [c.strip() for c in re.split(r"[。！？；\n]+", r_text) if c.strip()]
    for dk in rule_distractor_keys:
        d_title = _clean_entity_subject(str(rule_source_map[dk].get("title") or ""))
        norm_title = _normalize_text_for_match(d_title) if d_title else ""
        excluded_in_clause = any(
            ((dk in c) or (norm_title and norm_title in _normalize_text_for_match(c)))
            and bool(EXCLUSION_CUE_RE.search(c))
            for c in clauses
        )
        if not excluded_in_clause:
            g4_errors.append(f"unexcluded_rule_distractor:{dk}")
    for subj in entity_distractor_subjects:
        norm_subj = _normalize_text_for_match(subj)
        distinguished_in_clause = bool(norm_subj) and any(
            (norm_subj in _normalize_text_for_match(c)) and bool(EXCLUSION_CUE_RE.search(c))
            for c in clauses
        )
        if not distinguished_in_clause:
            g4_errors.append(f"undistinguished_entity_distractor:{subj}")
    g4_ok = len(g4_errors) == 0
    gate_results["gate4_distractor_discrimination"] = g4_ok
    if not g4_ok:
        errors.extend(f"gate4_{e}" for e in g4_errors)

    # Gate 5: Insufficient counterfactual double-branch check
    g5_ok = True
    if (not answerable) or task_type == "insufficient":
        has_missing = bool(MISSING_SLOT_RE.search(r_text))
        has_cf = bool(COUNTERFACTUAL_BRANCH_RE.search(r_text))
        g5_ok = has_missing and has_cf
        if not has_missing:
            errors.append("gate5_insufficient_missing_slot_not_identified")
        if not has_cf:
            errors.append("gate5_insufficient_missing_counterfactual_branch")
    gate_results["gate5_insufficient_counterfactual"] = g5_ok

    # Gate 6: Anti-tautology & verbatim quote cap across both rules and entity KB
    ngram_overlap = compute_ngram_overlap_ratio(r_text, a_text, n=4)
    combined_sources = {**rule_source_map, **entity_source_map}
    quote_ratio = compute_quote_coverage_ratio(r_text, combined_sources, min_span=12)
    g6_ok = (ngram_overlap <= 0.55) and (quote_ratio <= 0.45)
    gate_results["gate6_anti_tautology"] = g6_ok
    if ngram_overlap > 0.55:
        errors.append(f"gate6_high_tautology_overlap:{ngram_overlap:.3f}")
    if quote_ratio > 0.45:
        errors.append(f"gate6_high_quote_ratio:{quote_ratio:.3f}")

    # Gate 7: Dynamic length lower bound + soft upper bound
    min_len, soft_max_len = LENGTH_BUDGETS.get(task_type, (350, 600))
    r_len = len(r_text)
    g7_ok = r_len >= min_len
    gate_results["gate7_dynamic_length_lower"] = g7_ok
    if not g7_ok:
        errors.append(f"gate7_below_min_length:{r_len}<{min_len}")
    if r_len > soft_max_len:
        soft_warnings.append(f"gate7_soft_upper_exceeded:{r_len}>{soft_max_len}")

    # Gate 8: Style blacklist (anti-hindsight, anti-pleasantry, anti-hedging)
    g8_hits: list[str] = []
    if ANTI_HINDSIGHT_RE.search(r_text) or ANTI_HINDSIGHT_RE.search(a_text):
        g8_hits.append("hindsight_phrase")
    if ANTI_PLEASANTRY_RE.search(r_text) or ANTI_PLEASANTRY_RE.search(a_text):
        g8_hits.append("pleasantry_phrase")
    if ANTI_HEDGING_RE.search(r_text) or ANTI_HEDGING_RE.search(a_text):
        g8_hits.append("hedging_phrase")
    g8_ok = len(g8_hits) == 0
    gate_results["gate8_style_forbidden_words"] = g8_ok
    if not g8_ok:
        errors.extend(f"gate8_{h}" for h in g8_hits)

    return {
        "passed": len(errors) == 0,
        "gates": gate_results,
        "errors": errors,
        "soft_warnings": soft_warnings,
        "metrics": {
            "reasoning_chars": r_len,
            "min_chars": min_len,
            "soft_max_chars": soft_max_len,
            "entity_anchor_coverage": round(anchor_cov, 4),
            "matched_anchors": matched_anchors,
            "ngram_overlap": round(ngram_overlap, 4),
            "quote_ratio": round(quote_ratio, 4),
            "rule_refs": reasoning_rules,
        },
    }


def render_cot_bridge_prompt(
    template_text: str,
    *,
    task_type: str,
    answerable: bool,
    distractor_keys: list[str],
    source_bundle: str,
    question: str,
    gold_answer: str,
) -> str:
    prompt = template_text
    if "```text" in template_text:
        prompt = template_text.split("```text", 1)[1].split("```", 1)[0].strip()
    min_chars, soft_max = LENGTH_BUDGETS.get(task_type, (350, 600))
    hint = ", ".join(distractor_keys) if distractor_keys else "无干扰项"
    replacements = {
        "{{task_type}}": task_type,
        "{{min_chars}}": str(min_chars),
        "{{soft_max_chars}}": str(soft_max),
        "{{answerable}}": str(answerable).lower(),
        "{{distractors_hint}}": hint,
        "{{source_bundle}}": source_bundle,
        "{{question}}": question,
        "{{gold_answer}}": gold_answer,
    }
    for k, v in replacements.items():
        prompt = prompt.replace(k, v)
    return prompt


def generate_mock_cot_response(
    *,
    task_type: str,
    answerable: bool,
    question: str,
    gold_answer: str,
    gold_citations: list[str],
    distractor_keys: list[str],
    source_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Generates a deterministic, high-density cold/objective Chinese reasoning trace."""
    gold_blocks: list[str] = []
    for k in gold_citations:
        src = source_map[k]
        gold_blocks.append(
            f"核对 [{k}]（{src['title']}，版本 {src['version']}，定位 {src['locator']}）："
            f"该条款界定了主体适用要件与边界条件，能够直接支撑核心事实要件校验。"
        )
    dist_blocks: list[str] = []
    for dk in distractor_keys:
        src = source_map[dk]
        dist_blocks.append(
            f"继续排查 [{dk}]（{src['title']}，版本 {src['version']}）："
            f"其规范对象、适用类目或版本效力和当前待核验题干约束不匹配，应显式排除该干扰条目。"
        )

    parts: list[str] = [
        f"先拆解问题约束：「{question}」属于 {task_type} 核验场景，需要锁定核心客体、适用条件及效力边界。"
    ]
    parts.extend(gold_blocks)
    parts.extend(dist_blocks)

    if (not answerable) or task_type == "insufficient":
        c1 = gold_citations[0] if gold_citations else "S1"
        c2 = gold_citations[-1] if gold_citations else c1
        parts.append(
            f"进一步核查待审要件是否闭合：当前材料存在关键前置字段未披露或缺失，尚不足以形成唯一确定结论。"
            f"执行反事实双分支推演：若补充核验信息满足第一类特征要件，则适用 [{c1}] 的规范指向特定识别分支；"
            f"如果后续核实属于普通共性外观或具备完整线下许可，则触发 [{c2}] 的豁免与非违规分支，不应予以下架处置。"
            f"由于上述决定性槽位在现行检索上下文中未提供，继续核查各条文均无法单向闭合，必须保持未定并明示缺失要件。"
        )
    elif task_type == "audit":
        parts.append(
            "按无标客体审核路径逐步校验：首先将题干陈述的视觉部件与工艺构造映射到标准认定要素，确认仅存在局部外观共性而缺乏组合性排他标识；"
            "接着检索但书与类目共性豁免条款，逐项核对是否存在翻盖结构、通用扣件或普通装饰纹样豁免情形；"
            "同时复核排除全部无关干扰条目，确认不触发高置信侵权认定门槛，最终风险边界收敛于合规放行并说明豁免依据。"
        )
    elif task_type == "synthesis":
        parts.append(
            "执行跨源效力层级与时效性仲裁：先比较不同年份版本条文与平台通用规则的管辖边界，明确“新版专项细则优先于历史通用规则”的冲突裁决顺位；"
            "再将专项细则中的保留特征判定要件串联至当前题干场景，核实变形重绘情形下只要核心组合特征完整即可建立有效关联，"
            "排除无关安全抽检类干扰标准后，所有前提链路完整闭合。"
        )
    elif task_type == "comparison":
        parts.append(
            "建立双主体对比对齐维度矩阵：第一维度核验自旗杆侧至飘扬侧的色序方向是否存在镜像反转；"
            "第二维度核对官方标准宽高比例是否存在明显差异。逐维比对后确认两项核心参数完全可分，"
            "同时排除同系竖条纹第三国干扰项，最终对比结论与依据编号一一对齐。"
        )
    elif task_type == "condition":
        parts.append(
            "重点排查前置顺序与但书限定词：明确“补正完成前不开始审查”属于硬性程序阻断条件，"
            "“已一次性告知补正”与“申请人已补正完成并核验入库”属于不同时间节点，不能跳过前置要件直接并轨进入实质审查，排除快速预审通道干扰条文后结论收敛。"
        )
    else:  # fact
        parts.append(
            "逐字对照核心实体名称、图形中央构成与外圈装饰要素，确认题干询问要点与有效来源精确匹配，且无未核验外部特征外推，推理至此收敛。"
        )

    reasoning = "".join(parts)
    min_len, _ = LENGTH_BUDGETS.get(task_type, (180, 300))
    if len(reasoning) < min_len:
        reasoning += (
            "进一步交叉核验所有已定位条款的限定词、主体范围、例外条件与排除边界，"
            "确保每一项事实主张均能回溯至已保留的有效来源编号，无跳步推论与无依据外推，整条因果核验链至此完全闭合。"
        )

    return {
        "reasoning": reasoning,
        "reasoning_citations": sorted(set(gold_citations + distractor_keys)),
        "excluded_distractors": distractor_keys,
        "final_answer": gold_answer,
    }


def render_cot_v04_prompt(
    template_text: str,
    *,
    task_type: str,
    answerable: bool,
    user_given_rules: str,
    entity_kb_reference: str,
    entity_distractors_hint: str,
    question: str,
    clean_gold_answer: str,
) -> str:
    prompt = template_text
    if "```text" in template_text:
        prompt = template_text.split("```text", 1)[1].split("```", 1)[0].strip()
    min_chars, soft_max = LENGTH_BUDGETS.get(task_type, (350, 600))
    replacements = {
        "{{task_type}}": task_type,
        "{{min_chars}}": str(min_chars),
        "{{soft_max_chars}}": str(soft_max),
        "{{answerable}}": str(answerable).lower(),
        "{{user_given_rules}}": user_given_rules,
        "{{entity_kb_reference}}": entity_kb_reference,
        "{{entity_distractors_hint}}": entity_distractors_hint,
        "{{question}}": question,
        "{{gold_answer}}": clean_gold_answer,
    }
    for k, v in replacements.items():
        prompt = prompt.replace(k, v)
    return prompt


def generate_mock_cot_v04_response(
    *,
    task_type: str,
    answerable: bool,
    question: str,
    clean_gold_answer: str,
    gold_entity_anchors: list[str],
    entity_sources: list[dict[str, Any]],
    entity_distractor_subjects: list[str],
    rule_gold_keys: list[str],
    rule_distractor_keys: list[str],
    rule_source_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Generates first-person internalized CoT without RAG tags or verbatim copy."""
    parts: list[str] = [
        f"首先拆解问题核心：{question}。此类核验需要调用参数内化的域本体认知并对齐动态约束。"
    ]

    if entity_sources:
        anchor_snippets = [f"“{a}”" for a in gold_entity_anchors]
        subjects = "、".join(_clean_entity_subject(str(e.get("title") or "")) for e in entity_sources)
        parts.append(
            f"在自身领域知识记忆中，{subjects} 的核心辨识要素包含{' 与 '.join(anchor_snippets)}等特征点，"
            "这些稳定客体特征能够直接在内部表征中完成定位与互证，无需借助外部索引引用。"
        )

    for rk in rule_gold_keys:
        r_src = rule_source_map[rk]
        parts.append(
            f"接着逐条对照用户本次给出的可适用条文 [{rk}]《{r_src['title']}》（{r_src['version']}）："
            "重点检查其中触发前置条件、时间节点与限定边界，确保动作要件一一闭合。"
        )

    for dk in rule_distractor_keys:
        d_src = rule_source_map[dk]
        parts.append(
            f"排查混入的给定条文 [{dk}]《{d_src['title']}》（{d_src['version']}）："
            "其适用场景、前置资质或版本号和当前审核客体均不相符，应当在推演中显式排除该干扰条款。"
        )

    for subj in entity_distractor_subjects:
        parts.append(
            f"同时注意将当前客体与容易混同的 {subj} 进行严格区分："
            "两者在结构特征、局部配色顺序或组件组合关系上存在本质差异，不能交叉误判。"
        )

    if (not answerable) or task_type == "insufficient":
        parts.append(
            "深入校验闭合条件时发现，决定裁决方向的关键前置字段处于未提供状态，现有信息不足以单向收敛。"
            "必须执行反事实双分支推演：若补充取证核实其命中明确标识序列且缺乏合法备案，则落入从严拦截处置路径；"
            "如果进一步核查证明其仅属于无特定指代装饰或已完成合规许可登记，则触发免予下架放行分支。"
            "因决定性事实尚未披露，只能保持悬置并明确指出缺漏要件。"
        )
    elif task_type == "audit":
        parts.append(
            "针对无标立体商品外观审核场景，严禁凭空补写未陈述细节：必须确认多重专属特征是否齐全，"
            "排查一般翻盖造型与常见通用锁扣落入共性工艺豁免边界的条件，随后收敛合规判断。"
        )
    elif task_type == "synthesis":
        parts.append(
            "在多口径并存冲突时严格遵循效力层级仲裁顺序：新版类目专项细则的约束优先于历史通用条文，"
            "结合前文已召回的实体造型组合，能够完整判断重绘状态下核心特征依然成立。"
        )
    elif task_type == "comparison":
        parts.append(
            "建立平行比对矩阵逐步校验两侧参数差异：首先逐项核查排列起止方向是否发生镜像对调，"
            "继而比较法定标准几何比例是否一致，确认核心维度相互独立可分后完成收敛。"
        )
    elif task_type == "condition":
        parts.append(
            "紧盯流程前置阻断条款的先后时序关系：通知补正仅开启等待窗口，实际补正材料核验入库"
            "才是启动实质审核的唯一刚性前置要件，不能跳过前置检查强行并轨。"
        )
    else:
        parts.append(
            "经逐项核验各项细节要素之间的结构对应关系，未发现自相矛盾或未给定约束外推，至此清晰得出确定结论。"
        )

    reasoning = "".join(parts)
    min_len, _ = LENGTH_BUDGETS.get(task_type, (180, 300))
    if len(reasoning) < min_len:
        reasoning += (
            "再次复核全部前置要件、但书限制、易混淆客体区分边界以及动态规则的适用顺位，"
            "确保结论纯粹源于内部领域本体推演与给定规则严格对齐，无任何无依据假设或跳步推论。"
        )

    return {
        "reasoning": reasoning,
        "covered_entity_anchors": list(gold_entity_anchors),
        "excluded_rules": list(rule_distractor_keys),
        "distinguished_entities": list(entity_distractor_subjects),
        "final_answer": clean_gold_answer,
    }


def call_qwen_teacher_api(
    *,
    api_base: str,
    model_id: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    thinking_budget: int,
    seed: int,
    api_key: str = "",
) -> dict[str, Any]:
    """Calls OpenAI-compatible vLLM endpoint for `qwen3.8-flash` with thinking ON."""
    url = api_base.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "thinking_budget": max(2048, thinking_budget),
        },
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.load(resp)
    msg = body["choices"][0]["message"]
    content = str(msg.get("content") or "").strip()
    native_reasoning = str(msg.get("reasoning_content") or "").strip()

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object returned by teacher API")
    parsed = json.loads(content[start : end + 1])
    if not parsed.get("reasoning") and native_reasoning:
        parsed["reasoning"] = native_reasoning
    return parsed


def build_sft_row(
    *,
    sample_id: str,
    doc_family_id: str,
    cate_id: str,
    task_type: str,
    source_bundle: str,
    question: str,
    reasoning: str,
    final_answer: str,
    enable_thinking: bool,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_content: str | None = None,
    cot_mode: str = "rag_citation",
) -> dict[str, Any]:
    if enable_thinking and reasoning:
        assistant_content = f"<think>\n{reasoning}\n</think>\n\n{final_answer}"
        channel = "wiki_rag_qa_cot"
    else:
        assistant_content = final_answer
        channel = "wiki_rag_qa_direct"

    effective_user = user_content if user_content is not None else f"{source_bundle}\n\n问题：{question}"
    return {
        "sample_id": sample_id,
        "doc_family_id": doc_family_id,
        "cate_id": cate_id,
        "task_type": task_type,
        "cot_mode": cot_mode,
        "channel": channel,
        "enable_thinking": bool(enable_thinking and reasoning),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": effective_user},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def build_clean_direct_twin_row(
    *,
    sample_id: str,
    doc_family_id: str,
    cate_id: str,
    task_type: str,
    source_bundle: str,
    question: str,
    clean_gold_answer: str,
    gold_entity_anchors: list[str],
    system_prompt: str,
    user_content: str | None,
    cot_mode: str = "internalized",
) -> dict[str, Any]:
    """Builds a rationale-first non-thinking twin from a PASSED CoT to prevent Thinking=OFF collapse."""
    if gold_entity_anchors:
        anchor_str = "、".join(gold_entity_anchors[:4])
        compact_rationale = f"核心特征锚点：{anchor_str}。"
    else:
        compact_rationale = "核验要点：严格对齐给定规则的前置触发要件、但书限制与例外边界。"

    direct_answer = f"{compact_rationale}\n{clean_gold_answer}"
    row = build_sft_row(
        sample_id=f"{sample_id}__direct_twin",
        doc_family_id=doc_family_id,
        cate_id=cate_id,
        task_type=task_type,
        source_bundle=source_bundle,
        question=question,
        reasoning="",
        final_answer=direct_answer,
        enable_thinking=False,
        system_prompt=system_prompt,
        user_content=user_content,
        cot_mode=cot_mode,
    )
    row["direct_source"] = "clean_twin"
    row["parent_sample_id"] = sample_id
    return row


def convert_cot_to_midtrain_prose(
    *,
    sample_id: str,
    doc_family_id: str,
    cate_id: str,
    task_type: str,
    question: str,
    reasoning: str,
    clean_gold_answer: str,
    entity_sources: list[dict[str, Any]],
    entity_distractor_subjects: list[str],
) -> dict[str, Any]:
    """Converts validated QA-CoT into CPT-safe encyclopedic prose (zero ChatML/<think>/[Rx])."""
    cleaned_reason = THINK_TAG_RE.sub("", reasoning or "")
    cleaned_reason = CITATION_TAG_RE.sub("", cleaned_reason)
    cleaned_reason = RULE_TAG_RE.sub("", cleaned_reason).strip()

    entity_titles = "、".join(
        _clean_entity_subject(str(e.get("title") or "")) for e in entity_sources if e.get("title")
    )
    header = (
        f"[领域知识精炼 · {entity_titles or cate_id}]"
        if entity_titles
        else f"[领域合规要点 · {cate_id}]"
    )
    diff_note = (
        f"\n[易混淆辨析] 需与 {'、'.join(entity_distractor_subjects)} 明确区分。"
        if entity_distractor_subjects
        else ""
    )
    prose_body = (
        f"{header}\n"
        f"[核心判定命题] {question}\n"
        f"[本体推演知识] {cleaned_reason}\n"
        f"[稳定结论锚点] {clean_gold_answer}"
        f"{diff_note}"
    )
    return {
        "sample_id": f"{sample_id}__prose",
        "doc_family_id": doc_family_id,
        "cate_id": cate_id,
        "task_type": task_type,
        "channel": "ch_qa_knowledge_prose",
        "is_qa_prose": True,
        "is_unresolved_tail": task_type in {"audit", "synthesis", "insufficient"},
        "images": [],
        "messages": [{"role": "assistant", "content": prose_body}],
    }


def _stratified_sample(rows: list[dict[str, Any]], k: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) <= k:
        return list(rows)
    rng = random.Random(seed)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_type[str(r.get("task_type") or "fact")].append(r)
    for bucket in by_type.values():
        rng.shuffle(bucket)

    picked: list[dict[str, Any]] = []
    keys = sorted(by_type.keys())
    idx = 0
    while len(picked) < k and any(by_type.values()):
        key = keys[idx % len(keys)]
        if by_type[key]:
            picked.append(by_type[key].pop(0))
        idx += 1
    return picked


def run_cot_generation_pipeline(
    *,
    qa_records: list[dict[str, Any]],
    output_dir: Path,
    prompt_path: Path | None = None,
    cot_mode: str = "internalized",
    dedup: bool | None = None,
    mode: str = "mock",
    api_base: str = "http://127.0.0.1:8000",
    model_id: str = "qwen3.8-flash",
    temperature: float = 0.4,
    max_tokens: int = 4096,
    thinking_budget: int = 2048,
    max_retries: int = 1,
    inject_distractors: bool = True,
    target_cot_direct_ratio: float | None = None,
    generate_clean_direct_twins: bool | None = None,
    max_fallback_share_in_direct: float = 0.20,
    anchor_coverage_threshold: float = 0.65,
    smoke_sample: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    """Runs QA->CoT v0.4 internalized (or v0.3 rag_citation) bridging & dual-mode SFT/CPT-prose export."""
    if cot_mode not in {"internalized", "rag_citation"}:
        raise ValueError(f"Unsupported cot_mode: {cot_mode}")

    effective_ratio = (
        target_cot_direct_ratio
        if target_cot_direct_ratio is not None
        else (0.7 if cot_mode == "internalized" else 0.8)
    )
    do_clean_twins = (
        generate_clean_direct_twins
        if generate_clean_direct_twins is not None
        else (cot_mode == "internalized")
    )

    effective_prompt_path = prompt_path or (
        DEFAULT_PROMPT_V04_PATH if cot_mode == "internalized" else DEFAULT_PROMPT_PATH
    )
    template_text = effective_prompt_path.read_text(encoding="utf-8")

    do_dedup = (cot_mode == "internalized") if dedup is None else dedup
    if do_dedup:
        active_records, dedup_stats = dedup_qa_records(qa_records)
    else:
        active_records = list(qa_records)
        dedup_stats = {
            "raw_input_rows": len(qa_records),
            "unique_input_rows": len(active_records),
            "dedup_dropped_rows": 0,
        }

    # Build global distractor pool from all available sources/distractors
    distractor_pool: list[dict[str, Any]] = []
    for rec in active_records:
        cate_id = str(rec.get("cate_id") or "")
        for s in rec.get("distractors") or []:
            item = dict(s)
            item.setdefault("cate_id", cate_id)
            distractor_pool.append(item)
        for s in rec.get("sources") or []:
            item = dict(s)
            item.setdefault("cate_id", cate_id)
            distractor_pool.append(item)

    cot_passed_rows: list[dict[str, Any]] = []
    clean_direct_twin_rows: list[dict[str, Any]] = []
    direct_fallback_rows: list[dict[str, Any]] = []
    midtrain_prose_rows: list[dict[str, Any]] = []
    dpo_rejected_rows: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []

    task_type_counts: Counter[str] = Counter()
    raw_type_aliases_fixed: Counter[str] = Counter()
    gate_fail_counts: Counter[str] = Counter()
    mutability_totals: Counter[str] = Counter()
    anchor_cov_values: list[float] = []
    soft_warning_count = 0

    for idx, raw in enumerate(active_records):
        sample_id = str(raw.get("sample_id") or f"qa_{idx}")
        doc_family_id = str(raw.get("doc_family_id") or f"fam_{sample_id}")
        cate_id = str(raw.get("cate_id") or "general")
        task_type, raw_task_type = normalize_task_type(raw.get("task_type"))
        if raw_task_type != task_type:
            raw_type_aliases_fixed[f"{raw_task_type}->{task_type}"] += 1
        task_type_counts[task_type] += 1
        answerable = bool(raw.get("answerable", True))
        question = str(raw.get("question") or "").strip()

        if cot_mode == "internalized":
            ctx_v04 = split_evidence_by_mutability(
                raw,
                distractor_pool=distractor_pool,
                inject_distractors=inject_distractors,
                seed=seed + idx,
            )
            for mk, mv in ctx_v04["split_counts"].items():
                mutability_totals[mk] += mv
            prompt = render_cot_v04_prompt(
                template_text,
                task_type=task_type,
                answerable=answerable,
                user_given_rules=ctx_v04["user_given_rules"],
                entity_kb_reference=ctx_v04["entity_kb_reference"],
                entity_distractors_hint=ctx_v04["entity_distractors_hint"],
                question=question,
                clean_gold_answer=ctx_v04["clean_gold_answer"],
            )
            gold_answer_target = ctx_v04["clean_gold_answer"]
            source_bundle_val = ctx_v04["user_given_rules"]
            user_content_val = ctx_v04["user_prompt"]
            sys_prompt_val = DEFAULT_V04_SYSTEM_PROMPT
        else:
            ctx = inject_distractors_and_remap(
                raw,
                distractor_pool=distractor_pool,
                inject_distractors=inject_distractors,
                seed=seed + idx,
            )
            prompt = render_cot_bridge_prompt(
                template_text,
                task_type=task_type,
                answerable=answerable,
                distractor_keys=ctx["distractor_keys"],
                source_bundle=ctx["source_bundle"],
                question=question,
                gold_answer=ctx["gold_answer"],
            )
            gold_answer_target = ctx["gold_answer"]
            source_bundle_val = ctx["source_bundle"]
            user_content_val = None
            sys_prompt_val = DEFAULT_SYSTEM_PROMPT

        best_report: dict[str, Any] | None = None
        best_reasoning = ""
        best_final = gold_answer_target
        failed_traces: list[tuple[str, list[str]]] = []

        for attempt in range(max_retries + 1):
            call_seed = seed + idx * 10 + attempt
            if mode == "mock":
                if raw.get("_mock_force_fail"):
                    resp = {
                        "reasoning": "参考答案指出合规。[S1]",
                        "final_answer": gold_answer_target,
                    }
                elif cot_mode == "internalized":
                    resp = generate_mock_cot_v04_response(
                        task_type=task_type,
                        answerable=answerable,
                        question=question,
                        clean_gold_answer=ctx_v04["clean_gold_answer"],
                        gold_entity_anchors=ctx_v04["gold_entity_anchors"],
                        entity_sources=ctx_v04["entity_sources"],
                        entity_distractor_subjects=ctx_v04["entity_distractor_subjects"],
                        rule_gold_keys=ctx_v04["rule_gold_keys"],
                        rule_distractor_keys=ctx_v04["rule_distractor_keys"],
                        rule_source_map=ctx_v04["rule_source_map"],
                    )
                else:
                    resp = generate_mock_cot_response(
                        task_type=task_type,
                        answerable=answerable,
                        question=question,
                        gold_answer=ctx["gold_answer"],
                        gold_citations=ctx["gold_citations"],
                        distractor_keys=ctx["distractor_keys"],
                        source_map=ctx["source_map"],
                    )
            elif mode == "vllm":
                resp = call_qwen_teacher_api(
                    api_base=api_base,
                    model_id=model_id,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    thinking_budget=thinking_budget,
                    seed=call_seed,
                    api_key=os.environ.get("API_KEY", ""),
                )
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            reasoning = str(resp.get("reasoning") or "").strip()
            final_answer = str(resp.get("final_answer") or gold_answer_target).strip()
            if cot_mode == "internalized":
                report = verify_cot_v04_8gates(
                    reasoning=reasoning,
                    final_answer=final_answer,
                    clean_gold_answer=ctx_v04["clean_gold_answer"],
                    task_type=task_type,
                    answerable=answerable,
                    gold_entity_anchors=ctx_v04["gold_entity_anchors"],
                    entity_sources=ctx_v04["entity_sources"],
                    entity_distractor_subjects=ctx_v04["entity_distractor_subjects"],
                    rule_gold_keys=ctx_v04["rule_gold_keys"],
                    rule_distractor_keys=ctx_v04["rule_distractor_keys"],
                    rule_source_map=ctx_v04["rule_source_map"],
                    entity_source_map=ctx_v04["entity_source_map"],
                    anchor_coverage_threshold=anchor_coverage_threshold,
                )
            else:
                report = verify_cot_8gates(
                    reasoning=reasoning,
                    final_answer=final_answer,
                    gold_answer=ctx["gold_answer"],
                    task_type=task_type,
                    answerable=answerable,
                    gold_citations=ctx["gold_citations"],
                    distractor_keys=ctx["distractor_keys"],
                    source_map=ctx["source_map"],
                    evidence=ctx["evidence"],
                )
            best_report = report
            best_reasoning = reasoning
            best_final = final_answer
            if report["passed"]:
                break
            failed_traces.append((reasoning, report["errors"]))

        assert best_report is not None
        soft_warning_count += len(best_report["soft_warnings"])
        if "entity_anchor_coverage" in best_report["metrics"]:
            anchor_cov_values.append(float(best_report["metrics"]["entity_anchor_coverage"]))

        if best_report["passed"]:
            sft_row = build_sft_row(
                sample_id=sample_id,
                doc_family_id=doc_family_id,
                cate_id=cate_id,
                task_type=task_type,
                source_bundle=source_bundle_val,
                question=question,
                reasoning=best_reasoning,
                final_answer=best_final,
                enable_thinking=True,
                system_prompt=sys_prompt_val,
                user_content=user_content_val,
                cot_mode=cot_mode,
            )
            cot_passed_rows.append(sft_row)
            if do_clean_twins:
                anchors_for_twin = (
                    ctx_v04["gold_entity_anchors"] if cot_mode == "internalized" else []
                )
                clean_direct_twin_rows.append(
                    build_clean_direct_twin_row(
                        sample_id=sample_id,
                        doc_family_id=doc_family_id,
                        cate_id=cate_id,
                        task_type=task_type,
                        source_bundle=source_bundle_val,
                        question=question,
                        clean_gold_answer=best_final,
                        gold_entity_anchors=anchors_for_twin,
                        system_prompt=sys_prompt_val,
                        user_content=user_content_val,
                        cot_mode=cot_mode,
                    )
                )
            if cot_mode == "internalized":
                midtrain_prose_rows.append(
                    convert_cot_to_midtrain_prose(
                        sample_id=sample_id,
                        doc_family_id=doc_family_id,
                        cate_id=cate_id,
                        task_type=task_type,
                        question=question,
                        reasoning=best_reasoning,
                        clean_gold_answer=best_final,
                        entity_sources=ctx_v04["entity_sources"],
                        entity_distractor_subjects=ctx_v04["entity_distractor_subjects"],
                    )
                )
            for fail_trace, fail_errs in failed_traces:
                if fail_trace:
                    dpo_rejected_rows.append(
                        {
                            "sample_id": sample_id,
                            "messages": sft_row["messages"],
                            "rejected_response": f"<think>\n{fail_trace}\n</think>\n\n{best_final}",
                            "rejected_reasons": fail_errs,
                        }
                    )
        else:
            for err in best_report["errors"]:
                gate_fail_counts[err.split(":", 1)[0]] += 1
            # Strictly degrade: clear reasoning & set enable_thinking=False (never stitch quotes)
            fallback_row = build_sft_row(
                sample_id=sample_id,
                doc_family_id=doc_family_id,
                cate_id=cate_id,
                task_type=task_type,
                source_bundle=source_bundle_val,
                question=question,
                reasoning="",
                final_answer=gold_answer_target,
                enable_thinking=False,
                system_prompt=sys_prompt_val,
                user_content=user_content_val,
                cot_mode=cot_mode,
            )
            fallback_row["direct_source"] = "gate_fallback"
            fallback_row["fallback_reasons"] = best_report["errors"]
            direct_fallback_rows.append(fallback_row)
            if best_reasoning:
                dpo_rejected_rows.append(
                    {
                        "sample_id": sample_id,
                        "messages": fallback_row["messages"],
                        "rejected_response": f"<think>\n{best_reasoning}\n</think>\n\n{gold_answer_target}",
                        "rejected_reasons": best_report["errors"],
                    }
                )

        audit_entry: dict[str, Any] = {
            "sample_id": sample_id,
            "cot_mode": cot_mode,
            "task_type": task_type,
            "raw_task_type": raw_task_type,
            "passed": best_report["passed"],
            "gates": best_report["gates"],
            "errors": best_report["errors"],
            "soft_warnings": best_report["soft_warnings"],
            "metrics": best_report["metrics"],
        }
        if cot_mode == "internalized":
            audit_entry["split_counts"] = ctx_v04["split_counts"]
            audit_entry["rule_distractor_keys"] = ctx_v04["rule_distractor_keys"]
            audit_entry["entity_distractor_subjects"] = ctx_v04["entity_distractor_subjects"]
        else:
            audit_entry["distractor_keys"] = ctx["distractor_keys"]
            audit_entry["key_mapping"] = ctx["key_mapping"]
        audit_records.append(audit_entry)

    # Mix CoT:Direct at target ratio (clean rationale-first twins >= 80% of Direct slice)
    rng = random.Random(seed)
    mixed_rows = list(cot_passed_rows)
    selected_direct_rows: list[dict[str, Any]] = []
    if cot_passed_rows and 0.0 < effective_ratio < 1.0:
        desired_direct = max(
            1,
            int(round(len(cot_passed_rows) * (1.0 - effective_ratio) / effective_ratio)),
        )
        clean_pool = list(clean_direct_twin_rows)
        fallback_pool = list(direct_fallback_rows)
        rng.shuffle(clean_pool)
        rng.shuffle(fallback_pool)
        if clean_pool:
            max_fb = int(desired_direct * max(0.0, min(1.0, max_fallback_share_in_direct)))
            picked_fb = fallback_pool[:max_fb]
            needed_clean = desired_direct - len(picked_fb)
            picked_clean = clean_pool[:needed_clean]
            selected_direct_rows = picked_clean + picked_fb
            if len(selected_direct_rows) < desired_direct:
                rem = desired_direct - len(selected_direct_rows)
                selected_direct_rows.extend(fallback_pool[len(picked_fb) : len(picked_fb) + rem])
        elif fallback_pool:
            selected_direct_rows = fallback_pool[:desired_direct]
        mixed_rows.extend(selected_direct_rows)
    elif not cot_passed_rows:
        selected_direct_rows = list(clean_direct_twin_rows) + list(direct_fallback_rows)
        mixed_rows.extend(selected_direct_rows)
    rng.shuffle(mixed_rows)

    clean_in_direct = sum(1 for r in selected_direct_rows if r.get("direct_source") == "clean_twin")
    direct_bucket_clean_share = (
        round(clean_in_direct / len(selected_direct_rows), 4)
        if selected_direct_rows
        else 1.0
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    cot_path = output_dir / "qa_cot_passed.jsonl"
    clean_twins_path = output_dir / "qa_direct_clean_twins.jsonl"
    direct_path = output_dir / "qa_direct_fallback.jsonl"
    prose_path = output_dir / "qa_midtrain_prose.jsonl"
    mixed_path = output_dir / "qa_sft_mixed_8_2.jsonl"
    dpo_rej_path = output_dir / "qa_dpo_rejected_cots.jsonl"
    audit_path = output_dir / "cot_audit_log.jsonl"
    manifest_path = output_dir / "cot_manifest.json"

    write_jsonl(cot_path, cot_passed_rows)
    write_jsonl(clean_twins_path, clean_direct_twin_rows)
    write_jsonl(direct_path, direct_fallback_rows)
    write_jsonl(prose_path, midtrain_prose_rows)
    write_jsonl(mixed_path, mixed_rows)
    write_jsonl(dpo_rej_path, dpo_rejected_rows)
    write_jsonl(audit_path, audit_records)

    smoke_path = None
    smoke_count = 0
    if smoke_sample > 0:
        smoke_rows = _stratified_sample(cot_passed_rows or mixed_rows, smoke_sample, seed=seed)
        smoke_path = output_dir / f"smoke_{smoke_sample}_review.jsonl"
        write_jsonl(smoke_path, smoke_rows)
        smoke_count = len(smoke_rows)

    avg_anchor_cov = (
        round(sum(anchor_cov_values) / len(anchor_cov_values), 4)
        if anchor_cov_values
        else None
    )
    manifest = {
        "schema": "qa_cot_v0.4" if cot_mode == "internalized" else "qa_cot_v0.3",
        "cot_mode": cot_mode,
        "teacher_model": model_id if mode == "vllm" else f"mock({model_id})",
        "thinking_budget": thinking_budget,
        **dedup_stats,
        "input_rows": len(active_records),
        "cot_passed_rows": len(cot_passed_rows),
        "clean_direct_twin_rows": len(clean_direct_twin_rows),
        "direct_fallback_rows": len(direct_fallback_rows),
        "midtrain_prose_rows": len(midtrain_prose_rows),
        "mixed_sft_rows": len(mixed_rows),
        "dpo_rejected_cots_rows": len(dpo_rejected_rows),
        "target_cot_direct_ratio": effective_ratio,
        "direct_bucket_clean_share": direct_bucket_clean_share,
        "task_type_counts": dict(sorted(task_type_counts.items())),
        "raw_type_aliases_fixed": dict(sorted(raw_type_aliases_fixed.items())),
        "mutability_split_counts": dict(sorted(mutability_totals.items())),
        "avg_entity_anchor_coverage": avg_anchor_cov,
        "gate_fail_counts": dict(sorted(gate_fail_counts.items())),
        "soft_warning_count": soft_warning_count,
        "smoke_sample_rows": smoke_count,
        "artifacts": {
            "qa_cot_passed": str(cot_path),
            "qa_direct_clean_twins": str(clean_twins_path),
            "qa_direct_fallback": str(direct_path),
            "qa_midtrain_prose": str(prose_path),
            "qa_sft_mixed_8_2": str(mixed_path),
            "qa_dpo_rejected_cots": str(dpo_rej_path),
            "cot_audit_log": str(audit_path),
            "smoke_review": str(smoke_path) if smoke_path else None,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="Path to qa_kb_raw.jsonl")
    parser.add_argument("--demo", action="store_true", help="Run on bundled demo fixture")
    parser.add_argument(
        "--cot-mode",
        choices=["internalized", "rag_citation"],
        default="internalized",
        help="v0.4 internalized knowledge vs v0.3 legacy RAG citation mode",
    )
    parser.add_argument("--no-dedup", action="store_true", help="Disable canonical QA dedup")
    parser.add_argument("--no-clean-direct-twins", action="store_true", help="Disable rationale-first direct twins")
    parser.add_argument("--prompt-template", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["mock", "vllm"], default="mock")
    parser.add_argument("--api-base", default=os.environ.get("API_BASE", "http://127.0.0.1:8000"))
    parser.add_argument("--model-id", default=os.environ.get("COT_TEACHER_MODEL", "qwen3.8-flash"))
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--thinking-budget", type=int, default=2048)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--no-distractors", action="store_true")
    parser.add_argument("--target-cot-direct-ratio", type=float, default=None)
    parser.add_argument("--anchor-coverage-threshold", type=float, default=0.65)
    parser.add_argument("--smoke-sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.demo:
        demo_file = (
            "qa_kb_v04_example.jsonl"
            if args.cot_mode == "internalized"
            else "qa_kb_raw_example.jsonl"
        )
        input_path = THIS_DIR / "examples" / demo_file
    elif args.input:
        input_path = args.input
    else:
        raise SystemExit("Specify --demo or --input <qa_kb_raw.jsonl>.")

    qa_records = load_jsonl(input_path)
    manifest = run_cot_generation_pipeline(
        qa_records=qa_records,
        output_dir=args.output_dir,
        prompt_path=args.prompt_template,
        cot_mode=args.cot_mode,
        dedup=not args.no_dedup,
        mode=args.mode,
        api_base=args.api_base,
        model_id=args.model_id,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        thinking_budget=args.thinking_budget,
        max_retries=args.max_retries,
        inject_distractors=not args.no_distractors,
        target_cot_direct_ratio=args.target_cot_direct_ratio,
        generate_clean_direct_twins=False if args.no_clean_direct_twins else None,
        anchor_coverage_threshold=args.anchor_coverage_threshold,
        smoke_sample=args.smoke_sample,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
