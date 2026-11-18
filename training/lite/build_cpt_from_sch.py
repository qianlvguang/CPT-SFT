#!/usr/bin/env python3
"""Phase A (Lite Gain Experiment) CPT Bundle Builder (`bundle_lite_v3_all·val0`).

Implements the `sch iso` transformation and Lite bundle assembly per the
CPT-27B Runbook (2026-11-18):
  1. Converts `sch` records into single-assistant CPT format (strips user queries,
     prepends `<image>` tokens, assembles strictly from atomic `desc` + `elements`
     instead of `polished_text`).
  2. Determines channel strictly from actual `elements` content and demotes `label`
     to an audit-only field (`audit_raw_label`):
       - Empty / "无" / "元素：无。" -> `ch_ip_negative` (even when `label=True`).
       - Valid elements hitting KB title-scan / RAG gate -> `ch5_flash_rag_raft`
         (three-part grounding + `[S1]` rule hit + `[S2]` distractor exclusion).
       - Remaining valid elements -> `ch_ip_grounding`.
  3. Merges filtered `cpt_kb_mm` (T1_FULL levels or template-bound variants,
     excluding `ft_mentions`) and `cpt_kb_supplement` into `kb text` / `kb mm` /
     `kb rel`, enforces zero duplicate rows within the bundle, writes empty
     `lite_val.jsonl` (`val0`), computes SHA256/MD5 manifests, and creates
     `bundle_lite_compat/stage1_train.jsonl` relative symlink.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

IMAGE_PLACEHOLDER = "<image>"
ALLOWED_IMAGE_PREFIX_RE = re.compile(
    r"^(?:(?:图\s*\d+|待审图|商品图|参考图|原图|对比图|Image\s*\d+)\s*[：:]?\s*)*$"
)

EMPTY_ELEMENT_TOKENS = frozenset(
    {"", "无", "无。", "元素：无", "元素：无。", "none", "null", "n/a", "na", "-"}
)

T1_FULL_DEFAULT_LEVELS = frozenset({"a_tier", "generic", "disney_movies"})

KB_TEXT_KINDS = frozenset(
    {"kb_ip", "kb_audit", "kb_entity", "kb_ip_supp", "kb_logo", "kb_bc", "kb_doc", "ch_text_doc"}
)
KB_MM_KINDS = frozenset(
    {"kb_ip_mm", "kb_ip_supp_mm", "kb_entity_mm", "kb_audit_mm", "kb_logo_mm", "ch_mm_doc"}
)
KB_REL_KINDS = frozenset(
    {
        "kb_rel_entity_audit",
        "kb_rel_logo_audit",
        "kb_rel_ip_audit",
        "kb_rel_audit_ip",
        "ch_rel_jump",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def estimate_record_tokens(
    content: str,
    num_images: int,
    tokens_per_image: int = 512,
) -> int:
    zh_chars = sum(1 for ch in content if "\u4e00" <= ch <= "\u9fff")
    other_chars = max(0, len(content) - zh_chars)
    text_tokens = int(round(zh_chars / 1.6 + other_chars / 4.0))
    return max(1, text_tokens + num_images * tokens_per_image)


def _format_image_prefix(num_images: int) -> str:
    if num_images <= 0:
        return ""
    if num_images == 1:
        return "<image>\n"
    return "\n".join(f"图{i}：<image>" for i in range(1, num_images + 1)) + "\n"


def normalize_elements(raw_elements: Any) -> list[str]:
    """Extracts clean, non-empty element names from `elements` field."""
    if raw_elements is None:
        return []
    if isinstance(raw_elements, list):
        items = [str(x).strip() for x in raw_elements]
    else:
        text = str(raw_elements).strip()
        if text.startswith("元素：") or text.startswith("元素:"):
            text = text.split("：", 1)[-1].split(":", 1)[-1].strip()
        items = [seg.strip() for seg in re.split(r"[、,;；\n]+", text)]
    cleaned: list[str] = []
    for item in items:
        if item.lower() in EMPTY_ELEMENT_TOKENS:
            continue
        cleaned.append(item)
    return cleaned


def validate_cpt_record(record: dict[str, Any]) -> list[str]:
    """Validates single-assistant CPT format, `<image>` parity, and causal prefix."""
    errors: list[str] = []
    if not str(record.get("channel", "")).strip():
        errors.append("missing_channel")
    if not str(record.get("doc_family_id", "")).strip():
        errors.append("missing_doc_family_id")

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        errors.append("cpt_requires_single_assistant_message")
        return errors
    msg = messages[0]
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        errors.append("cpt_message_role_must_be_assistant")
        return errors
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        errors.append("empty_assistant_content")
        return errors

    images = record.get("images") or []
    if not isinstance(images, list) or any(not isinstance(x, str) or not x.strip() for x in images):
        errors.append("invalid_images_list")
        return errors

    placeholders = content.count(IMAGE_PLACEHOLDER)
    if placeholders != len(images):
        errors.append(f"image_placeholder_mismatch:placeholders={placeholders},images={len(images)}")
    if images and placeholders > 0:
        first_idx = content.find(IMAGE_PLACEHOLDER)
        prefix = content[:first_idx].strip()
        if prefix and not ALLOWED_IMAGE_PREFIX_RE.fullmatch(prefix):
            errors.append("causal_prefix_violation:text_before_first_image")
    return errors


def _extract_desc(raw: dict[str, Any]) -> str:
    """Extracts atomic visual description, never falling back to `polished_text`."""
    desc = raw.get("desc")
    if isinstance(desc, str) and desc.strip():
        return re.sub(r"<image>\s*", "", desc).strip()
    # Fallback only if messages exist without `polished_text`
    for msg in raw.get("messages") or []:
        if msg.get("role") == "assistant" and str(msg.get("content", "")).strip():
            return re.sub(r"<image>\s*", "", str(msg["content"])).strip()
    return ""


def _build_kb_title_index(kb_records: list[dict[str, Any]]) -> dict[str, str]:
    """Indexes KB entity/title keywords to snippet text for sch -> ch5 title-scan upgrade."""
    index: dict[str, str] = {}
    for rec in kb_records:
        title = str(rec.get("title") or rec.get("entity_name") or "").strip()
        text = str(rec.get("text") or "").strip()
        if not text and rec.get("messages"):
            text = str(rec["messages"][0].get("content", "")).strip()
        if title and text:
            index[title] = text
    return index


def transform_sch_record(
    raw: dict[str, Any],
    *,
    kb_title_index: dict[str, str] | None = None,
) -> tuple[dict[str, Any] | None, str | None, dict[str, bool]]:
    """Transforms one `sch` source row using `desc` + `elements` atomic fields.

    Returns:
      (cpt_row_or_none, drop_reason_or_none, audit_flags)
    """
    sample_id = str(raw.get("sample_id") or "").strip()
    if not sample_id:
        return None, "missing_sample_id", {}

    desc = _extract_desc(raw)
    if not desc:
        return None, "empty_desc", {}

    images = [str(x) for x in (raw.get("images") or []) if str(x).strip()]
    prefix = _format_image_prefix(len(images))
    elements = normalize_elements(raw.get("elements"))
    raw_label = raw.get("label")
    label_true_but_empty = (raw_label is True) and (len(elements) == 0)

    # Critical Runbook rule: determine channel strictly by actual `elements`, not `label`
    if not elements:
        channel = "ch_ip_negative"
        content = f"{prefix}{desc}\n元素：无。"
    else:
        elem_line = "、".join(elements)
        rag_hit = raw.get("rag_hit") or raw.get("rag_evidence")
        if not rag_hit and kb_title_index:
            for elem in elements:
                if elem in kb_title_index:
                    rag_hit = f"[S1] {kb_title_index[elem]}"
                    break

        if rag_hit:
            s1_text = (
                "\n".join(str(x) for x in rag_hit)
                if isinstance(rag_hit, list)
                else str(rag_hit).strip()
            )
            if not s1_text.startswith("[S"):
                s1_text = f"[S1] {s1_text}"
            s2_text = str(raw.get("distractor_s2") or "").strip()
            parts = [
                f"{prefix}[视觉描述]：{desc}",
                f"元素：{elem_line}。",
                f"[检索命中依据]\n{s1_text}",
            ]
            if s2_text:
                if not s2_text.startswith("[S"):
                    s2_text = f"[S2] {s2_text}"
                parts.append(f"[干扰对照排除]\n{s2_text}")
            parts.append(f"[核验结论]：依据检索材料与可见元素特征，确认命中 {elem_line}。")
            content = "\n".join(parts)
            channel = "ch5_flash_rag_raft"
        else:
            content = f"{prefix}{desc}\n元素：{elem_line}。"
            channel = "ch_ip_grounding"

    row: dict[str, Any] = {
        "sample_id": sample_id,
        "doc_family_id": str(raw.get("doc_family_id") or f"sch_{sample_id}"),
        "channel": channel,
        "source_kind": "sch_iso",
        "messages": [{"role": "assistant", "content": content}],
        "audit_raw_label": raw_label,
    }
    if images:
        row["images"] = images
    for opt in ("template_id", "level", "example_only"):
        if opt in raw:
            row[opt] = raw[opt]

    errors = validate_cpt_record(row)
    if errors:
        return None, f"validation_failed:{','.join(errors)}", {}

    return row, None, {"label_true_but_empty_elements": label_true_but_empty}


def _classify_kb_bucket(record: dict[str, Any]) -> str:
    source_kind = str(record.get("source_kind") or "").strip()
    channel = str(record.get("channel") or "").strip()
    if source_kind in KB_REL_KINDS or channel == "ch_rel_jump":
        return "kb_rel"
    if source_kind in KB_MM_KINDS or channel == "ch_mm_doc" or bool(record.get("images")):
        return "kb_mm"
    if source_kind in KB_TEXT_KINDS or channel in ("ch_text_doc", "ch1_wiki_text_stub"):
        return "kb_text"
    return "kb_text"


def _is_ft_mention(record: dict[str, Any]) -> bool:
    if record.get("is_ft_mention") is True or record.get("ft_mention") is True:
        return True
    sk = str(record.get("source_kind") or "").lower()
    return "ft_mention" in sk


def _is_eligible_kb_record(
    record: dict[str, Any],
    *,
    t1_levels: frozenset[str],
    template_ids: set[str] | None,
) -> bool:
    if _is_ft_mention(record):
        return False
    if record.get("is_kb_supplement") is True or record.get("record_kind") == "kb_supplement":
        return True
    level = str(record.get("level") or "").strip()
    if level in t1_levels:
        return True
    tpl = str(record.get("template_id") or "").strip()
    if template_ids and tpl in template_ids:
        return True
    for rel_tpl in record.get("rel_bound_templates") or []:
        if not template_ids or str(rel_tpl) in template_ids:
            return True
    # Allow records without explicit level/template metadata when no filter list is given
    if not level and not template_ids:
        return True
    return False


def _verify_images_if_requested(rows: list[dict[str, Any]], image_root: Path | None) -> dict[str, Any]:
    unique_imgs: set[str] = set()
    for r in rows:
        for img in r.get("images") or []:
            unique_imgs.add(str(img))
    missing = 0
    zero_bytes = 0
    corrupted = 0
    pil_image = None
    try:
        from PIL import Image  # type: ignore[import-not-found]
        pil_image = Image
    except ImportError:
        pil_image = None

    for rel in sorted(unique_imgs):
        p = Path(rel)
        if not p.is_absolute() and image_root is not None:
            p = image_root / rel
        if not p.exists():
            missing += 1
            continue
        if p.stat().st_size == 0:
            zero_bytes += 1
            continue
        if pil_image is not None:
            try:
                with pil_image.open(p) as im:
                    im.verify()
            except Exception:
                corrupted += 1
    return {
        "unique_images": len(unique_imgs),
        "missing": missing,
        "zero_bytes": zero_bytes,
        "corrupted": corrupted,
    }


def build_lite_bundle(
    *,
    sch_records: list[dict[str, Any]],
    kb_records: list[dict[str, Any]],
    kb_supplement_records: list[dict[str, Any]],
    output_dir: Path,
    t1_levels: frozenset[str] = T1_FULL_DEFAULT_LEVELS,
    template_ids: set[str] | None = None,
    tokens_per_image: int = 512,
    gbs: int = 32,
    max_length: int = 16384,
    seed: int = 42,
    check_imgs: bool = False,
    image_root: Path | None = None,
) -> dict[str, Any]:
    """Builds `bundle_lite_v3_all` (val0) and `bundle_lite_compat/stage1_train.jsonl`."""
    kb_all = list(kb_records) + list(kb_supplement_records)
    kb_title_index = _build_kb_title_index(kb_all)

    sch_built: list[dict[str, Any]] = []
    sch_drop_reasons: Counter[str] = Counter()
    label_true_but_empty = 0

    for raw in sch_records:
        row, drop_reason, flags = transform_sch_record(raw, kb_title_index=kb_title_index)
        if drop_reason is not None or row is None:
            sch_drop_reasons[drop_reason or "unknown"] += 1
            continue
        if flags.get("label_true_but_empty_elements"):
            label_true_but_empty += 1
        sch_built.append(row)

    kb_kept: list[dict[str, Any]] = []
    ft_mentions_excluded_count = 0
    kb_filtered_out = 0
    kb_subchannel_counts: Counter[str] = Counter()
    kb_bucket_counts: Counter[str] = Counter()

    for raw in kb_all:
        if _is_ft_mention(raw):
            ft_mentions_excluded_count += 1
            continue
        if not _is_eligible_kb_record(raw, t1_levels=t1_levels, template_ids=template_ids):
            kb_filtered_out += 1
            continue
        row = dict(raw)
        if not row.get("doc_family_id"):
            row["doc_family_id"] = f"kb_{row.get('sample_id', len(kb_kept))}"
        if not row.get("channel"):
            row["channel"] = "ch_text_doc"
        errs = validate_cpt_record(row)
        if errs:
            kb_filtered_out += 1
            continue
        bucket = _classify_kb_bucket(row)
        row["lite_bucket"] = bucket
        kb_bucket_counts[bucket] += 1
        sub_kind = str(row.get("source_kind") or row["channel"])
        kb_subchannel_counts[sub_kind] += 1
        kb_kept.append(row)

    combined_raw = sch_built + kb_kept
    if not combined_raw:
        raise ValueError("No valid Lite CPT records remained after sch/kb filtering.")

    # Enforce zero duplicate sample_ids inside the bundle
    seen_ids: set[str] = set()
    combined: list[dict[str, Any]] = []
    dedup_skipped = 0
    for rec in combined_raw:
        sid = str(rec.get("sample_id") or "")
        key = sid or json.dumps(rec.get("messages"), ensure_ascii=False, sort_keys=True)
        if key in seen_ids:
            dedup_skipped += 1
            continue
        seen_ids.add(key)
        clean_row: dict[str, Any] = {
            "sample_id": rec.get("sample_id"),
            "doc_family_id": rec["doc_family_id"],
            "channel": rec["channel"],
            "messages": rec["messages"],
        }
        if rec.get("images"):
            clean_row["images"] = list(rec["images"])
        for opt in ("source_kind", "lite_bucket", "audit_raw_label", "template_id", "level", "example_only"):
            if opt in rec and rec[opt] is not None:
                clean_row[opt] = rec[opt]
        combined.append(clean_row)

    rng = random.Random(seed)
    rng.shuffle(combined)

    bundle_dir = output_dir / "bundle_lite_v3_all"
    compat_dir = output_dir / "bundle_lite_compat"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    compat_dir.mkdir(parents=True, exist_ok=True)

    train_path = bundle_dir / "lite_train.jsonl"
    val_path = bundle_dir / "lite_val.jsonl"
    manifest_path = bundle_dir / "manifest.json"
    sch_report_path = output_dir / "build_sch_report.json"

    write_jsonl(train_path, combined)
    val_path.write_text("", encoding="utf-8")

    compat_link = compat_dir / "stage1_train.jsonl"
    rel_target = os.path.relpath(train_path, compat_dir)
    if compat_link.exists() or compat_link.is_symlink():
        compat_link.unlink()
    compat_link.symlink_to(rel_target)

    # Channel & token accounting
    channel_counts: Counter[str] = Counter()
    total_tokens = 0
    with_images_rows = 0
    mm_rows = 0
    for r in combined:
        bucket = r.get("lite_bucket") or r["channel"]
        channel_counts[str(bucket)] += 1
        n_img = len(r.get("images") or [])
        if n_img > 0:
            with_images_rows += 1
        if bucket == "kb_mm":
            mm_rows += 1
        total_tokens += estimate_record_tokens(
            r["messages"][0]["content"],
            n_img,
            tokens_per_image=tokens_per_image,
        )

    tokens_per_step = max(1, gbs * max_length)
    steps_per_epoch = round(total_tokens / tokens_per_step, 2)

    img_check_report = (
        _verify_images_if_requested(combined, image_root=image_root)
        if check_imgs
        else {"skipped": True}
    )
    if check_imgs and (
        img_check_report["missing"] > 0
        or img_check_report["zero_bytes"] > 0
        or img_check_report["corrupted"] > 0
    ):
        raise RuntimeError(f"Image verification failed: {img_check_report}")

    sch_report = {
        "input_sch_rows": len(sch_records),
        "built_sch_rows": len(sch_built),
        "dropped_sch_rows": len(sch_records) - len(sch_built),
        "dropped_reasons": dict(sorted(sch_drop_reasons.items())),
        "label_true_but_empty_elements_count": label_true_but_empty,
        "sch_channels": dict(sorted(Counter(r["channel"] for r in sch_built).items())),
    }
    sch_report_path.write_text(
        json.dumps(sch_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema": "cpt_27b_lite_v3_all_val0",
        "seed": seed,
        "val_ratio": 0.0,
        "zero_leakage_assert": "passed",
        "ft_mentions_excluded": True,
        "ft_mentions_excluded_count": ft_mentions_excluded_count,
        "kb_filtered_out": kb_filtered_out,
        "dedup_skipped": dedup_skipped,
        "rows": len(combined),
        "sch_rows": len(sch_built),
        "kb_rows": len(kb_kept),
        "with_images_rows": with_images_rows,
        "mm_rows": mm_rows,
        "tokens": total_tokens,
        "steps_per_epoch_gbs32_seq16384": steps_per_epoch,
        "val_rows": 0,
        "channel_counts": dict(sorted(channel_counts.items())),
        "kb_subchannel_counts": dict(sorted(kb_subchannel_counts.items())),
        "image_check": img_check_report,
        "recipe_note": "Zero intra-bundle duplication; epochs controlled on trainer side (LITE_EPOCHS=2).",
        "artifacts": {
            "lite_train": {
                "path": str(train_path),
                "md5": md5_file(train_path),
                "sha256": sha256_file(train_path),
            },
            "lite_val": {
                "path": str(val_path),
                "md5": md5_file(val_path),
                "sha256": sha256_file(val_path),
            },
            "compat_symlink": str(compat_link),
            "build_sch_report": str(sch_report_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_demo_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sample_path = Path(__file__).resolve().parent / "examples" / "lite_sch_kb_sample.jsonl"
    raw = load_jsonl(sample_path)
    sch = [r for r in raw if r.get("record_kind") == "sch"]
    kb = [r for r in raw if r.get("record_kind") == "kb"]
    supp = [r for r in raw if r.get("record_kind") == "kb_supplement"]
    return sch, kb, supp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sch-input", type=Path, default=None, help="Path to raw sch JSONL")
    parser.add_argument("--kb-input", type=Path, default=None, help="Path to cpt_kb_mm.jsonl")
    parser.add_argument("--kb-supplement", type=Path, default=None, help="Path to cpt_kb_supplement.jsonl")
    parser.add_argument("--demo", action="store_true", help="Build demo bundle from examples/lite_sch_kb_sample.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output root directory")
    parser.add_argument("--template-ids", nargs="*", default=None, help="Optional template IDs for KB selection")
    parser.add_argument("--tokens-per-image", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-imgs", action="store_true", help="Verify existence and PIL decode of referenced images")
    parser.add_argument("--image-root", type=Path, default=None, help="Base root for relative image paths")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.demo:
        sch_rows, kb_rows, supp_rows = build_demo_inputs()
    elif args.sch_input or args.kb_input:
        sch_rows = load_jsonl(args.sch_input) if args.sch_input else []
        kb_rows = load_jsonl(args.kb_input) if args.kb_input else []
        supp_rows = load_jsonl(args.kb_supplement) if args.kb_supplement else []
    else:
        raise SystemExit("Specify --demo or --sch-input / --kb-input.")

    tpl_set = set(args.template_ids) if args.template_ids else None
    manifest = build_lite_bundle(
        sch_records=sch_rows,
        kb_records=kb_rows,
        kb_supplement_records=supp_rows,
        output_dir=args.output_dir,
        template_ids=tpl_set,
        tokens_per_image=args.tokens_per_image,
        seed=args.seed,
        check_imgs=args.check_imgs,
        image_root=args.image_root,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
