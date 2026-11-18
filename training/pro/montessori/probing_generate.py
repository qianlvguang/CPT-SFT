#!/usr/bin/env python3
"""Stage 1: Probing QA candidate generation for Montessori-Instruct.

Reads source document records (JSONL), constructs source_bundle prompts using
prompts/qa_generate.md, and samples K distinct QA candidates per source chunk via
an OpenAI-compatible vLLM endpoint (or deterministic mock generator for testing).
Performs programmatic citation/evidence validation and attaches formatted SFT
training messages for downstream student influence probing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_SYSTEM_PROMPT = (
    "使用中文严谨、详尽地回答。先回应问题，再说明必要依据和边界。"
    "只使用给定来源，事实主张贴近引用；区分明确事实、推导与证据不足。"
    "不能编造来源、条号或链接。"
)

TASK_TYPES = ("fact", "condition", "comparison", "synthesis", "insufficient", "version")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_source_bundle(sources: list[dict[str, Any]]) -> tuple[str, dict[str, dict[str, Any]]]:
    """Formats a list of source records into a source_bundle string and source_map."""
    lines: list[str] = []
    source_map: dict[str, dict[str, Any]] = {}
    for idx, src in enumerate(sources, start=1):
        key = src.get("source_key") or f"S{idx}"
        doc_id = str(src["doc_id"])
        title = str(src.get("title", doc_id))
        version = str(src.get("version", "v1"))
        locator = str(src.get("locator", "全文"))
        text = str(src["text"]).strip()
        content_hash = src.get("content_sha256") or sha256_text(text)
        source_map[key] = {
            "source_key": key,
            "doc_id": doc_id,
            "doc_family_id": src.get("doc_family_id", doc_id),
            "title": title,
            "version": version,
            "locator": locator,
            "text": text,
            "content_sha256": content_hash,
            "source_url": src.get("source_url"),
        }
        lines.append(
            f"来源 [{key}]\n"
            f"标题：{title}\n"
            f"版本：{version}\n"
            f"定位：{locator}\n"
            f"正文：{text}"
        )
    return "\n\n".join(lines), source_map


def render_generation_prompt(template_text: str, source_bundle: str, task_type: str | None = None) -> str:
    """Extracts the prompt block from qa_generate.md or renders {{source_bundle}}."""
    prompt = template_text
    if "```text" in template_text:
        parts = template_text.split("```text", 1)[1].split("```", 1)
        prompt = parts[0].strip()
    rendered = prompt.replace("{{source_bundle}}", source_bundle)
    if task_type:
        rendered = f"[指定优先题型: {task_type}]\n\n" + rendered
    return rendered


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Parses JSON from model output, stripping optional markdown fences."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in response")
    return json.loads(cleaned[start : end + 1])


def verify_and_enrich_item(
    item: dict[str, Any],
    source_map: dict[str, dict[str, Any]],
) -> tuple[bool, list[str], list[dict[str, Any]]]:
    """Programmatically checks citations, exact quote substrings, and character offsets."""
    errors: list[str] = []
    question = str(item.get("question", "")).strip()
    answer = str(item.get("answer", "")).strip()
    task_type = str(item.get("task_type", "")).strip()
    answerable = item.get("answerable")

    if not question:
        errors.append("missing_question")
    if not answer:
        errors.append("missing_answer")
    if task_type not in TASK_TYPES:
        errors.append(f"invalid_task_type:{task_type}")
    if not isinstance(answerable, bool):
        errors.append("invalid_answerable_flag")

    citations = item.get("citations")
    if not isinstance(citations, list) or not citations:
        errors.append("missing_citations")
        citations = []

    for c_key in citations:
        if c_key not in source_map:
            errors.append(f"unknown_citation_key:{c_key}")
        elif f"[{c_key}]" not in answer:
            errors.append(f"citation_not_in_answer_text:{c_key}")

    raw_evidence = item.get("evidence")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        errors.append("missing_evidence_list")
        raw_evidence = []

    enriched_evidence: list[dict[str, Any]] = []
    for ev in raw_evidence:
        if not isinstance(ev, dict):
            errors.append("malformed_evidence_entry")
            continue
        s_key = str(ev.get("source_key", "")).strip()
        quote = str(ev.get("quote", "")).strip()
        if s_key not in source_map:
            errors.append(f"evidence_unknown_source_key:{s_key}")
            continue
        if not quote:
            errors.append(f"evidence_empty_quote:{s_key}")
            continue
        src_info = source_map[s_key]
        src_text = src_info["text"]
        start_char = src_text.find(quote)
        if start_char == -1:
            errors.append(f"evidence_quote_not_exact_substring:{s_key}")
            end_char = -1
        else:
            end_char = start_char + len(quote)
        enriched_evidence.append(
            {
                "source_key": s_key,
                "doc_id": src_info["doc_id"],
                "locator": str(ev.get("locator") or src_info["locator"]),
                "quote": quote,
                "start_char": start_char,
                "end_char": end_char,
                "content_sha256": src_info["content_sha256"],
            }
        )

    return len(errors) == 0, errors, enriched_evidence


def build_student_sft_messages(
    source_bundle: str,
    question: str,
    answer: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Constructs the student SFT messages for a QA candidate."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{source_bundle}\n\n问题：{question}"},
        {"role": "assistant", "content": answer},
    ]


def call_vllm_chat(
    api_base: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    api_key: str = "",
) -> str:
    """Calls an OpenAI-compatible vLLM chat completion endpoint."""
    url = api_base.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.load(resp)
    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError(f"Empty choices from {url}")
    return str(choices[0]["message"]["content"])


def generate_mock_response(source_map: dict[str, dict[str, Any]], candidate_idx: int) -> tuple[str, str]:
    """Generates deterministic synthetic teacher responses for local testing and CI."""
    first_key = next(iter(source_map.keys()))
    src = source_map[first_key]
    text = src["text"]
    # Pick a valid substring quote from the source text
    sentences = [s.strip() for s in text.replace("。", "。\n").splitlines() if s.strip()]
    quote_primary = sentences[0] if sentences else text[: min(15, len(text))]
    quote_secondary = sentences[-1] if len(sentences) > 1 else quote_primary

    variants = [
        (
            "condition",
            "根据材料规定，当申请材料存在缺失时，应当按什么顺序处理，能否直接进入审查？",
            f"不能直接进入审查。按所给规则，必须待补正完成后方可开始审查。[{first_key}]\n\n"
            f"具体处理顺序为：首先一次性告知需要补正的全部内容，待申请人补正完成之后再启动审查程序。[{first_key}]\n\n"
            f"当前材料未规定补正的具体时限，对此无法仅凭现有条文作进一步推断。",
            quote_secondary,
        ),
        (
            "fact",
            "材料对于告知补正的方式有何明确要求？",
            f"材料明确要求，当申请材料不完整时，应当一次性告知需要补正的全部内容，并且在补正完成前不开始审查。[{first_key}]",
            quote_primary,
        ),
        (
            "insufficient",
            "如果申请人超过三十日仍未完成补正，材料规定应当如何处理？",
            f"仅凭当前给定材料无法判定逾期未补正的处理方式。[{first_key}]\n\n"
            f"现有规则仅说明“申请材料不完整时，应一次性告知需要补正的全部内容。补正完成前，不开始审查”，并未包含补正期限或逾期退回等规定。[{first_key}]",
            quote_primary,
        ),
        (
            "condition",
            "只要完成了补正告知，是否就可以同步启动审查？",
            f"不可以。未完成补正前不审查。[{first_key}]",
            quote_secondary,
        ),
    ]
    task_type, question, answer, quote = variants[candidate_idx % len(variants)]
    answerable = task_type != "insufficient"
    payload = {
        "items": [
            {
                "question": question,
                "task_type": task_type,
                "answerable": answerable,
                "answer": answer,
                "citations": [first_key],
                "evidence": [
                    {
                        "source_key": first_key,
                        "quote": quote,
                        "locator": src["locator"],
                    }
                ],
            }
        ]
    }
    return task_type, json.dumps(payload, ensure_ascii=False, indent=2)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def run_probing_generation(
    sources_path: Path,
    prompt_template_path: Path,
    output_path: Path,
    num_candidates_per_source: int = 4,
    mode: str = "mock",
    api_base: str = "http://127.0.0.1:8000",
    model_id: str = "Qwen/Qwen3.5-27B",
    temperature: float = 0.8,
    max_tokens: int = 1024,
    seed: int = 42,
) -> dict[str, Any]:
    """Generates K probing candidates per source record and writes validated JSONL."""
    sources = load_jsonl(sources_path)
    template_text = prompt_template_path.read_text(encoding="utf-8")
    template_hash = sha256_text(template_text)[:16]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    valid_count = 0
    rejected_count = 0

    for src_idx, src in enumerate(sources):
        doc_id = str(src["doc_id"])
        chunk_id = str(src.get("chunk_id") or f"{doc_id}_{src.get('locator', src_idx)}")
        source_bundle, source_map = format_source_bundle([src])

        for k in range(num_candidates_per_source):
            call_seed = seed + src_idx * 100 + k
            if mode == "mock":
                preferred_type, raw_response = generate_mock_response(source_map, k)
                prompt_text = render_generation_prompt(template_text, source_bundle, preferred_type)
            elif mode == "vllm":
                preferred_type = TASK_TYPES[(src_idx + k) % len(TASK_TYPES)]
                prompt_text = render_generation_prompt(template_text, source_bundle, preferred_type)
                raw_response = call_vllm_chat(
                    api_base=api_base,
                    model=model_id,
                    prompt=prompt_text,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=call_seed,
                    api_key=os.environ.get("API_KEY", ""),
                )
            else:
                raise ValueError(f"Unsupported generation mode: {mode}")

            candidate_id = f"probe_{doc_id}_k{k}"
            try:
                parsed = extract_json_object(raw_response)
                items = parsed.get("items", [])
                if not isinstance(items, list) or not items:
                    raise ValueError("Empty items list in teacher output")
                first_item = items[0]
                is_valid, errors, enriched_evidence = verify_and_enrich_item(first_item, source_map)
            except Exception as exc:
                first_item = {}
                is_valid = False
                errors = [f"parse_error:{exc}"]
                enriched_evidence = []

            if is_valid:
                valid_count += 1
                sft_messages = build_student_sft_messages(
                    source_bundle=source_bundle,
                    question=str(first_item["question"]).strip(),
                    answer=str(first_item["answer"]).strip(),
                )
            else:
                rejected_count += 1
                sft_messages = []

            record = {
                "sample_id": candidate_id,
                "prompt_group_id": f"group_{chunk_id}",
                "source_doc_ids": [doc_id],
                "source_chunk_ids": [chunk_id],
                "teacher_prompt": prompt_text,
                "teacher_response": raw_response,
                "question": str(first_item.get("question", "")).strip(),
                "task_type": str(first_item.get("task_type", "")).strip(),
                "answerable": first_item.get("answerable"),
                "answer": str(first_item.get("answer", "")).strip(),
                "citations": first_item.get("citations", []),
                "evidence": enriched_evidence,
                "sft_messages": sft_messages,
                "teacher": {
                    "model_id": model_id if mode == "vllm" else "mock_teacher",
                    "mode": mode,
                    "temperature": temperature,
                    "seed": call_seed,
                    "prompt_template_sha256": template_hash,
                },
                "checks": {
                    "programmatic_valid": is_valid,
                    "errors": errors,
                },
                "rejection_reason": "; ".join(errors) if errors else None,
                "split": src.get("split", "train"),
            }
            records.append(record)

    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "total_sources": len(sources),
        "num_candidates_per_source": num_candidates_per_source,
        "total_generated": len(records),
        "valid_candidates": valid_count,
        "rejected_candidates": rejected_count,
        "output_path": str(output_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True, help="Input source_documents JSONL")
    parser.add_argument("--prompt-template", type=Path, required=True, help="Path to qa_generate.md")
    parser.add_argument("--output", type=Path, required=True, help="Output probing candidates JSONL")
    parser.add_argument("--num-candidates", type=int, default=4, help="Candidates K per source chunk")
    parser.add_argument("--mode", choices=["mock", "vllm"], default="mock", help="Teacher backend mode")
    parser.add_argument("--api-base", default=os.environ.get("API_BASE", "http://127.0.0.1:8000"))
    parser.add_argument("--model-id", default=os.environ.get("SERVED_MODEL_NAME", "Qwen/Qwen3.5-27B"))
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    summary = run_probing_generation(
        sources_path=args.sources,
        prompt_template_path=args.prompt_template,
        output_path=args.output,
        num_candidates_per_source=args.num_candidates,
        mode=args.mode,
        api_base=args.api_base,
        model_id=args.model_id,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
