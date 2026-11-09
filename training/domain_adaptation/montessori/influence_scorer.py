#!/usr/bin/env python3
"""Stage 2: Local Data Influence Scorer for Montessori-Instruct.

Computes the Local Data Influence of each probing candidate (x_i, y_i) on a student
model with respect to a held-out reference validation set D_ref:

    I(x_i, y_i; D_ref) = L(D_ref; theta) - L(D_ref; theta - eta * grad_theta L(x_i, y_i; theta))
                       ≈ eta * <grad_theta L(D_ref; theta), grad_theta L(x_i, y_i; theta)>

Supports two execution modes:
1. `one_step_loss_diff` (Default / Recommended):
   - Computes baseline validation loss L(D_ref; theta) once.
   - For each candidate, computes grad_theta L(x_i, y_i; theta) over trainable
     parameters (LoRA / target linear modules), applies an in-place step
     `p.data.add_(p.grad, alpha=-eta)`, evaluates L(D_ref; theta'), and restores
     `p.data` from an exact parameter snapshot.
2. `grad_dot`:
   - Caches the normalized validation gradient vector `g_ref = grad_theta L(D_ref; theta)`
     once, then scores each candidate via `eta * dot(g_ref, g_cand)` in a single
     backward pass per candidate without re-evaluating D_ref.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def extract_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Extracts SFT messages from either `sft_messages` or `messages`."""
    msgs = record.get("sft_messages") or record.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ValueError(f"Record {record.get('sample_id', '<unknown>')} has no valid messages")
    return msgs


def encode_assistant_supervised_example(
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_length: int = 4096,
) -> dict[str, list[int]]:
    """Tokenizes a conversation and masks non-assistant tokens with -100."""
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Last message must have role='assistant'")

    prefix_messages = messages[:-1]
    if hasattr(tokenizer, "apply_chat_template"):
        prefix_ids = tokenizer.apply_chat_template(
            prefix_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    else:
        prefix_text = "\n".join(f"{m['role']}: {m['content']}" for m in prefix_messages) + "\nassistant: "
        full_text = prefix_text + messages[-1]["content"]
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
    prefix_len = min(len(prefix_ids), len(full_ids))
    labels = [-100] * prefix_len + list(full_ids[prefix_len:])
    attention_mask = [1] * len(full_ids)
    return {
        "input_ids": list(full_ids),
        "labels": labels,
        "attention_mask": attention_mask,
    }


def compute_mock_influence(record: dict[str, Any], probe_lr: float) -> tuple[float, float, float]:
    """Deterministic influence simulator for unit tests and dry runs without GPU."""
    baseline_loss = 2.150000
    checks = record.get("checks", {})
    if not checks.get("programmatic_valid", True):
        # Invalid or hallucinated samples degrade validation loss (negative influence)
        perturbed_loss = baseline_loss + 0.045
        return baseline_loss, perturbed_loss, round(baseline_loss - perturbed_loss, 8)

    question = str(record.get("question", ""))
    answer = str(record.get("answer", ""))
    task_type = str(record.get("task_type", ""))
    citations = record.get("citations", [])

    # Reward rigorous, well-cited answers with clear condition/boundary explanations
    quality_signal = 0.0
    if citations and all(f"[{c}]" in answer for c in citations):
        quality_signal += 0.8
    if "处理顺序" in answer or "无法判定" in answer or "前提条件" in answer:
        quality_signal += 0.6
    if task_type in ("condition", "insufficient", "synthesis"):
        quality_signal += 0.4
    if len(answer) < 45:
        quality_signal -= 2.0

    # Add deterministic micro-jitter from content hash so ties break cleanly
    digest = hashlib.sha256(f"{question}||{answer}".encode("utf-8")).hexdigest()
    jitter = ((int(digest[:8], 16) / 0xFFFFFFFF) - 0.5) * 0.2
    raw_delta = (quality_signal + jitter) * (probe_lr * 100.0)
    perturbed_loss = baseline_loss - raw_delta
    influence = baseline_loss - perturbed_loss
    return round(baseline_loss, 8), round(perturbed_loss, 8), round(influence, 8)


class TorchStudentInfluenceEngine:
    """PyTorch engine for computing exact one-step loss difference or gradient dot-product."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cuda",
        probe_lr: float = 1e-4,
        max_length: int = 4096,
        trainable_name_filter: str | None = None,
    ) -> None:
        import torch

        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.probe_lr = probe_lr
        self.max_length = max_length

        self.trainable_params: list[tuple[str, Any]] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if trainable_name_filter and trainable_name_filter not in name:
                continue
            self.trainable_params.append((name, param))

        if not self.trainable_params:
            raise ValueError("No trainable parameters found on student model for influence probing")

    def _to_tensor_batch(self, encoded: dict[str, list[int]]) -> dict[str, Any]:
        torch = self.torch
        return {
            "input_ids": torch.tensor([encoded["input_ids"]], dtype=torch.long, device=self.device),
            "labels": torch.tensor([encoded["labels"]], dtype=torch.long, device=self.device),
            "attention_mask": torch.tensor([encoded["attention_mask"]], dtype=torch.long, device=self.device),
        }

    def evaluate_ref_loss(self, encoded_refs: list[dict[str, list[int]]]) -> float:
        """Evaluates mean token-level cross-entropy loss over D_ref without gradients."""
        torch = self.torch
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for enc in encoded_refs:
                batch = self._to_tensor_batch(enc)
                outputs = self.model(**batch)
                total_loss += float(outputs.loss.detach().cpu().item())
        return total_loss / max(len(encoded_refs), 1)

    def score_candidates_one_step(
        self,
        encoded_candidates: list[dict[str, list[int]] | None],
        encoded_refs: list[dict[str, list[int]]],
    ) -> list[tuple[float, float, float]]:
        """Computes I(z) = L(D_ref; theta) - L(D_ref; theta - eta * grad L(z; theta))."""
        torch = self.torch
        self.model.eval()
        baseline_ref_loss = self.evaluate_ref_loss(encoded_refs)

        # Save exact copy of trainable weights
        saved_weights = {
            name: param.detach().clone() for name, param in self.trainable_params
        }

        results: list[tuple[float, float, float]] = []
        for enc_cand in encoded_candidates:
            if enc_cand is None:
                results.append((baseline_ref_loss, baseline_ref_loss, 0.0))
                continue

            self.model.zero_grad(set_to_none=True)
            batch = self._to_tensor_batch(enc_cand)
            loss = self.model(**batch).loss
            loss.backward()

            with torch.no_grad():
                for _, param in self.trainable_params:
                    if param.grad is not None:
                        param.data.add_(param.grad, alpha=-self.probe_lr)

            perturbed_ref_loss = self.evaluate_ref_loss(encoded_refs)
            influence = baseline_ref_loss - perturbed_ref_loss
            results.append((baseline_ref_loss, perturbed_ref_loss, influence))

            # Restore exact baseline parameters
            with torch.no_grad():
                for name, param in self.trainable_params:
                    param.data.copy_(saved_weights[name])

        self.model.zero_grad(set_to_none=True)
        return results

    def score_candidates_grad_dot(
        self,
        encoded_candidates: list[dict[str, list[int]] | None],
        encoded_refs: list[dict[str, list[int]]],
    ) -> list[tuple[float, float, float]]:
        """Computes first-order Taylor influence I(z) ≈ eta * <grad L(D_ref), grad L(z)>."""
        torch = self.torch
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        total_ref_loss = 0.0
        n_refs = max(len(encoded_refs), 1)
        for enc in encoded_refs:
            batch = self._to_tensor_batch(enc)
            loss = self.model(**batch).loss / n_refs
            total_ref_loss += float(loss.detach().cpu().item())
            loss.backward()

        ref_grads = {
            name: (
                param.grad.detach().clone()
                if param.grad is not None
                else torch.zeros_like(param.data)
            )
            for name, param in self.trainable_params
        }
        self.model.zero_grad(set_to_none=True)

        results: list[tuple[float, float, float]] = []
        for enc_cand in encoded_candidates:
            if enc_cand is None:
                results.append((total_ref_loss, total_ref_loss, 0.0))
                continue

            self.model.zero_grad(set_to_none=True)
            batch = self._to_tensor_batch(enc_cand)
            cand_loss = self.model(**batch).loss
            cand_loss.backward()

            dot_val = 0.0
            with torch.no_grad():
                for name, param in self.trainable_params:
                    if param.grad is not None:
                        dot_val += float(
                            torch.sum(ref_grads[name].float() * param.grad.detach().float()).cpu().item()
                        )
            influence = self.probe_lr * dot_val
            approx_perturbed_loss = total_ref_loss - influence
            results.append((total_ref_loss, approx_perturbed_loss, influence))

        self.model.zero_grad(set_to_none=True)
        return results


def build_hf_student_engine(
    model_name_or_path: str,
    adapter_path: str | None,
    device: str,
    probe_lr: float,
    max_length: int,
    trainable_filter: str | None,
    torch_dtype_str: str = "bfloat16",
) -> TorchStudentInfluenceEngine:
    """Loads a HuggingFace causal LM (with optional LoRA adapter) for influence scoring."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype_str, torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        # Freeze vision/merger by default and only enable gradients on target linear layers
        for name, param in model.named_parameters():
            if "visual" in name or "vision" in name or "merger" in name:
                param.requires_grad_(False)
            elif trainable_filter and trainable_filter not in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

    return TorchStudentInfluenceEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        probe_lr=probe_lr,
        max_length=max_length,
        trainable_name_filter=trainable_filter,
    )


def run_influence_scoring(
    candidates_path: Path,
    ref_dataset_path: Path,
    output_path: Path,
    backend: str = "mock",
    method: str = "one_step_loss_diff",
    student_model: str = "Qwen/Qwen3.5-27B",
    student_adapter: str | None = None,
    device: str = "cuda",
    probe_lr: float = 1e-4,
    max_length: int = 4096,
    trainable_filter: str | None = "lora",
) -> dict[str, Any]:
    """Scores each candidate in `candidates_path` against `ref_dataset_path`."""
    candidates = load_jsonl(candidates_path)
    ref_records = load_jsonl(ref_dataset_path)
    if not ref_records:
        raise ValueError(f"Reference validation set is empty: {ref_dataset_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if backend == "mock":
        scores = [compute_mock_influence(rec, probe_lr=probe_lr) for rec in candidates]
    elif backend == "torch":
        engine = build_hf_student_engine(
            model_name_or_path=student_model,
            adapter_path=student_adapter,
            device=device,
            probe_lr=probe_lr,
            max_length=max_length,
            trainable_filter=trainable_filter,
        )
        encoded_refs = [
            encode_assistant_supervised_example(
                engine.tokenizer,
                extract_messages(ref_rec),
                max_length=max_length,
            )
            for ref_rec in ref_records
        ]
        encoded_cands: list[dict[str, list[int]] | None] = []
        for rec in candidates:
            if not rec.get("checks", {}).get("programmatic_valid", True):
                encoded_cands.append(None)
            else:
                encoded_cands.append(
                    encode_assistant_supervised_example(
                        engine.tokenizer,
                        extract_messages(rec),
                        max_length=max_length,
                    )
                )
        if method == "one_step_loss_diff":
            scores = engine.score_candidates_one_step(encoded_cands, encoded_refs)
        elif method == "grad_dot":
            scores = engine.score_candidates_grad_dot(encoded_cands, encoded_refs)
        else:
            raise ValueError(f"Unsupported influence method: {method}")
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    positive_count = 0
    negative_count = 0
    scored_records: list[dict[str, Any]] = []

    for rec, (base_loss, pert_loss, inf_score) in zip(candidates, scores):
        rec_copy = dict(rec)
        rec_copy["influence"] = {
            "score": inf_score,
            "baseline_ref_loss": base_loss,
            "perturbed_ref_loss": pert_loss,
            "method": method,
            "backend": backend,
            "probe_lr": probe_lr,
            "student_model": student_model if backend == "torch" else "mock_student",
            "ref_samples": len(ref_records),
        }
        if inf_score > 0:
            positive_count += 1
        elif inf_score < 0:
            negative_count += 1
        scored_records.append(rec_copy)

    with output_path.open("w", encoding="utf-8") as f:
        for rec in scored_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = max(len(scored_records), 1)
    return {
        "total_candidates": len(scored_records),
        "positive_influence_count": positive_count,
        "negative_influence_count": negative_count,
        "positive_influence_ratio": round(positive_count / total, 4),
        "method": method,
        "backend": backend,
        "output_path": str(output_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True, help="Input probing candidates JSONL")
    parser.add_argument("--ref-dataset", type=Path, required=True, help="Held-out validation JSONL (D_ref)")
    parser.add_argument("--output", type=Path, required=True, help="Output scored candidates JSONL")
    parser.add_argument("--backend", choices=["mock", "torch"], default="mock")
    parser.add_argument(
        "--method",
        choices=["one_step_loss_diff", "grad_dot"],
        default="one_step_loss_diff",
        help="Influence calculation method",
    )
    parser.add_argument("--student-model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--student-adapter", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--trainable-filter", default=None, help="Substring filter for trainable param names")
    args = parser.parse_args(argv)

    summary = run_influence_scoring(
        candidates_path=args.candidates,
        ref_dataset_path=args.ref_dataset,
        output_path=args.output,
        backend=args.backend,
        method=args.method,
        student_model=args.student_model,
        student_adapter=args.student_adapter,
        device=args.device,
        probe_lr=args.probe_lr,
        max_length=args.max_length,
        trainable_filter=args.trainable_filter,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
