#!/usr/bin/env bash
set -euo pipefail

# End-to-End Montessori-Instruct Orchestrator for Domain QA Synthesis:
#   Step 1: Probing QA generation (Teacher -> K candidates per source chunk)
#   Step 2: Student Local Data Influence scoring (One-step Loss Diff / Grad Dot on D_ref)
#   Step 3: Preference pair construction (DPO pairs for Teacher + SFT rows for Student)
#   Step 4: Teacher DPO alignment command generation / execution (MS-SWIFT)
#   Step 5: Student SFT command generation / execution (MS-SWIFT)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOMAIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SWIFT_DIR="$(cd "$DOMAIN_DIR/../swift" && pwd)"

SOURCES=${SOURCES:-"$DOMAIN_DIR/examples/source_documents_example.jsonl"}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-"$DOMAIN_DIR/prompts/qa_generate.md"}
REF_DATASET=${REF_DATASET:-"$DOMAIN_DIR/examples/style_sft_example.jsonl"}
WORK_DIR=${WORK_DIR:-"$SCRIPT_DIR/output_demo"}

MODE=${MODE:-mock}                      # mock | vllm
INFLUENCE_BACKEND=${INFLUENCE_BACKEND:-mock}  # mock | torch
INFLUENCE_METHOD=${INFLUENCE_METHOD:-one_step_loss_diff} # one_step_loss_diff | grad_dot
NUM_CANDIDATES=${NUM_CANDIDATES:-4}
PROBE_LR=${PROBE_LR:-0.0001}
DRY_RUN=${DRY_RUN:-1}

mkdir -p "$WORK_DIR"

PROBING_JSONL="$WORK_DIR/probing_candidates.jsonl"
SCORED_JSONL="$WORK_DIR/scored_candidates.jsonl"
TEACHER_DPO_JSONL="$WORK_DIR/teacher_dpo_pairs.jsonl"
STUDENT_SFT_JSONL="$WORK_DIR/student_sft_messages.jsonl"
PROVENANCE_JSONL="$WORK_DIR/student_sft_provenance.jsonl"

echo "=== [1/6] Generating Probing QA Candidates (mode=$MODE, K=$NUM_CANDIDATES) ==="
python3 "$SCRIPT_DIR/probing_generate.py" \
  --sources "$SOURCES" \
  --prompt-template "$PROMPT_TEMPLATE" \
  --output "$PROBING_JSONL" \
  --num-candidates "$NUM_CANDIDATES" \
  --mode "$MODE"

echo "=== [2/6] Scoring Local Data Influence on Student (backend=$INFLUENCE_BACKEND, method=$INFLUENCE_METHOD) ==="
python3 "$SCRIPT_DIR/influence_scorer.py" \
  --candidates "$PROBING_JSONL" \
  --ref-dataset "$REF_DATASET" \
  --output "$SCORED_JSONL" \
  --backend "$INFLUENCE_BACKEND" \
  --method "$INFLUENCE_METHOD" \
  --probe-lr "$PROBE_LR"

echo "=== [3/6] Building Teacher DPO Pairs & Student SFT Dataset ==="
python3 "$SCRIPT_DIR/build_dpo_pairs.py" \
  --scored-candidates "$SCORED_JSONL" \
  --dpo-output "$TEACHER_DPO_JSONL" \
  --sft-output "$STUDENT_SFT_JSONL" \
  --provenance-output "$PROVENANCE_JSONL"

echo "=== [4/6] Synthesizing API Black-Box Steering Prompts & Multi-Format Student Preference Data (DPO/KTO/GRPO) ==="
python3 "$SCRIPT_DIR/api_preference_synthesizer.py" \
  --sources "$SOURCES" \
  --scored-candidates "$SCORED_JSONL" \
  --prompt-template "$PROMPT_TEMPLATE" \
  --output-dir "$WORK_DIR/api_preference"

echo "=== [5/6] Teacher DPO Alignment (MS-SWIFT, DRY_RUN=$DRY_RUN) ==="
DATASET="$TEACHER_DPO_JSONL" OUTPUT_DIR="$WORK_DIR/teacher_dpo_ckpt" DRY_RUN="$DRY_RUN" \
  bash "$SCRIPT_DIR/train_teacher_dpo_swift.sh"

echo "=== [6/6] Student SFT on High-Influence Data (MS-SWIFT, DRY_RUN=$DRY_RUN) ==="
DATASET="$STUDENT_SFT_JSONL" OUTPUT_DIR="$WORK_DIR/student_sft_ckpt" DRY_RUN="$DRY_RUN" \
  bash "$SWIFT_DIR/train_qwen35_27b_swift.sh"

