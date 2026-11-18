#!/usr/bin/env bash
set -euo pipefail

# Phase B (Flash Full-Scale WSD CPT + QA->CoT v0.3 + Three-Arm Unified SFT) Orchestrator:
#   Step 1: Build 6+2 channel Multimodal CPT bundle (`build_mm_wiki_cpt.py` + `refactor_v1_v4_pipeline.py`)
#   Step 2: Run Two-Phase WSD CPT via `cpt_27b/run_cpt_27b_b300.sh` (stage1 constant -> stage2 linear WSD)
#   Step 3: Export merged BF16 HF base for Unified SFT
#   Step 4: Generate v0.3 QA-CoT + Direct fallback (`gen_cot_from_qa.py`, `qwen3.8-flash` thinking >= 2048)
#   Step 5: Build & launch Three-Arm SFT (`ood_only` | `ood_plus_qa_direct` | `ood_plus_qa_cot` 8:2)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CPT27B_DIR="$TRAINING_DIR/cpt_27b"

WORK_DIR=${WORK_DIR:-"${TMPDIR:-/tmp}/cpt_sft_flash_pipeline"}
MODEL=${MODEL:-/data/autovision-cfs/autovision-notebook/algo4/models/Qwen3.5-27B}
BACKEND=${BACKEND:-megatron}          # megatron | swift
CPT_TUNER_TYPE=${CPT_TUNER_TYPE:-full} # full | lora
SFT_ARM=${SFT_ARM:-ood_plus_qa_cot}   # unified | ood_only | ood_plus_qa_direct | ood_plus_qa_cot
COT_MODE=${COT_MODE:-mock}            # mock | vllm
SMOKE=${SMOKE:-0}
DRY_RUN=${DRY_RUN:-1}
DEMO=${DEMO:-0}

if [[ "$MODEL" == *Flash-Next* || "$MODEL" == *qwen4_exp* ]]; then
  echo 'Use training/swift/train_b300_swift.sh for Qwen3.8-Flash-Next; Flash pipeline targets 27B.' >&2
  exit 2
fi

if [[ "$DEMO" == "1" || ( -z ${V1_V4_INPUTS:-} && "$DRY_RUN" == "1" ) ]]; then
  V1_V4_INPUTS=${V1_V4_INPUTS:-"$SCRIPT_DIR/examples/v1_v4_legacy_sample.jsonl"}
  WIKI_CPT_INPUTS=${WIKI_CPT_INPUTS:-"$SCRIPT_DIR/examples/cpt_mm_6ch_example.jsonl"}
  QA_RAW_INPUT=${QA_RAW_INPUT:-"$SCRIPT_DIR/examples/qa_kb_raw_example.jsonl"}
  DEMO_MODE=1
else
  : "${V1_V4_INPUTS:?Set space-separated V1_V4_INPUTS (or pass DEMO=1)}"
  : "${WIKI_CPT_INPUTS:?Set space-separated WIKI_CPT_INPUTS (or pass DEMO=1)}"
  QA_RAW_INPUT=${QA_RAW_INPUT:-""}
  DEMO_MODE=0
fi

read -r -a v1_v4_arr <<< "$V1_V4_INPUTS"
read -r -a wiki_arr <<< "$WIKI_CPT_INPUTS"

COT_DIR="$WORK_DIR/step0_qa_cot_v03"
REFACTOR_DIR="$WORK_DIR/step1_v1_v4_refactored"
CPT_DATA_DIR="$WORK_DIR/step2_cpt_bundle"
CPT_OUT_DIR="$WORK_DIR/step3_cpt_ckpt"
CPT_EXPORT_DIR="$WORK_DIR/step4_cpt_hf_merged"
SFT_OUT_DIR="$WORK_DIR/step5_flash_sft_${SFT_ARM}"

mkdir -p "$WORK_DIR"

echo "=== [1/5] Generating QA->CoT v0.3 & Direct Fallback (mode=$COT_MODE) ==="
cot_cmd=(python3 "$SCRIPT_DIR/gen_cot_from_qa.py" --output-dir "$COT_DIR" --mode "$COT_MODE")
if [[ -n "$QA_RAW_INPUT" ]]; then
  cot_cmd+=(--input "$QA_RAW_INPUT")
else
  cot_cmd+=(--demo)
fi
if [[ "$DEMO_MODE" == "1" ]]; then
  cot_cmd+=(--smoke-sample 6)
fi
"${cot_cmd[@]}" >/dev/null

NEW_QA_INPUTS=${NEW_QA_INPUTS:-"$COT_DIR/qa_sft_mixed_8_2.jsonl $SCRIPT_DIR/examples/style_sft_example.jsonl"}
read -r -a new_qa_arr <<< "$NEW_QA_INPUTS"

echo "=== [2/5] Refactoring V1-V4 + QA-CoT into CPT Replay & SFT Arm ($SFT_ARM) ==="
refactor_cmd=(
  python3 "$SCRIPT_DIR/refactor_v1_v4_pipeline.py"
  --v1-v4-inputs "${v1_v4_arr[@]}"
  --new-qa-inputs "${new_qa_arr[@]}"
  --sft-arm "$SFT_ARM"
  --cot-direct-ratio "${COT_DIRECT_RATIO:-0.8}"
  --output-dir "$REFACTOR_DIR"
)
if [[ -n ${RAG_EVIDENCE_MAP:-} ]]; then
  refactor_cmd+=(--rag-evidence-map "$RAG_EVIDENCE_MAP")
fi
if [[ "$DEMO_MODE" == "1" ]]; then
  refactor_cmd+=(--val-ratio 0.25 --sft-target-rows 24)
elif [[ -n ${SFT_TARGET_ROWS:-} ]]; then
  refactor_cmd+=(--sft-target-rows "$SFT_TARGET_ROWS")
fi
"${refactor_cmd[@]}" >/dev/null

echo "=== [3/5] Building Full WSD CPT Bundle (bundle v2.3 with sch support) ==="
cpt_build_cmd=(
  python3 "$SCRIPT_DIR/build_mm_wiki_cpt.py"
  --inputs "${wiki_arr[@]}" "$REFACTOR_DIR/cpt_v1_v4_replay.jsonl"
  --output-dir "$CPT_DATA_DIR"
  --max-row-repeats "${MAX_ROW_REPEATS:-3}"
)
if [[ "$DEMO_MODE" == "1" ]]; then
  cpt_build_cmd+=(--val-ratio 0.20 --stage1-rows 24 --stage2-rows 16)
else
  [[ -n ${CPT_STAGE1_ROWS:-} ]] && cpt_build_cmd+=(--stage1-rows "$CPT_STAGE1_ROWS")
  [[ -n ${CPT_STAGE2_ROWS:-} ]] && cpt_build_cmd+=(--stage2-rows "$CPT_STAGE2_ROWS")
fi
"${cpt_build_cmd[@]}" >/dev/null

echo "=== [4/5] Running Two-Phase WSD CPT (stage1 constant -> stage2 linear WSD, DRY_RUN=$DRY_RUN) ==="
STAGE2_DECAY_ITERS=${STAGE2_DECAY_ITERS:-28}
launch_flag="--dry-run"
[[ "$DRY_RUN" == "0" ]] && launch_flag="--launch"

BACKEND="$BACKEND" TUNER_TYPE="$CPT_TUNER_TYPE" SMOKE="$SMOKE" \
  bash "$CPT27B_DIR/run_cpt_27b_b300.sh" stage1 \
  --model "$MODEL" \
  --dataset "$CPT_DATA_DIR/stage1_train.jsonl" \
  --val-dataset "$CPT_DATA_DIR/val.jsonl" \
  --output-dir "$CPT_OUT_DIR/stage1_stable" \
  "$launch_flag"

BACKEND="$BACKEND" TUNER_TYPE="$CPT_TUNER_TYPE" SMOKE="$SMOKE" \
  bash "$CPT27B_DIR/run_cpt_27b_b300.sh" stage2 \
  --model "$CPT_OUT_DIR/stage1_stable/latest" \
  --dataset "$CPT_DATA_DIR/stage2_cooldown.jsonl" \
  --val-dataset "$CPT_DATA_DIR/val.jsonl" \
  --output-dir "$CPT_OUT_DIR/stage2_cooldown" \
  --stage2-decay-iters "$STAGE2_DECAY_ITERS" \
  "$launch_flag"

echo "=== [5/5] Exporting CPT BF16 Base & Launching Unified SFT ($SFT_ARM, DRY_RUN=$DRY_RUN) ==="
if [[ "$CPT_TUNER_TYPE" == "lora" ]]; then
  export_cmd=(swift export --model "$MODEL" --adapters "$CPT_OUT_DIR/stage2_cooldown/latest" --merge_lora true --output_dir "$CPT_EXPORT_DIR")
else
  export_cmd=(swift export --mcore_model "$CPT_OUT_DIR/stage2_cooldown/latest" --to_hf true --torch_dtype bfloat16 --output_dir "$CPT_EXPORT_DIR")
fi

sft_cmd=(
  megatron sft
  --model "$CPT_EXPORT_DIR"
  --dataset "$REFACTOR_DIR/sft_unified_train.jsonl"
  --val_dataset "$REFACTOR_DIR/sft_unified_val.jsonl"
  --split_dataset_ratio 0
  --tuner_type lora
  --target_modules all-linear
  --lora_rank "${SFT_LORA_RANK:-64}"
  --lora_alpha "${SFT_LORA_ALPHA:-128}"
  --lora_dropout "${SFT_LORA_DROPOUT:-0.05}"
  --merge_lora false
  --torch_dtype bfloat16
  --freeze_vit "${SFT_FREEZE_VIT:-true}"
  --freeze_aligner "${SFT_FREEZE_ALIGNER:-false}"
  --aligner_lr "${SFT_ALIGNER_LR:-2e-6}"
  --tensor_model_parallel_size "${TP_SIZE:-4}"
  --pipeline_model_parallel_size "${PP_SIZE:-1}"
  --sequence_parallel true
  --recompute_granularity selective
  --attention_backend "${ATTENTION_BACKEND:-flash}"
  --gradient_accumulation_fusion "${GRADIENT_ACCUMULATION_FUSION:-false}"
  --packing true
  --padding_free true
  --max_length "${SFT_MAX_LENGTH:-16384}"
  --truncation_strategy delete
  --strict true
  --enable_channel_loss true
  --micro_batch_size "${SFT_MBS:-4}"
  --global_batch_size "${SFT_GBS:-64}"
  --optimizer adam
  --lr "${SFT_LR:-1e-5}"
  --min_lr "${SFT_MIN_LR:-1e-6}"
  --lr_decay_style cosine
  --lr_warmup_fraction 0.03
  --weight_decay 0.01
  --clip_grad 1.0
  --save_safetensors true
  --output_dir "$SFT_OUT_DIR"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '%q ' "${export_cmd[@]}"; printf '\n'
  printf 'CUDA_VISIBLE_DEVICES=%q NPROC_PER_NODE=%q IMAGE_MAX_TOKEN_NUM=%q PYTORCH_CUDA_ALLOC_CONF=%q ' \
    "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" "${NPROC_PER_NODE:-8}" "${IMAGE_MAX_TOKEN_NUM:-16384}" "${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  printf '%q ' "${sft_cmd[@]}"; printf '\n'
  exit 0
fi

"${export_cmd[@]}"
exec "${sft_cmd[@]}"
