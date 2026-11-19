#!/usr/bin/env bash
set -euo pipefail

# Phase A (Lite Gain Control) End-to-End Orchestrator:
#   LITE_MODE=proportional_flash (default):
#     - Slices Flash's sanitized 4-bucket Mid-Training pool proportionally (--ratio 0.15, whole families, val0)
#     - Runs calibrated B300 Lite CPT (MBS=1, GBS=4, MAX_LENGTH=8192, constant 8e-6, SAVE_STEPS=40 -> ~100-150 steps)
#   LITE_MODE=sch_legacy (fallback):
#     - Legacy sch iso build via `build_cpt_from_sch.py`
#   Step 3:
#     - Exports HF BF16 checkpoint & runs Dual-Arm External OOD SFT + Dual-Mode (Think-ON/OFF) Eval

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FLASH_DIR="$TRAINING_DIR/flash"
CPT27B_DIR="$TRAINING_DIR/cpt_27b"

WORK_DIR=${WORK_DIR:-"${TMPDIR:-/tmp}/cpt_sft_lite_pipeline"}
MODEL=${MODEL:-/data/autovision-cfs/autovision-notebook/algo4/models/Qwen3.5-27B}
LITE_MODE=${LITE_MODE:-proportional_flash} # proportional_flash | sch_legacy
LITE_RATIO=${LITE_RATIO:-0.15}
OOD_SFT_DATASET=${OOD_SFT_DATASET:-"$SCRIPT_DIR/examples/lite_sch_kb_sample.jsonl"}
DEMO=${DEMO:-0}
DRY_RUN=${DRY_RUN:-1}

mkdir -p "$WORK_DIR"

DATA_DIR="$WORK_DIR/pt"
CPT_OUT_DIR="$WORK_DIR/cpt_lite_ckpt/stage1"
CPT_HF_DIR="$WORK_DIR/cpt_lite_hf"
ARM_A_OUT="$WORK_DIR/sft_arm_a_cpt_ood"
ARM_B_OUT="$WORK_DIR/sft_arm_b_base_ood"

if [[ "$LITE_MODE" == "proportional_flash" ]]; then
  echo "=== [1/3] Slicing Flash Mid-Training Pool -> Lite (ratio=$LITE_RATIO, val0) ==="
  FLASH_SRC_TRAIN="${FLASH_MIDTRAIN_INPUT:-}"
  if [[ -z "$FLASH_SRC_TRAIN" ]]; then
    if [[ "$DEMO" == "1" || "$DRY_RUN" == "1" ]]; then
      FLASH_STAGE_DIR="$WORK_DIR/flash_source"
      mkdir -p "$FLASH_STAGE_DIR"
      python3 "$FLASH_DIR/gen_cot_from_qa.py" \
        --input "$FLASH_DIR/examples/qa_kb_v04_example.jsonl" \
        --cot-mode internalized \
        --output-dir "$FLASH_STAGE_DIR/cot" >/dev/null
      python3 "$FLASH_DIR/build_v2_image_pipeline.py" \
        --entities "$FLASH_DIR/examples/v2_image_entities_sample.jsonl" \
        --v1-audit "$FLASH_DIR/examples/v2_image_v1_audit_sample.jsonl" \
        --output-dir "$FLASH_STAGE_DIR/v2_img" >/dev/null
      python3 "$FLASH_DIR/refactor_v1_v4_pipeline.py" \
        --v1-v4-inputs "$FLASH_DIR/examples/v1_v4_legacy_sample.jsonl" \
        --new-qa-inputs "$FLASH_STAGE_DIR/cot/qa_sft_mixed_8_2.jsonl" "$FLASH_DIR/examples/style_sft_example.jsonl" \
        --output-dir "$FLASH_STAGE_DIR/refactor" \
        --val-ratio 0.25 >/dev/null
      python3 "$FLASH_DIR/build_mm_wiki_cpt.py" \
        --mode midtrain \
        --inputs \
          "$FLASH_DIR/examples/cpt_mm_6ch_example.jsonl" \
          "$FLASH_STAGE_DIR/refactor/cpt_v1_v4_replay.jsonl" \
          "$FLASH_STAGE_DIR/v2_img/v2_packed_cpt_docs.jsonl" \
          "$FLASH_STAGE_DIR/v2_img/v2_cf_neg_pairs.jsonl" \
          "$FLASH_STAGE_DIR/cot/qa_midtrain_prose.jsonl" \
        --exclude-families-from "$FLASH_STAGE_DIR/refactor/sft_unified_val.jsonl" \
        --output-dir "$FLASH_STAGE_DIR/midtrain" \
        --val-ratio 0.15 \
        --total-rows 28 \
        --max-tokens 8192 \
        --max-row-repeats 2 >/dev/null
      FLASH_SRC_TRAIN="$FLASH_STAGE_DIR/midtrain/midtrain_train.jsonl"
    else
      echo "Set FLASH_MIDTRAIN_INPUT=<path/to/midtrain_train.jsonl> when LITE_MODE=proportional_flash (or pass DEMO=1)." >&2
      exit 2
    fi
  fi
  python3 "$SCRIPT_DIR/slice_flash_to_lite.py" \
    --input "$FLASH_SRC_TRAIN" \
    --ratio "$LITE_RATIO" \
    --epochs "${LITE_EPOCHS:-2}" \
    --output-dir "$DATA_DIR" >/dev/null
elif [[ "$LITE_MODE" == "sch_legacy" ]]; then
  echo "=== [1/3] Building Legacy Lite Bundle (sch_legacy, bundle_lite_v3_all val0) ==="
  build_cmd=(python3 "$SCRIPT_DIR/build_cpt_from_sch.py" --output-dir "$DATA_DIR")
  if [[ "$DEMO" == "1" || ( -z ${SCH_INPUT:-} && -z ${KB_INPUT:-} && "$DRY_RUN" == "1" ) ]]; then
    build_cmd+=(--demo)
  else
    [[ -n ${SCH_INPUT:-} ]] && build_cmd+=(--sch-input "$SCH_INPUT")
    [[ -n ${KB_INPUT:-} ]] && build_cmd+=(--kb-input "$KB_INPUT")
    [[ -n ${KB_SUPPLEMENT:-} ]] && build_cmd+=(--kb-supplement "$KB_SUPPLEMENT")
  fi
  "${build_cmd[@]}" >/dev/null
else
  echo "Unsupported LITE_MODE: $LITE_MODE (expected proportional_flash or sch_legacy)" >&2
  exit 2
fi

LITE_DATASET="$DATA_DIR/bundle_lite_compat/stage1_train.jsonl"

echo "=== [2/3] B300 Calibrated Lite CPT (constant 8e-6, MBS=1, GBS=4, 8192 len, DRY_RUN=$DRY_RUN) ==="
launch_args=(lite --model "$MODEL" --dataset "$LITE_DATASET" --output-dir "$CPT_OUT_DIR")
if [[ "$DRY_RUN" == "1" ]]; then
  launch_args+=(--dry-run)
else
  launch_args+=(--launch)
fi
bash "$CPT27B_DIR/run_cpt_27b_b300.sh" "${launch_args[@]}"

echo "=== [3/3] Dual-Arm External OOD SFT Comparison (Arm A: CPT+OOD vs Arm B: Base+OOD, DRY_RUN=$DRY_RUN) ==="
export_cmd=(
  swift export
  --mcore_model "$CPT_OUT_DIR/latest"
  --to_hf true
  --torch_dtype bfloat16
  --output_dir "$CPT_HF_DIR"
)

arm_a_cmd=(
  megatron sft
  --model "$CPT_HF_DIR"
  --dataset "$OOD_SFT_DATASET"
  --tuner_type lora
  --target_modules all-linear
  --lora_rank "${SFT_LORA_RANK:-32}"
  --lora_alpha "${SFT_LORA_ALPHA:-64}"
  --torch_dtype bfloat16
  --tensor_model_parallel_size "${TP_SIZE:-4}"
  --micro_batch_size "${SFT_MBS:-2}"
  --global_batch_size "${SFT_GBS:-32}"
  --max_length "${SFT_MAX_LENGTH:-8192}"
  --lr "${SFT_LR:-5e-6}"
  --output_dir "$ARM_A_OUT"
)

arm_b_cmd=(
  megatron sft
  --model "$MODEL"
  --dataset "$OOD_SFT_DATASET"
  --tuner_type lora
  --target_modules all-linear
  --lora_rank "${SFT_LORA_RANK:-32}"
  --lora_alpha "${SFT_LORA_ALPHA:-64}"
  --torch_dtype bfloat16
  --tensor_model_parallel_size "${TP_SIZE:-4}"
  --micro_batch_size "${SFT_MBS:-2}"
  --global_batch_size "${SFT_GBS:-32}"
  --max_length "${SFT_MAX_LENGTH:-8192}"
  --lr "${SFT_LR:-5e-6}"
  --output_dir "$ARM_B_OUT"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '%q ' "${export_cmd[@]}"; printf '\n'
  printf '%q ' "${arm_a_cmd[@]}"; printf '\n'
  printf '%q ' "${arm_b_cmd[@]}"; printf '\n'
  echo "=== Dual-Mode Retention Gate: compare Arm A vs Arm B under BOTH enable_thinking=true AND enable_thinking=false ==="
  exit 0
fi

"${export_cmd[@]}"
"${arm_a_cmd[@]}"
"${arm_b_cmd[@]}"
