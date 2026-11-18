#!/usr/bin/env bash
set -euo pipefail

# Phase A (Lite Gain Control) End-to-End Orchestrator:
#   Step 1: Build `bundle_lite_v3_all·val0` + `bundle_lite_compat` via `build_cpt_from_sch.py`
#   Step 2: Launch / Dry-Run B300 27B Lite CPT (`cpt_27b/run_cpt_27b_b300.sh lite`, 2 epochs, constant 1e-5)
#   Step 3: Export HF BF16 checkpoint & print/run dual-arm external OOD SFT evaluation:
#             Arm A: CPT(lite) + external OOD SFT
#             Arm B: base      + same external OOD SFT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CPT27B_DIR="$TRAINING_DIR/cpt_27b"

WORK_DIR=${WORK_DIR:-"${TMPDIR:-/tmp}/cpt_sft_lite_pipeline"}
MODEL=${MODEL:-/data/autovision-cfs/autovision-notebook/algo4/models/Qwen3.5-27B}
OOD_SFT_DATASET=${OOD_SFT_DATASET:-"$SCRIPT_DIR/examples/lite_sch_kb_sample.jsonl"}
DEMO=${DEMO:-0}
DRY_RUN=${DRY_RUN:-1}

mkdir -p "$WORK_DIR"

DATA_DIR="$WORK_DIR/pt"
CPT_OUT_DIR="$WORK_DIR/cpt_lite_ckpt/stage1"
CPT_HF_DIR="$WORK_DIR/cpt_lite_hf"
ARM_A_OUT="$WORK_DIR/sft_arm_a_cpt_ood"
ARM_B_OUT="$WORK_DIR/sft_arm_b_base_ood"

echo "=== [1/3] Building Lite Bundle (bundle_lite_v3_all·val0) ==="
build_cmd=(python3 "$SCRIPT_DIR/build_cpt_from_sch.py" --output-dir "$DATA_DIR")
if [[ "$DEMO" == "1" || ( -z ${SCH_INPUT:-} && -z ${KB_INPUT:-} && "$DRY_RUN" == "1" ) ]]; then
  build_cmd+=(--demo)
else
  [[ -n ${SCH_INPUT:-} ]] && build_cmd+=(--sch-input "$SCH_INPUT")
  [[ -n ${KB_INPUT:-} ]] && build_cmd+=(--kb-input "$KB_INPUT")
  [[ -n ${KB_SUPPLEMENT:-} ]] && build_cmd+=(--kb-supplement "$KB_SUPPLEMENT")
fi
"${build_cmd[@]}" >/dev/null

LITE_DATASET="$DATA_DIR/bundle_lite_compat/stage1_train.jsonl"

echo "=== [2/3] B300 Lite CPT Command (constant 1e-5, 2 epochs, MBS=4, GBS=32, DRY_RUN=$DRY_RUN) ==="
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
  --max_length "${SFT_MAX_LENGTH:-16384}"
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
  --max_length "${SFT_MAX_LENGTH:-16384}"
  --lr "${SFT_LR:-5e-6}"
  --output_dir "$ARM_B_OUT"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '%q ' "${export_cmd[@]}"; printf '\n'
  printf '%q ' "${arm_a_cmd[@]}"; printf '\n'
  printf '%q ' "${arm_b_cmd[@]}"; printf '\n'
  exit 0
fi

"${export_cmd[@]}"
"${arm_a_cmd[@]}"
"${arm_b_cmd[@]}"
