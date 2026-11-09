#!/usr/bin/env bash
set -euo pipefail

# Stage 4: MS-SWIFT Teacher DPO Launcher for Montessori-Instruct.
# Aligns the Teacher generator to prefer synthesizing high-positive-influence QA pairs.
: "${DATASET:?Set DATASET to the teacher_dpo_pairs.jsonl file}"
MODEL=${MODEL:-Qwen/Qwen3.5-27B}
OUTPUT_DIR=${OUTPUT_DIR:-./output/montessori-teacher-dpo}
LORA_SCOPE=${LORA_SCOPE:-all_linear}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NPROC_PER_NODE=${NPROC_PER_NODE:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

args=(rlhf --rlhf_type dpo --model "$MODEL" --dataset "$DATASET" --tuner_type lora
  --torch_dtype bfloat16 --num_train_epochs "${EPOCHS:-1}"
  --per_device_train_batch_size "${MICRO_BATCH_SIZE:-1}"
  --gradient_accumulation_steps "${GRAD_ACCUM:-8}"
  --learning_rate "${LR:-5e-6}" --beta "${DPO_BETA:-0.1}"
  --rpo_alpha "${RPO_ALPHA:-0.1}"
  --lora_rank "${LORA_RANK:-16}" --lora_alpha "${LORA_ALPHA:-32}"
  --lora_dropout "${LORA_DROPOUT:-0.05}"
  --max_length "${MAX_LENGTH:-4096}" --save_steps "${SAVE_STEPS:-50}"
  --logging_steps "${LOGGING_STEPS:-5}" --warmup_ratio "${WARMUP_RATIO:-0.05}"
  --freeze_vit "${FREEZE_VIT:-true}" --freeze_aligner "${FREEZE_ALIGNER:-true}"
  --output_dir "$OUTPUT_DIR")

case "$LORA_SCOPE" in
  all_linear) args+=(--target_modules all-linear) ;;
  modules)
    : "${TARGET_MODULES:?Set space-separated TARGET_MODULES}"
    read -r -a target_modules <<< "$TARGET_MODULES"
    args+=(--target_modules "${target_modules[@]}") ;;
  regex)
    : "${TARGET_REGEX:?Set TARGET_REGEX}"
    args+=(--target_regex "$TARGET_REGEX") ;;
  *) echo 'LORA_SCOPE must be all_linear, modules, or regex' >&2; exit 2 ;;
esac

args+=("$@")

if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q NPROC_PER_NODE=%q PYTORCH_CUDA_ALLOC_CONF=%q swift ' \
    "$CUDA_VISIBLE_DEVICES" "$NPROC_PER_NODE" "$PYTORCH_CUDA_ALLOC_CONF"
  printf '%q ' "${args[@]}"; printf '\n'; exit 0
fi

if ! command -v swift >/dev/null 2>&1; then
  echo 'Cannot find swift. Activate the MS-SWIFT environment first.' >&2; exit 1
fi
exec swift "${args[@]}"

