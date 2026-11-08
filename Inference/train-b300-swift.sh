#!/usr/bin/env bash
set -euo pipefail

# Single-node, 8-GPU Megatron LoRA SFT, based on the upstream qwen4_exp example.
# Run in the existing MS-SWIFT 4.7.0.dev environment, not the vLLM image.
: "${DATASET:?Set DATASET to a local JSONL dataset or a supported dataset ID}"
MODEL=${MODEL:-Qwen/Qwen3.8-Flash-Next}
if [[ $MODEL == *Qwen3.5-27B* ]]; then
  echo 'Use train-qwen35-27b-swift.sh for Qwen3.5-27B; this script is for Flash-Next.' >&2; exit 2
fi
OUTPUT_DIR=${OUTPUT_DIR:-./output/qwen38-flash-next-lora}
LORA_SCOPE=${LORA_SCOPE:-baseline}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NPROC_PER_NODE=8
export PLE_CPU_OFFLOAD=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

args=(sft --model "$MODEL" --dataset "$DATASET"
  --num_train_epochs "${EPOCHS:-1}"
  --tuner_type lora --lora_rank "${LORA_RANK:-8}" --lora_alpha "${LORA_ALPHA:-32}"
  --tensor_model_parallel_size 2 --expert_model_parallel_size 4
  --expert_tensor_parallel_size 1 --pipeline_model_parallel_size 2
  --decoder_first_pipeline_num_layers 12 --sequence_parallel true
  --moe_permute_fusion true --recompute_granularity full
  --recompute_method uniform --recompute_num_layers 1 --padding_free true
  --max_length "${MAX_LENGTH:-8192}" --micro_batch_size 1
  --global_batch_size "${GLOBAL_BATCH_SIZE:-8}" --lr "${LR:-1e-4}"
  --save_steps "${SAVE_STEPS:-500}" --output_dir "$OUTPUT_DIR"
  --lora_dropout "${LORA_DROPOUT:-0.05}" --lr_decay_style "${LR_DECAY_STYLE:-cosine}"
  --min_lr "${MIN_LR:-0}" --weight_decay "${WEIGHT_DECAY:-0.1}"
  --clip_grad "${CLIP_GRAD:-1.0}")

# target_regex replaces target_modules upstream, including the all-router keyword.
case "$LORA_SCOPE" in
  baseline) args+=(--target_modules in_proj out_proj linear_proj linear_qkv) ;;
  baseline_router) args+=(--target_modules in_proj out_proj linear_proj linear_qkv all-router) ;;
  router) args+=(--target_modules all-router) ;;
  modules)
    : "${TARGET_MODULES:?Set space-separated TARGET_MODULES for LORA_SCOPE=modules}"
    read -r -a target_modules <<< "$TARGET_MODULES"
    args+=(--target_modules "${target_modules[@]}") ;;
  regex)
    : "${TARGET_REGEX:?Set TARGET_REGEX for LORA_SCOPE=regex}"
    args+=(--target_regex "$TARGET_REGEX") ;;
  *) echo 'LORA_SCOPE must be baseline, baseline_router, router, modules, or regex' >&2; exit 2 ;;
esac
if [[ -n ${MODULES_TO_SAVE:-} ]]; then
  read -r -a modules_to_save <<< "$MODULES_TO_SAVE"
  args+=(--modules_to_save "${modules_to_save[@]}")
fi
if [[ -n ${LR_WARMUP_FRACTION:-} ]]; then args+=(--lr_warmup_fraction "$LR_WARMUP_FRACTION"); fi
if [[ -n ${MOE_AUX_LOSS_COEFF:-} ]]; then args+=(--moe_aux_loss_coeff "$MOE_AUX_LOSS_COEFF"); fi
if [[ -n ${MOE_Z_LOSS_COEFF:-} ]]; then args+=(--moe_z_loss_coeff "$MOE_Z_LOSS_COEFF"); fi
args+=("$@")
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf 'PLE_CPU_OFFLOAD=%q CUDA_VISIBLE_DEVICES=%q NPROC_PER_NODE=%q PYTORCH_CUDA_ALLOC_CONF=%q megatron ' \
    "$PLE_CPU_OFFLOAD" "$CUDA_VISIBLE_DEVICES" "$NPROC_PER_NODE" "$PYTORCH_CUDA_ALLOC_CONF"
  printf '%q ' "${args[@]}"; printf '\n'; exit 0
fi
if ! command -v megatron >/dev/null 2>&1; then
  echo 'Cannot find megatron. Activate the MS-SWIFT Megatron environment first.' >&2; exit 1
fi
exec megatron "${args[@]}"
