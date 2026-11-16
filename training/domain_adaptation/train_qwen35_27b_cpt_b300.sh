#!/usr/bin/env bash
set -euo pipefail

# Single-node, 8x B300 (288GB HBM3e) multimodal CPT starter for Qwen3.5-27B
# (and structurally isomorphic Qwen3.6/3.8-27B dense VLM backbones).
#
# Supports:
#   BACKEND=megatron (default: megatron pt, WSD / cosine, TP2/PP1/DP4)
#   BACKEND=swift    (swift pt, HF + DeepSpeed / single-or-multi-GPU LoRA probes)
#   STAGE=stage1_stable | stage2_cooldown
#   TUNER_TYPE=lora (default) | full
#   LORA_SCOPE=all_linear | deep_mlp_regex | modules | regex
#   SMOKE=1          (10-step sanity check without epochs override)
#   DRY_RUN=1        (prints full command without GPU or binary requirement)

: "${DATASET:?Set DATASET to a local JSONL/Parquet file or space-separated dataset list}"
MODEL=${MODEL:-Qwen/Qwen3.5-27B}
if [[ $MODEL == *Flash-Next* || $MODEL == *qwen4_exp* ]]; then
  echo 'Use training/swift/train_b300_swift.sh for Qwen3.8-Flash-Next; this script is for 27B dense CPT.' >&2
  exit 2
fi

BACKEND=${BACKEND:-megatron}
STAGE=${STAGE:-stage1_stable}
TUNER_TYPE=${TUNER_TYPE:-lora}
LORA_SCOPE=${LORA_SCOPE:-all_linear}
SMOKE=${SMOKE:-0}

if [[ $BACKEND != megatron && $BACKEND != swift ]]; then
  echo "Unsupported BACKEND: $BACKEND (expected megatron or swift)" >&2
  exit 2
fi
if [[ $STAGE != stage1_stable && $STAGE != stage2_cooldown ]]; then
  echo "Unsupported STAGE: $STAGE (expected stage1_stable or stage2_cooldown)" >&2
  exit 2
fi
if [[ $TUNER_TYPE != lora && $TUNER_TYPE != full ]]; then
  echo "Unsupported TUNER_TYPE: $TUNER_TYPE (expected lora or full)" >&2
  exit 2
fi

BASE_OUTPUT_DIR=${OUTPUT_DIR:-"./output/qwen35-27b-mm-cpt/${STAGE}"}
if [[ $SMOKE == 1 && -z ${OUTPUT_DIR:-} ]]; then
  OUTPUT_DIR="${BASE_OUTPUT_DIR}_smoke"
else
  OUTPUT_DIR="${BASE_OUTPUT_DIR}"
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-16384}
export VIDEO_MAX_TOKEN_NUM=${VIDEO_MAX_TOKEN_NUM:-128}
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-16}

FREEZE_LLM=${FREEZE_LLM:-false}
FREEZE_VIT=${FREEZE_VIT:-true}
FREEZE_ALIGNER=${FREEZE_ALIGNER:-false}
MAX_LENGTH=${MAX_LENGTH:-16384}
PACKING=${PACKING:-true}
PADDING_FREE=${PADDING_FREE:-true}
ENABLE_CHANNEL_LOSS=${ENABLE_CHANNEL_LOSS:-true}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
CLIP_GRAD=${CLIP_GRAD:-1.0}

if [[ $TUNER_TYPE == full ]]; then
  DEFAULT_LR="1e-5"
  DEFAULT_MIN_LR="1e-6"
else
  DEFAULT_LR="2e-5"
  DEFAULT_MIN_LR="2e-6"
fi
if [[ $STAGE == stage2_cooldown && -z ${LR:-} && $TUNER_TYPE == lora ]]; then
  DEFAULT_LR="1e-5"
  DEFAULT_MIN_LR="1e-6"
fi

LR=${LR:-$DEFAULT_LR}
MIN_LR=${MIN_LR:-$DEFAULT_MIN_LR}
ALIGNER_LR=${ALIGNER_LR:-5e-6}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}

read -r -a dataset_list <<< "$DATASET"

# HF late-layer MLP (layers 22..63 out of 64) + visual.merger regex for deep factual injection.
DEFAULT_HF_DEEP_MLP_REGEX='^(?:.*model\.)?(language_model\.layers\.(2[2-9]|[3-5][0-9]|6[0-3])\.mlp\.(gate_proj|up_proj|down_proj)|visual\.merger\..*)$'

append_lora_args() {
  args+=(
    --lora_rank "$LORA_RANK"
    --lora_alpha "$LORA_ALPHA"
    --lora_dropout "$LORA_DROPOUT"
  )
  case "$LORA_SCOPE" in
    all_linear)
      args+=(--target_modules all-linear)
      ;;
    deep_mlp_regex)
      if [[ -n ${TARGET_REGEX:-} ]]; then
        args+=(--target_regex "$TARGET_REGEX")
      elif [[ $BACKEND == swift ]]; then
        args+=(--target_regex "$DEFAULT_HF_DEEP_MLP_REGEX")
      else
        echo 'For BACKEND=megatron and LORA_SCOPE=deep_mlp_regex, inspect rank named_modules() with tools/check_target_regex.py and set TARGET_REGEX explicitly.' >&2
        exit 2
      fi
      ;;
    modules)
      : "${TARGET_MODULES:?Set space-separated TARGET_MODULES for LORA_SCOPE=modules}"
      read -r -a target_modules <<< "$TARGET_MODULES"
      args+=(--target_modules "${target_modules[@]}")
      ;;
    regex)
      : "${TARGET_REGEX:?Set TARGET_REGEX for LORA_SCOPE=regex}"
      args+=(--target_regex "$TARGET_REGEX")
      ;;
    *)
      echo 'LORA_SCOPE must be all_linear, deep_mlp_regex, modules, or regex' >&2
      exit 2
      ;;
  esac
}

if [[ $BACKEND == megatron ]]; then
  TP_SIZE=${TP_SIZE:-2}
  PP_SIZE=${PP_SIZE:-1}
  CP_SIZE=${CP_SIZE:-1}
  MBS=${MBS:-1}
  GBS=${GBS:-64}
  FINETUNE=${FINETUNE:-true}

  if [[ $SMOKE == 1 ]]; then
    EFFECTIVE_ITERS=10
    SCHED_MODE=${LR_DECAY_STYLE:-cosine}
    SAVE_STEPS_VAL=${SAVE_STEPS:-1000000}
  elif [[ -n ${TRAIN_ITERS:-} ]]; then
    EFFECTIVE_ITERS=$TRAIN_ITERS
    SCHED_MODE=${LR_DECAY_STYLE:-WSD}
    SAVE_STEPS_VAL=${SAVE_STEPS:-100}
  else
    EFFECTIVE_ITERS=""
    SCHED_MODE=${LR_DECAY_STYLE:-cosine}
    SAVE_STEPS_VAL=${SAVE_STEPS:-100}
    if [[ $SCHED_MODE == WSD ]]; then
      echo 'LR_DECAY_STYLE=WSD requires explicit TRAIN_ITERS (e.g., TRAIN_ITERS=1000) so lr_decay_iters and lr_wsd_decay_iters are deterministic.' >&2
      exit 2
    fi
  fi

  args=(pt
    --model "$MODEL"
    --dataset "${dataset_list[@]}"
    --tuner_type "$TUNER_TYPE"
    --torch_dtype bfloat16
    --freeze_llm "$FREEZE_LLM"
    --freeze_vit "$FREEZE_VIT"
    --freeze_aligner "$FREEZE_ALIGNER"
    --tensor_model_parallel_size "$TP_SIZE"
    --pipeline_model_parallel_size "$PP_SIZE"
    --context_parallel_size "$CP_SIZE"
    --sequence_parallel "${SEQUENCE_PARALLEL:-true}"
    --use_distributed_optimizer true
    --recompute_granularity "${RECOMPUTE_GRANULARITY:-selective}"
    --attention_backend flash
    --packing "$PACKING"
    --padding_free "$PADDING_FREE"
    --max_length "$MAX_LENGTH"
    --truncation_strategy delete
    --strict true
    --micro_batch_size "$MBS"
    --global_batch_size "$GBS"
    --optimizer adam
    --lr "$LR"
    --min_lr "$MIN_LR"
    --weight_decay "$WEIGHT_DECAY"
    --clip_grad "$CLIP_GRAD"
    --enable_channel_loss "$ENABLE_CHANNEL_LOSS"
    --finetune "$FINETUNE"
    --no_save_optim "${NO_SAVE_OPTIM:-false}"
    --no_save_rng "${NO_SAVE_RNG:-false}"
    --save_safetensors true
    --save_strategy steps
    --save_steps "$SAVE_STEPS_VAL"
    --logging_steps "${LOGGING_STEPS:-5}"
    --output_dir "$OUTPUT_DIR"
  )

  if [[ $FREEZE_ALIGNER == false && -n ${ALIGNER_LR:-} ]]; then
    args+=(--aligner_lr "$ALIGNER_LR")
  fi
  if [[ -n ${VAL_DATASET:-} ]]; then
    read -r -a val_list <<< "$VAL_DATASET"
    args+=(--val_dataset "${val_list[@]}" --split_dataset_ratio 0)
  fi
  if [[ -n $EFFECTIVE_ITERS ]]; then
    args+=(--train_iters "$EFFECTIVE_ITERS" --lr_decay_style "$SCHED_MODE")
    if [[ $SCHED_MODE == WSD ]]; then
      LR_DECAY_ITERS=${LR_DECAY_ITERS:-$EFFECTIVE_ITERS}
      LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-$(( EFFECTIVE_ITERS * 3 / 100 ))}
      LR_WSD_DECAY_ITERS=${LR_WSD_DECAY_ITERS:-$(( EFFECTIVE_ITERS * 15 / 100 ))}
      if (( LR_WARMUP_ITERS < 1 )); then LR_WARMUP_ITERS=1; fi
      if (( LR_WSD_DECAY_ITERS < 1 )); then LR_WSD_DECAY_ITERS=1; fi
      args+=(
        --lr_decay_iters "$LR_DECAY_ITERS"
        --lr_warmup_iters "$LR_WARMUP_ITERS"
        --lr_wsd_decay_iters "$LR_WSD_DECAY_ITERS"
        --lr_wsd_decay_style "${LR_WSD_DECAY_STYLE:-linear}"
      )
    else
      args+=(--lr_warmup_fraction "${LR_WARMUP_FRACTION:-0.03}")
    fi
  else
    args+=(
      --num_train_epochs "${EPOCHS:-1}"
      --lr_decay_style "$SCHED_MODE"
      --lr_warmup_fraction "${LR_WARMUP_FRACTION:-0.03}"
    )
  fi

  if [[ $TUNER_TYPE == lora ]]; then
    append_lora_args
    args+=(--merge_lora false)
    if [[ -n ${ADAPTERS:-} ]]; then
      args+=(--adapters "$ADAPTERS")
    fi
  fi
  if [[ -n ${MODULES_TO_SAVE:-} ]]; then
    read -r -a modules_to_save <<< "$MODULES_TO_SAVE"
    args+=(--modules_to_save "${modules_to_save[@]}")
  fi
  args+=("$@")

  if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q NPROC_PER_NODE=%q CUDA_DEVICE_MAX_CONNECTIONS=%q IMAGE_MAX_TOKEN_NUM=%q PYTORCH_CUDA_ALLOC_CONF=%q megatron ' \
      "$CUDA_VISIBLE_DEVICES" "$NPROC_PER_NODE" "$CUDA_DEVICE_MAX_CONNECTIONS" "$IMAGE_MAX_TOKEN_NUM" "$PYTORCH_CUDA_ALLOC_CONF"
    printf '%q ' "${args[@]}"
    printf '\n'
    exit 0
  fi
  if ! command -v megatron >/dev/null 2>&1; then
    echo 'Cannot find megatron. Activate the MS-SWIFT Megatron environment first.' >&2
    exit 1
  fi
  exec megatron "${args[@]}"
else
  MBS=${MBS:-1}
  GRAD_ACCUM=${GRAD_ACCUM:-8}
  SAVE_STEPS_VAL=${SAVE_STEPS:-100}
  if [[ $SMOKE == 1 ]]; then
    SAVE_STEPS_VAL=${SAVE_STEPS:-1000000}
  fi

  args=(pt
    --model "$MODEL"
    --dataset "${dataset_list[@]}"
    --tuner_type "$TUNER_TYPE"
    --torch_dtype bfloat16
    --freeze_llm "$FREEZE_LLM"
    --freeze_vit "$FREEZE_VIT"
    --freeze_aligner "$FREEZE_ALIGNER"
    --attn_impl flash_attn
    --packing "$PACKING"
    --padding_free "$PADDING_FREE"
    --max_length "$MAX_LENGTH"
    --truncation_strategy delete
    --strict true
    --per_device_train_batch_size "$MBS"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --learning_rate "$LR"
    --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}"
    --warmup_ratio "${WARMUP_RATIO:-0.03}"
    --weight_decay "$WEIGHT_DECAY"
    --max_grad_norm "$CLIP_GRAD"
    --gradient_checkpointing true
    --enable_channel_loss "$ENABLE_CHANNEL_LOSS"
    --save_steps "$SAVE_STEPS_VAL"
    --logging_steps "${LOGGING_STEPS:-5}"
    --output_dir "$OUTPUT_DIR"
  )

  if [[ -n ${DEEPSPEED:-} ]]; then
    args+=(--deepspeed "$DEEPSPEED")
  fi
  if [[ $FREEZE_ALIGNER == false && -n ${ALIGNER_LR:-} ]]; then
    args+=(--aligner_lr "$ALIGNER_LR")
  fi
  if [[ -n ${VAL_DATASET:-} ]]; then
    read -r -a val_list <<< "$VAL_DATASET"
    args+=(--val_dataset "${val_list[@]}" --split_dataset_ratio 0)
  fi
  if [[ $SMOKE == 1 ]]; then
    args+=(--max_steps 10)
  elif [[ -n ${MAX_STEPS:-} ]]; then
    args+=(--max_steps "$MAX_STEPS")
  else
    args+=(--num_train_epochs "${EPOCHS:-1}")
  fi

  if [[ $TUNER_TYPE == lora ]]; then
    append_lora_args
    if [[ -n ${ADAPTERS:-} ]]; then
      args+=(--adapters "$ADAPTERS")
    fi
  fi
  if [[ -n ${MODULES_TO_SAVE:-} ]]; then
    read -r -a modules_to_save <<< "$MODULES_TO_SAVE"
    args+=(--modules_to_save "${modules_to_save[@]}")
  fi
  args+=("$@")

  if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q NPROC_PER_NODE=%q IMAGE_MAX_TOKEN_NUM=%q PYTORCH_CUDA_ALLOC_CONF=%q swift ' \
      "$CUDA_VISIBLE_DEVICES" "$NPROC_PER_NODE" "$IMAGE_MAX_TOKEN_NUM" "$PYTORCH_CUDA_ALLOC_CONF"
    printf '%q ' "${args[@]}"
    printf '\n'
    exit 0
  fi
  if ! command -v swift >/dev/null 2>&1; then
    echo 'Cannot find swift. Activate the MS-SWIFT environment first.' >&2
    exit 1
  fi
  exec swift "${args[@]}"
fi
