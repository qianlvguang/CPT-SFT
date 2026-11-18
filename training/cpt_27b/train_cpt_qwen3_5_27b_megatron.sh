#!/usr/bin/env bash
set -euo pipefail

# Thin B300 Megatron/SWIFT launcher for Qwen3.5-27B domain adaptation CPT.
#
# Launch Discipline (Runbook §4 & §6.3):
#   - Check `train_iters` in the startup log right after launch (lite 2ep ~ 36 iters).
#     If observed `train_iters` deviates >10% from expectation (e.g. tqdm 0/7 anomaly),
#     kill immediately, diagnose packing/step accounting, and re-register before relaunch.
#   - Defaults to `--gradient_accumulation_fusion false` (APEX fusion guardrail).
#   - Supports `DRY_RUN=1` (or omits `--launch`) for zero-GPU command verification.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$(cd "$SCRIPT_DIR/../lib" && pwd)"

# shellcheck source=../lib/_core.sh
source "$LIB_DIR/_core.sh"
# shellcheck source=../lib/hw_b300.sh
source "$LIB_DIR/hw_b300.sh"

: "${DATASET:?Set DATASET to a local JSONL/Parquet file or space-separated dataset list}"
MODEL=${MODEL:-${MODEL_PATH:-Qwen/Qwen3.5-27B}}
if [[ "$MODEL" == *Flash-Next* || "$MODEL" == *qwen4_exp* ]]; then
  die 2 "Use training/swift/train_b300_swift.sh for Qwen3.8-Flash-Next; this launcher targets 27B dense CPT."
fi

BACKEND=${BACKEND:-megatron}
if [[ -n ${PHASE:-} ]]; then
  case "$PHASE" in
    lite) STAGE=lite ;;
    stage1|stage1_stable) STAGE=stage1_stable ;;
    stage2|stage2_cooldown) STAGE=stage2_cooldown ;;
    pro|pro_incremental) STAGE=pro_incremental ;;
    *) die 2 "Unsupported PHASE: $PHASE (expected lite, stage1, stage2, or pro)" ;;
  esac
else
  STAGE=${STAGE:-stage1_stable}
fi

TUNER_TYPE=${TUNER_TYPE:-full}
LORA_SCOPE=${LORA_SCOPE:-all_linear}
SMOKE=${SMOKE:-0}
DRY_RUN=${DRY_RUN:-1}
LAUNCH=${LAUNCH:-0}

if [[ "$BACKEND" != megatron && "$BACKEND" != swift ]]; then
  die 2 "Unsupported BACKEND: $BACKEND (expected megatron or swift)"
fi
if [[ "$STAGE" != lite && "$STAGE" != stage1_stable && "$STAGE" != stage2_cooldown && "$STAGE" != pro_incremental ]]; then
  die 2 "Unsupported STAGE: $STAGE"
fi
if [[ "$TUNER_TYPE" != lora && "$TUNER_TYPE" != full ]]; then
  die 2 "Unsupported TUNER_TYPE: $TUNER_TYPE (expected lora or full)"
fi

BASE_OUTPUT_DIR=${OUTPUT_DIR:-"./output/qwen35-27b-cpt/${STAGE}"}
if [[ "$SMOKE" == "1" && -z ${OUTPUT_DIR:-} ]]; then
  OUTPUT_DIR="${BASE_OUTPUT_DIR}_smoke"
else
  OUTPUT_DIR="${BASE_OUTPUT_DIR}"
fi

export_b300_runtime_env
verify_te_sm103_preflight

FREEZE_LLM=${FREEZE_LLM:-false}
FREEZE_VIT=${FREEZE_VIT:-true}
FREEZE_ALIGNER=${FREEZE_ALIGNER:-false}
MAX_LENGTH=${MAX_LENGTH:-$DEF_MAX_LENGTH}
PACKING=${PACKING:-true}
PADDING_FREE=${PADDING_FREE:-true}
ENABLE_CHANNEL_LOSS=${ENABLE_CHANNEL_LOSS:-true}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
CLIP_GRAD=${CLIP_GRAD:-1.0}

if [[ "$STAGE" == "pro_incremental" ]]; then
  DEFAULT_LR="5e-6"
  DEFAULT_MIN_LR="1e-6"
elif [[ "$TUNER_TYPE" == "full" ]]; then
  DEFAULT_LR="1e-5"
  DEFAULT_MIN_LR="1e-6"
else
  DEFAULT_LR="2e-5"
  DEFAULT_MIN_LR="2e-6"
fi
if [[ "$STAGE" == "stage2_cooldown" && -z ${LR:-} && "$TUNER_TYPE" == "lora" ]]; then
  DEFAULT_LR="1e-5"
  DEFAULT_MIN_LR="1e-6"
fi

LR=${LR:-$DEFAULT_LR}
MIN_LR=${MIN_LR:-$DEFAULT_MIN_LR}
ALIGNER_LR=${ALIGNER_LR:-2e-6}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}

read -r -a dataset_list <<< "$DATASET"

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
      elif [[ "$BACKEND" == "swift" ]]; then
        args+=(--target_regex "$DEFAULT_HF_DEEP_MLP_REGEX")
      else
        die 2 "For BACKEND=megatron and LORA_SCOPE=deep_mlp_regex, inspect rank named_modules() with tools/check_target_regex.py and set TARGET_REGEX explicitly."
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
      die 2 "LORA_SCOPE must be all_linear, deep_mlp_regex, modules, or regex"
      ;;
  esac
}

if [[ "$BACKEND" == "megatron" ]]; then
  TP_SIZE=${TP_SIZE:-$DEF_TP_SIZE}
  validate_b300_27b_tp_gqa "$TP_SIZE"
  PP_SIZE=${PP_SIZE:-$DEF_PP_SIZE}
  CP_SIZE=${CP_SIZE:-$DEF_CP_SIZE}
  MBS=${MBS:-${MICRO_BATCH_SIZE:-$DEF_MICRO_BATCH_SIZE}}
  GBS=${GBS:-${GLOBAL_BATCH_SIZE:-$DEF_GLOBAL_BATCH_SIZE}}
  FINETUNE=${FINETUNE:-true}
  NO_SAVE_OPTIM=${NO_SAVE_OPTIM:-false}
  NO_SAVE_RNG=${NO_SAVE_RNG:-false}
  ATTENTION_BACKEND=${ATTENTION_BACKEND:-$DEF_ATTENTION_BACKEND}
  GRADIENT_ACCUMULATION_FUSION=${GRADIENT_ACCUMULATION_FUSION:-$DEF_GRADIENT_ACCUMULATION_FUSION}
  MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-0}

  case "$STAGE" in
    lite|stage1_stable)
      DEFAULT_SCHED="constant"
      DEFAULT_EPOCHS=2
      [[ "$STAGE" == "stage1_stable" ]] && DEFAULT_EPOCHS=1
      DEFAULT_SAVE_STEPS=128
      DEFAULT_SAVE_LIMIT=3
      ;;
    stage2_cooldown)
      DEFAULT_SCHED="WSD"
      DEFAULT_EPOCHS=1
      DEFAULT_SAVE_STEPS=64
      DEFAULT_SAVE_LIMIT=5
      ;;
    pro_incremental)
      DEFAULT_SCHED="cosine"
      DEFAULT_EPOCHS=1
      DEFAULT_SAVE_STEPS=32
      DEFAULT_SAVE_LIMIT=3
      ;;
  esac

  if [[ "$SMOKE" == "1" ]]; then
    EFFECTIVE_ITERS=10
    SCHED_MODE=${LR_DECAY_STYLE:-cosine}
    SAVE_STEPS_VAL=${SAVE_STEPS:-1000000}
    SAVE_LIMIT_VAL=${SAVE_TOTAL_LIMIT:-$DEFAULT_SAVE_LIMIT}
  elif [[ -n ${TRAIN_ITERS:-} ]]; then
    EFFECTIVE_ITERS=$TRAIN_ITERS
    SCHED_MODE=${LR_DECAY_STYLE:-$DEFAULT_SCHED}
    SAVE_STEPS_VAL=${SAVE_STEPS:-$DEFAULT_SAVE_STEPS}
    SAVE_LIMIT_VAL=${SAVE_TOTAL_LIMIT:-$DEFAULT_SAVE_LIMIT}
  else
    EFFECTIVE_ITERS=""
    SCHED_MODE=${LR_DECAY_STYLE:-$DEFAULT_SCHED}
    SAVE_STEPS_VAL=${SAVE_STEPS:-$DEFAULT_SAVE_STEPS}
    SAVE_LIMIT_VAL=${SAVE_TOTAL_LIMIT:-$DEFAULT_SAVE_LIMIT}
  fi

  WSD_DECAY_EXPLICIT=${STAGE2_DECAY_ITERS:-${LR_WSD_DECAY_ITERS:-}}
  if [[ "$SCHED_MODE" == "WSD" && -z "$EFFECTIVE_ITERS" && -z "$WSD_DECAY_EXPLICIT" ]]; then
    die 2 "LR_DECAY_STYLE=WSD requires TRAIN_ITERS or STAGE2_DECAY_ITERS/LR_WSD_DECAY_ITERS so lr_wsd_decay_iters is deterministic."
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
    --attention_backend "$ATTENTION_BACKEND"
    --gradient_accumulation_fusion "$GRADIENT_ACCUMULATION_FUSION"
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
    --mtp_num_layers "$MTP_NUM_LAYERS"
    --finetune "$FINETUNE"
    --no_save_optim "$NO_SAVE_OPTIM"
    --no_save_rng "$NO_SAVE_RNG"
    --save_safetensors true
    --save_strategy steps
    --save_steps "$SAVE_STEPS_VAL"
    --save_total_limit "$SAVE_LIMIT_VAL"
    --logging_steps "${LOGGING_STEPS:-5}"
    --output_dir "$OUTPUT_DIR"
  )

  if [[ "$FREEZE_ALIGNER" == "false" && -n ${ALIGNER_LR:-} ]]; then
    args+=(--aligner_lr "$ALIGNER_LR")
  fi
  if [[ -n ${VAL_DATASET:-} ]]; then
    read -r -a val_list <<< "$VAL_DATASET"
    args+=(--val_dataset "${val_list[@]}" --split_dataset_ratio 0)
  fi

  if [[ -n "$EFFECTIVE_ITERS" ]]; then
    args+=(--train_iters "$EFFECTIVE_ITERS")
  else
    args+=(--num_train_epochs "${EPOCHS:-$DEFAULT_EPOCHS}")
  fi

  args+=(--lr_decay_style "$SCHED_MODE")
  if [[ "$SCHED_MODE" == "WSD" ]]; then
    if [[ -n "$WSD_DECAY_EXPLICIT" ]]; then
      WSD_DECAY_VAL=$WSD_DECAY_EXPLICIT
    else
      WSD_DECAY_VAL=$(( EFFECTIVE_ITERS * 15 / 100 ))
      if (( WSD_DECAY_VAL < 1 )); then WSD_DECAY_VAL=1; fi
    fi

    if [[ -n ${LR_DECAY_ITERS:-} ]]; then
      args+=(--lr_decay_iters "$LR_DECAY_ITERS")
    elif [[ -n "$EFFECTIVE_ITERS" ]]; then
      args+=(--lr_decay_iters "$EFFECTIVE_ITERS")
    fi

    if [[ -n ${LR_WARMUP_ITERS:-} ]]; then
      args+=(--lr_warmup_iters "$LR_WARMUP_ITERS")
    elif [[ "$STAGE" == "stage2_cooldown" ]]; then
      if [[ "$NO_SAVE_OPTIM" == "true" ]]; then
        args+=(--lr_warmup_fraction "${LR_WARMUP_FRACTION:-0.05}")
      else
        args+=(--lr_warmup_fraction "${LR_WARMUP_FRACTION:-0}")
      fi
    elif [[ -n "$EFFECTIVE_ITERS" ]]; then
      WARMUP_VAL=$(( EFFECTIVE_ITERS * 3 / 100 ))
      if (( WARMUP_VAL < 1 )); then WARMUP_VAL=1; fi
      args+=(--lr_warmup_iters "$WARMUP_VAL")
    else
      args+=(--lr_warmup_fraction "${LR_WARMUP_FRACTION:-0.03}")
    fi

    args+=(
      --lr_wsd_decay_iters "$WSD_DECAY_VAL"
      --lr_wsd_decay_style "${LR_WSD_DECAY_STYLE:-linear}"
    )
  else
    DEFAULT_WARMUP_FRAC=0.03
    if [[ "$STAGE" == "stage1_stable" || "$STAGE" == "lite" ]]; then
      DEFAULT_WARMUP_FRAC=0.06
    elif [[ "$STAGE" == "pro_incremental" ]]; then
      DEFAULT_WARMUP_FRAC=0.05
    fi
    args+=(--lr_warmup_fraction "${LR_WARMUP_FRACTION:-$DEFAULT_WARMUP_FRAC}")
  fi

  if [[ "$TUNER_TYPE" == "lora" ]]; then
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
  if [[ $# -gt 0 ]]; then args+=("$@"); fi

  if [[ "$DRY_RUN" == "1" ]] || [[ "$LAUNCH" != "1" && "$DRY_RUN" != "0" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q NPROC_PER_NODE=%q CUDA_DEVICE_MAX_CONNECTIONS=%q IMAGE_MAX_TOKEN_NUM=%q PYTORCH_CUDA_ALLOC_CONF=%q megatron ' \
      "$CUDA_VISIBLE_DEVICES" "$NPROC_PER_NODE" "$CUDA_DEVICE_MAX_CONNECTIONS" "$IMAGE_MAX_TOKEN_NUM" "$PYTORCH_CUDA_ALLOC_CONF"
    printf '%q ' "${args[@]}"
    printf '\n'
    exit 0
  fi
  print_or_exec_cmd megatron "${args[@]}"
else
  MBS=${MBS:-${MICRO_BATCH_SIZE:-$DEF_MICRO_BATCH_SIZE}}
  GRAD_ACCUM=${GRAD_ACCUM:-4}
  SAVE_STEPS_VAL=${SAVE_STEPS:-100}
  if [[ "$SMOKE" == "1" ]]; then
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
  if [[ "$FREEZE_ALIGNER" == "false" && -n ${ALIGNER_LR:-} ]]; then
    args+=(--aligner_lr "$ALIGNER_LR")
  fi
  if [[ -n ${VAL_DATASET:-} ]]; then
    read -r -a val_list <<< "$VAL_DATASET"
    args+=(--val_dataset "${val_list[@]}" --split_dataset_ratio 0)
  fi
  if [[ "$SMOKE" == "1" ]]; then
    args+=(--max_steps 10)
  elif [[ -n ${MAX_STEPS:-} ]]; then
    args+=(--max_steps "$MAX_STEPS")
  else
    args+=(--num_train_epochs "${EPOCHS:-1}")
  fi

  if [[ "$TUNER_TYPE" == "lora" ]]; then
    append_lora_args
    if [[ -n ${ADAPTERS:-} ]]; then
      args+=(--adapters "$ADAPTERS")
    fi
  fi
  if [[ -n ${MODULES_TO_SAVE:-} ]]; then
    read -r -a modules_to_save <<< "$MODULES_TO_SAVE"
    args+=(--modules_to_save "${modules_to_save[@]}")
  fi
  if [[ $# -gt 0 ]]; then args+=("$@"); fi

  if [[ "$DRY_RUN" == "1" ]] || [[ "$LAUNCH" != "1" && "$DRY_RUN" != "0" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q NPROC_PER_NODE=%q IMAGE_MAX_TOKEN_NUM=%q PYTORCH_CUDA_ALLOC_CONF=%q swift ' \
      "$CUDA_VISIBLE_DEVICES" "$NPROC_PER_NODE" "$IMAGE_MAX_TOKEN_NUM" "$PYTORCH_CUDA_ALLOC_CONF"
    printf '%q ' "${args[@]}"
    printf '\n'
    exit 0
  fi
  print_or_exec_cmd swift "${args[@]}"
fi
