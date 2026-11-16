#!/usr/bin/env bash
set -euo pipefail

# Unified B300 Pipeline Orchestrator:
#   V1-V4 legacy pools + Wiki/Logo/Qwen-Image/RAG + Montessori/New QA
#     -> [Step 1] refactor_v1_v4_pipeline.py (CPT replay + token-balanced SFT + edit seeds)
#     -> [Step 2] build_mm_wiki_cpt.py      (6-channel Stage1/Stage2 WSD CPT shards)
#     -> [Step 3] train_qwen35_27b_cpt_b300.sh (Stage 1 Stable + Stage 2 Cooldown CPT)
#     -> [Step 4] CPT merge/export to standalone HF BF16 base directory
#     -> [Step 5] Unified QA + V1-V4 SFT (monitoring V1..V4 + wiki_rag_qa channel losses)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOMAIN_DIR="$REPO_ROOT/training/domain_adaptation"

WORK_DIR=${WORK_DIR:-"${TMPDIR:-/tmp}/cpt_sft_unified_b300"}
MODEL=${MODEL:-Qwen/Qwen3.5-27B}
BACKEND=${BACKEND:-megatron}          # megatron | swift
CPT_TUNER_TYPE=${CPT_TUNER_TYPE:-lora} # lora | full
SMOKE=${SMOKE:-0}
DRY_RUN=${DRY_RUN:-1}
DEMO=${DEMO:-0}

if [[ $MODEL == *Flash-Next* || $MODEL == *qwen4_exp* ]]; then
  echo 'Use train_b300_swift.sh for Qwen3.8-Flash-Next; this unified pipeline targets 27B.' >&2
  exit 2
fi
if [[ $BACKEND != megatron && $BACKEND != swift ]]; then
  echo "Unsupported BACKEND: $BACKEND (expected megatron or swift)" >&2
  exit 2
fi
if [[ $CPT_TUNER_TYPE != lora && $CPT_TUNER_TYPE != full ]]; then
  echo "Unsupported CPT_TUNER_TYPE: $CPT_TUNER_TYPE (expected lora or full)" >&2
  exit 2
fi

if [[ $DEMO == 1 || ( -z ${V1_V4_INPUTS:-} && $DRY_RUN == 1 ) ]]; then
  V1_V4_INPUTS=${V1_V4_INPUTS:-"$DOMAIN_DIR/examples/v1_v4_legacy_sample.jsonl"}
  NEW_QA_INPUTS=${NEW_QA_INPUTS:-"$DOMAIN_DIR/examples/style_sft_example.jsonl"}
  WIKI_CPT_INPUTS=${WIKI_CPT_INPUTS:-"$DOMAIN_DIR/examples/cpt_mm_6ch_example.jsonl"}
  DEMO_MODE=1
else
  : "${V1_V4_INPUTS:?Set space-separated V1_V4_INPUTS (or pass DEMO=1)}"
  : "${WIKI_CPT_INPUTS:?Set space-separated WIKI_CPT_INPUTS (or pass DEMO=1)}"
  NEW_QA_INPUTS=${NEW_QA_INPUTS:-""}
  DEMO_MODE=0
fi

read -r -a v1_v4_arr <<< "$V1_V4_INPUTS"
read -r -a wiki_arr <<< "$WIKI_CPT_INPUTS"

REFACTOR_DIR="$WORK_DIR/step1_v1_v4_refactored"
CPT_DATA_DIR="$WORK_DIR/step2_cpt_bundle"
CPT_OUT_DIR="$WORK_DIR/step3_cpt_ckpt"
CPT_EXPORT_DIR="$WORK_DIR/step4_cpt_hf_merged"
SFT_OUT_DIR="$WORK_DIR/step5_unified_sft_ckpt"

mkdir -p "$WORK_DIR"

echo "=== [1/5] Refactoring V1-V4 into CPT Replay + Token-Balanced Unified SFT ==="
refactor_cmd=(
  python3 "$DOMAIN_DIR/refactor_v1_v4_pipeline.py"
  --v1-v4-inputs "${v1_v4_arr[@]}"
  --output-dir "$REFACTOR_DIR"
)
if [[ -n ${NEW_QA_INPUTS:-} ]]; then
  read -r -a new_qa_arr <<< "$NEW_QA_INPUTS"
  refactor_cmd+=(--new-qa-inputs "${new_qa_arr[@]}")
fi
if [[ -n ${RAG_EVIDENCE_MAP:-} ]]; then
  refactor_cmd+=(--rag-evidence-map "$RAG_EVIDENCE_MAP")
fi
if [[ $DEMO_MODE == 1 ]]; then
  refactor_cmd+=(--val-ratio 0.25 --sft-target-rows 20)
elif [[ -n ${SFT_TARGET_ROWS:-} ]]; then
  refactor_cmd+=(--sft-target-rows "$SFT_TARGET_ROWS")
fi
"${refactor_cmd[@]}" >/dev/null

echo "=== [2/5] Building 6-Channel Multimodal CPT Bundle (Zero-Leakage + WSD Stage 2) ==="
cpt_build_cmd=(
  python3 "$DOMAIN_DIR/build_mm_wiki_cpt.py"
  --inputs "${wiki_arr[@]}" "$REFACTOR_DIR/cpt_v1_v4_replay.jsonl"
  --output-dir "$CPT_DATA_DIR"
)
if [[ $DEMO_MODE == 1 ]]; then
  cpt_build_cmd+=(--val-ratio 0.20 --stage1-rows 24 --stage2-rows 16)
else
  if [[ -n ${CPT_STAGE1_ROWS:-} ]]; then cpt_build_cmd+=(--stage1-rows "$CPT_STAGE1_ROWS"); fi
  if [[ -n ${CPT_STAGE2_ROWS:-} ]]; then cpt_build_cmd+=(--stage2-rows "$CPT_STAGE2_ROWS"); fi
fi
"${cpt_build_cmd[@]}" >/dev/null

echo "=== [3/5] Running Two-Stage WSD Multimodal CPT (BACKEND=$BACKEND, TUNER=$CPT_TUNER_TYPE, DRY_RUN=$DRY_RUN) ==="
STAGE1_ITERS=${STAGE1_ITERS:-850}
STAGE2_ITERS=${STAGE2_ITERS:-150}

MODEL="$MODEL" BACKEND="$BACKEND" STAGE=stage1_stable TUNER_TYPE="$CPT_TUNER_TYPE" \
  DATASET="$CPT_DATA_DIR/stage1_train.jsonl" VAL_DATASET="$CPT_DATA_DIR/val.jsonl" \
  OUTPUT_DIR="$CPT_OUT_DIR/stage1_stable" TRAIN_ITERS="$STAGE1_ITERS" \
  LR_DECAY_STYLE="${CPT_SCHED:-WSD}" SMOKE="$SMOKE" DRY_RUN="$DRY_RUN" \
  bash "$SCRIPT_DIR/train_qwen35_27b_cpt_b300.sh"

if [[ $CPT_TUNER_TYPE == lora ]]; then
  ADAPTERS="${STAGE1_ADAPTER:-$CPT_OUT_DIR/stage1_stable/latest}" \
    MODEL="$MODEL" BACKEND="$BACKEND" STAGE=stage2_cooldown TUNER_TYPE="$CPT_TUNER_TYPE" \
    DATASET="$CPT_DATA_DIR/stage2_cooldown.jsonl" VAL_DATASET="$CPT_DATA_DIR/val.jsonl" \
    OUTPUT_DIR="$CPT_OUT_DIR/stage2_cooldown" TRAIN_ITERS="$STAGE2_ITERS" \
    LR_DECAY_STYLE="${CPT_SCHED:-WSD}" SMOKE="$SMOKE" DRY_RUN="$DRY_RUN" \
    bash "$SCRIPT_DIR/train_qwen35_27b_cpt_b300.sh"
else
  MODEL="$MODEL" BACKEND="$BACKEND" STAGE=stage2_cooldown TUNER_TYPE="$CPT_TUNER_TYPE" \
    DATASET="$CPT_DATA_DIR/stage2_cooldown.jsonl" VAL_DATASET="$CPT_DATA_DIR/val.jsonl" \
    OUTPUT_DIR="$CPT_OUT_DIR/stage2_cooldown" TRAIN_ITERS="$STAGE2_ITERS" \
    LR_DECAY_STYLE="${CPT_SCHED:-WSD}" SMOKE="$SMOKE" DRY_RUN="$DRY_RUN" \
    bash "$SCRIPT_DIR/train_qwen35_27b_cpt_b300.sh"
fi

echo "=== [4/5] Exporting Merged HF BF16 Base for Unified SFT ==="
if [[ $CPT_TUNER_TYPE == lora ]]; then
  export_cmd=(
    swift export
    --model "$MODEL"
    --adapters "$CPT_OUT_DIR/stage2_cooldown/latest"
    --merge_lora true
    --output_dir "$CPT_EXPORT_DIR"
  )
else
  export_cmd=(
    swift export
    --mcore_model "$CPT_OUT_DIR/stage2_cooldown/latest"
    --to_hf true
    --torch_dtype bfloat16
    --output_dir "$CPT_EXPORT_DIR"
  )
fi
if [[ $DRY_RUN == 1 ]]; then
  printf '%q ' "${export_cmd[@]}"
  printf '\n'
else
  "${export_cmd[@]}"
fi

echo "=== [5/5] Launching Unified V1-V4 + New QA/RAG SFT (DRY_RUN=$DRY_RUN) ==="
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-16384}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [[ $BACKEND == megatron ]]; then
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
    --aligner_lr "${SFT_ALIGNER_LR:-5e-6}"
    --tensor_model_parallel_size "${TP_SIZE:-2}"
    --pipeline_model_parallel_size "${PP_SIZE:-1}"
    --sequence_parallel true
    --recompute_granularity selective
    --attention_backend flash
    --packing true
    --padding_free true
    --max_length "${SFT_MAX_LENGTH:-16384}"
    --truncation_strategy delete
    --strict true
    --enable_channel_loss true
    --micro_batch_size "${SFT_MBS:-1}"
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
  if [[ $SMOKE == 1 ]]; then
    sft_cmd+=(--train_iters 10 --save_steps 1000000)
  else
    sft_cmd+=(--num_train_epochs "${SFT_EPOCHS:-1}" --save_steps "${SFT_SAVE_STEPS:-200}")
  fi
else
  sft_cmd=(
    swift sft
    --model "$CPT_EXPORT_DIR"
    --dataset "$REFACTOR_DIR/sft_unified_train.jsonl"
    --val_dataset "$REFACTOR_DIR/sft_unified_val.jsonl"
    --split_dataset_ratio 0
    --tuner_type lora
    --target_modules all-linear
    --lora_rank "${SFT_LORA_RANK:-64}"
    --lora_alpha "${SFT_LORA_ALPHA:-128}"
    --lora_dropout "${SFT_LORA_DROPOUT:-0.05}"
    --torch_dtype bfloat16
    --freeze_vit "${SFT_FREEZE_VIT:-true}"
    --freeze_aligner "${SFT_FREEZE_ALIGNER:-false}"
    --aligner_lr "${SFT_ALIGNER_LR:-5e-6}"
    --attn_impl flash_attn
    --packing true
    --padding_free true
    --max_length "${SFT_MAX_LENGTH:-16384}"
    --truncation_strategy delete
    --strict true
    --enable_channel_loss true
    --per_device_train_batch_size "${SFT_MBS:-1}"
    --gradient_accumulation_steps "${SFT_GRAD_ACCUM:-8}"
    --learning_rate "${SFT_LR:-1e-5}"
    --lr_scheduler_type cosine
    --warmup_ratio 0.03
    --weight_decay 0.01
    --max_grad_norm 1.0
    --gradient_checkpointing true
    --output_dir "$SFT_OUT_DIR"
  )
  if [[ $SMOKE == 1 ]]; then
    sft_cmd+=(--max_steps 10 --save_steps 1000000)
  else
    sft_cmd+=(--num_train_epochs "${SFT_EPOCHS:-1}" --save_steps "${SFT_SAVE_STEPS:-200}")
  fi
fi

if [[ $DRY_RUN == 1 ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q NPROC_PER_NODE=%q IMAGE_MAX_TOKEN_NUM=%q PYTORCH_CUDA_ALLOC_CONF=%q ' \
    "$CUDA_VISIBLE_DEVICES" "$NPROC_PER_NODE" "$IMAGE_MAX_TOKEN_NUM" "$PYTORCH_CUDA_ALLOC_CONF"
  printf '%q ' "${sft_cmd[@]}"
  printf '\n'
  exit 0
fi

exec "${sft_cmd[@]}"
