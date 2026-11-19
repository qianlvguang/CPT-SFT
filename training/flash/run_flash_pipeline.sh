#!/usr/bin/env bash
set -euo pipefail

# Phase B (Flash Streamlined 3-Step Backbone):
#   Step 1: Data Preparation
#             - QA->CoT v0.4 Knowledge Internalization + Clean Direct Twins + CPT-safe Prose (`gen_cot_from_qa.py`)
#             - ImageGen v2 Domain-Partitioned Routing + Multi-Image Doc Packing + V1 3-way Recycle (`build_v2_image_pipeline.py`)
#             - V1-V4 & SFT Arm Consolidation (`refactor_v1_v4_pipeline.py`)
#   Step 2: Single-Stage Instruct-Aware Mid-Training (`build_mm_wiki_cpt.py --mode midtrain` + `run_cpt_27b_b300.sh flash` + `swift export`)
#             - Enforces 30/35/15/20 mix, <=8192 seq len, <=25% synth images, <=15% QA prose, zero raw ChatML/<think> in pt, zero eval leak
#   Step 3: Single Unified Dual-Mode SFT (`megatron sft` Think-ON + Clean Direct Twins -> Think-ON/OFF dual eval)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CPT27B_DIR="$TRAINING_DIR/cpt_27b"

WORK_DIR=${WORK_DIR:-"${TMPDIR:-/tmp}/cpt_sft_flash_pipeline"}
MODEL=${MODEL:-/data/autovision-cfs/autovision-notebook/algo4/models/Qwen3.5-27B}
BACKEND=${BACKEND:-megatron}          # megatron | swift
CPT_TUNER_TYPE=${CPT_TUNER_TYPE:-full} # full | lora
SFT_ARM=${SFT_ARM:-ood_plus_qa_cot}   # unified | ood_only | ood_plus_qa_direct | ood_plus_qa_cot
COT_MODE=${COT_MODE:-mock}            # mock | vllm
COT_MODE_VER=${COT_MODE_VER:-internalized} # internalized (v0.4) | rag_citation (v0.3)
IMG_V2_MODE=${IMG_V2_MODE:-mock}      # mock | vllm_omni
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
  if [[ "$COT_MODE_VER" == "internalized" ]]; then
    QA_RAW_INPUT=${QA_RAW_INPUT:-"$SCRIPT_DIR/examples/qa_kb_v04_example.jsonl"}
  else
    QA_RAW_INPUT=${QA_RAW_INPUT:-"$SCRIPT_DIR/examples/qa_kb_raw_example.jsonl"}
  fi
  V2_ENTITIES_INPUT=${V2_ENTITIES_INPUT:-"$SCRIPT_DIR/examples/v2_image_entities_sample.jsonl"}
  V1_AUDIT_INPUT=${V1_AUDIT_INPUT:-"$SCRIPT_DIR/examples/v2_image_v1_audit_sample.jsonl"}
  DEMO_MODE=1
else
  : "${V1_V4_INPUTS:?Set space-separated V1_V4_INPUTS (or pass DEMO=1)}"
  : "${WIKI_CPT_INPUTS:?Set space-separated WIKI_CPT_INPUTS (or pass DEMO=1)}"
  QA_RAW_INPUT=${QA_RAW_INPUT:-""}
  V2_ENTITIES_INPUT=${V2_ENTITIES_INPUT:-""}
  V1_AUDIT_INPUT=${V1_AUDIT_INPUT:-""}
  DEMO_MODE=0
fi

read -r -a v1_v4_arr <<< "$V1_V4_INPUTS"
read -r -a wiki_arr <<< "$WIKI_CPT_INPUTS"

COT_DIR="$WORK_DIR/step0_qa_cot_v04"
V2_IMG_DIR="$WORK_DIR/step0_v2_images"
REFACTOR_DIR="$WORK_DIR/step1_v1_v4_refactored"
MIDTRAIN_DATA_DIR="$WORK_DIR/step2_midtrain_bundle"
CPT_OUT_DIR="$WORK_DIR/step2_midtrain_ckpt"
CPT_EXPORT_DIR="$WORK_DIR/step2_cpt_hf_merged"
SFT_OUT_DIR="$WORK_DIR/step3_flash_sft_${SFT_ARM}"

mkdir -p "$WORK_DIR"

echo "=== [Step 1a] Generating QA->CoT ($COT_MODE_VER, mode=$COT_MODE) + Clean Direct Twins + CPT Prose ==="
cot_cmd=(
  python3 "$SCRIPT_DIR/gen_cot_from_qa.py"
  --cot-mode "$COT_MODE_VER"
  --output-dir "$COT_DIR"
  --mode "$COT_MODE"
)
if [[ -n "$QA_RAW_INPUT" ]]; then
  cot_cmd+=(--input "$QA_RAW_INPUT")
else
  cot_cmd+=(--demo)
fi
if [[ "$DEMO_MODE" == "1" ]]; then
  cot_cmd+=(--smoke-sample "${SMOKE_SAMPLE_N:-6}")
fi
"${cot_cmd[@]}" >/dev/null

echo "=== [Step 1b] Running ImageGen v2 Domain-Partitioned Pipeline & Multi-Image Doc Packing ==="
v2_cmd=(
  python3 "$SCRIPT_DIR/build_v2_image_pipeline.py"
  --output-dir "$V2_IMG_DIR"
  --mode "$IMG_V2_MODE"
)
if [[ -n "$V2_ENTITIES_INPUT" ]]; then
  v2_cmd+=(--entities "$V2_ENTITIES_INPUT")
  [[ -n "$V1_AUDIT_INPUT" ]] && v2_cmd+=(--v1-audit "$V1_AUDIT_INPUT")
else
  v2_cmd+=(--demo)
fi
"${v2_cmd[@]}" >/dev/null

NEW_QA_INPUTS=${NEW_QA_INPUTS:-"$COT_DIR/qa_sft_mixed_8_2.jsonl $SCRIPT_DIR/examples/style_sft_example.jsonl"}
read -r -a new_qa_arr <<< "$NEW_QA_INPUTS"

echo "=== [Step 1c] Refactoring V1-V4 + Dual-Mode QA-CoT into Replay & Unified SFT ($SFT_ARM) ==="
refactor_cmd=(
  python3 "$SCRIPT_DIR/refactor_v1_v4_pipeline.py"
  --v1-v4-inputs "${v1_v4_arr[@]}"
  --new-qa-inputs "${new_qa_arr[@]}"
  --sft-arm "$SFT_ARM"
  --cot-direct-ratio "${COT_DIRECT_RATIO:-0.7}"
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

echo "=== [Step 2a] Building Instruct-Aware Mid-Training Bundle (30/35/15/20, <=25% synth, <=15% QA prose) ==="
midtrain_inputs=("${wiki_arr[@]}" "$REFACTOR_DIR/cpt_v1_v4_replay.jsonl" "$V2_IMG_DIR/v2_packed_cpt_docs.jsonl" "$V2_IMG_DIR/v2_cf_neg_pairs.jsonl")
if [[ -s "$COT_DIR/qa_midtrain_prose.jsonl" ]]; then
  midtrain_inputs+=("$COT_DIR/qa_midtrain_prose.jsonl")
fi

midtrain_cmd=(
  python3 "$SCRIPT_DIR/build_mm_wiki_cpt.py"
  --mode midtrain
  --inputs "${midtrain_inputs[@]}"
  --exclude-families-from "$REFACTOR_DIR/sft_unified_val.jsonl"
  --output-dir "$MIDTRAIN_DATA_DIR"
  --max-tokens "${MAX_LENGTH:-8192}"
  --max-row-repeats "${MAX_ROW_REPEATS:-2}"
  --max-synth-image-share "${MAX_SYNTH_IMAGE_SHARE:-0.25}"
  --max-qa-prose-share "${MAX_QA_PROSE_SHARE:-0.15}"
)
if [[ "$DEMO_MODE" == "1" ]]; then
  midtrain_cmd+=(--val-ratio 0.20 --total-rows 28)
elif [[ -n ${MIDTRAIN_TOTAL_ROWS:-} ]]; then
  midtrain_cmd+=(--total-rows "$MIDTRAIN_TOTAL_ROWS")
fi
"${midtrain_cmd[@]}" >/dev/null

echo "=== [Step 2b] Single-Stage B300 Flash WSD Mid-Training & HF Export (DRY_RUN=$DRY_RUN) ==="
FLASH_DECAY_ITERS=${STAGE2_DECAY_ITERS:-60}
launch_flag="--dry-run"
[[ "$DRY_RUN" == "0" ]] && launch_flag="--launch"

BACKEND="$BACKEND" TUNER_TYPE="$CPT_TUNER_TYPE" SMOKE="$SMOKE" \
  bash "$CPT27B_DIR/run_cpt_27b_b300.sh" flash \
  --model "$MODEL" \
  --dataset "$MIDTRAIN_DATA_DIR/midtrain_train.jsonl" \
  --val-dataset "$MIDTRAIN_DATA_DIR/midtrain_val.jsonl" \
  --output-dir "$CPT_OUT_DIR" \
  --stage2-decay-iters "$FLASH_DECAY_ITERS" \
  "$launch_flag"

if [[ "$CPT_TUNER_TYPE" == "lora" ]]; then
  export_cmd=(swift export --model "$MODEL" --adapters "$CPT_OUT_DIR/latest" --merge_lora true --output_dir "$CPT_EXPORT_DIR")
else
  export_cmd=(swift export --mcore_model "$CPT_OUT_DIR/latest" --to_hf true --torch_dtype bfloat16 --output_dir "$CPT_EXPORT_DIR")
fi

echo "=== [Step 3] Single Unified Dual-Mode SFT ($SFT_ARM Think-ON + Clean Direct Twins, DRY_RUN=$DRY_RUN) ==="
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
  --max_length "${SFT_MAX_LENGTH:-8192}"
  --truncation_strategy delete
  --strict true
  --enable_channel_loss true
  --micro_batch_size "${SFT_MBS:-2}"
  --global_batch_size "${SFT_GBS:-32}"
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
  echo "=== Dual-Mode Eval Gate Reminder: evaluate $SFT_OUT_DIR under BOTH enable_thinking=true AND enable_thinking=false ==="
  exit 0
fi

"${export_cmd[@]}"
exec "${sft_cmd[@]}"
