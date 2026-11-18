#!/usr/bin/env bash
set -euo pipefail

# Authoritative B300 wrapper for Qwen3.5-27B CPT profiles:
#   - lite   : Phase A gain experiment (bundle_lite_compat/stage1_train.jsonl, val0, 2 epochs, constant 1e-5)
#   - stage1 : Phase B full-scale Stage 1 (constant 1e-5 + 6% warmup, saves optim/rng)
#   - stage2 : Phase B full-scale Stage 2 WSD linear cooldown
#   - pro    : Phase C incremental error-cluster CPT (50/30/20 mix, cosine 5e-6)
#
# Safety gates:
#   - Dry-run is default; pass `--launch` (and do not set DRY_RUN=1) to actually execute.
#   - `--model <path>` (or env MODEL / MODEL_PATH) is strictly required.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$(cd "$SCRIPT_DIR/../lib" && pwd)"

# shellcheck source=../lib/_core.sh
source "$LIB_DIR/_core.sh"
# shellcheck source=../lib/hw_b300.sh
source "$LIB_DIR/hw_b300.sh"

usage() {
  cat <<'EOF'
Usage:
  run_cpt_27b_b300.sh <lite|stage1|stage2|pro> --model <MODEL_PATH> [options] [-- extra_args...]

Options:
  --model <path>               Required base model / checkpoint path (or set MODEL env)
  --dataset <path>             Override training JSONL dataset
  --val-dataset <path>         Override validation JSONL dataset
  --output-dir <path>          Override checkpoint output directory
  --epochs <n>                 Override num_train_epochs
  --train-iters <n>            Override train_iters explicitly
  --stage2-decay-iters <n>     Explicit WSD linear decay steps for stage2
  --tuner-type <full|lora>     Override TUNER_TYPE (default: full)
  --smoke                      Run 10-step smoke check
  --launch                     Actually launch training (default is dry-run unless --launch is passed)
  --dry-run                    Force dry-run even if --launch is passed
  -h, --help                   Show this help
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

PROFILE=""
case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  lite|stage1|stage2|pro)
    PROFILE="$1"
    shift
    ;;
  *)
    die 2 "First argument must be profile: lite | stage1 | stage2 | pro (got '${1:-}')"
    ;;
esac

CLI_MODEL="${MODEL:-${MODEL_PATH:-}}"
CLI_DATASET="${DATASET:-}"
CLI_VAL_DATASET="${VAL_DATASET:-}"
CLI_OUTPUT_DIR="${OUTPUT_DIR:-}"
CLI_EPOCHS="${EPOCHS:-}"
CLI_TRAIN_ITERS="${TRAIN_ITERS:-}"
CLI_STAGE2_DECAY_ITERS="${STAGE2_DECAY_ITERS:-}"
CLI_TUNER_TYPE="${TUNER_TYPE:-}"
CLI_SMOKE="${SMOKE:-0}"
LAUNCH_FLAG=0
FORCE_DRY_RUN="${DRY_RUN:-}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      [[ $# -ge 2 ]] || die 2 "--model requires a value"
      CLI_MODEL="$2"
      shift 2
      ;;
    --dataset)
      [[ $# -ge 2 ]] || die 2 "--dataset requires a value"
      CLI_DATASET="$2"
      shift 2
      ;;
    --val-dataset)
      [[ $# -ge 2 ]] || die 2 "--val-dataset requires a value"
      CLI_VAL_DATASET="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || die 2 "--output-dir requires a value"
      CLI_OUTPUT_DIR="$2"
      shift 2
      ;;
    --epochs)
      [[ $# -ge 2 ]] || die 2 "--epochs requires a value"
      CLI_EPOCHS="$2"
      shift 2
      ;;
    --train-iters)
      [[ $# -ge 2 ]] || die 2 "--train-iters requires a value"
      CLI_TRAIN_ITERS="$2"
      shift 2
      ;;
    --stage2-decay-iters)
      [[ $# -ge 2 ]] || die 2 "--stage2-decay-iters requires a value"
      CLI_STAGE2_DECAY_ITERS="$2"
      shift 2
      ;;
    --tuner-type)
      [[ $# -ge 2 ]] || die 2 "--tuner-type requires a value"
      CLI_TUNER_TYPE="$2"
      shift 2
      ;;
    --smoke)
      CLI_SMOKE=1
      shift
      ;;
    --launch)
      LAUNCH_FLAG=1
      shift
      ;;
    --dry-run)
      FORCE_DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$CLI_MODEL" ]]; then
  die 2 "--model <path> (or MODEL env var) is required."
fi

if [[ "$FORCE_DRY_RUN" == "1" ]]; then
  EFFECTIVE_DRY_RUN=1
  EFFECTIVE_LAUNCH=0
elif [[ "$LAUNCH_FLAG" == "1" ]]; then
  EFFECTIVE_DRY_RUN=0
  EFFECTIVE_LAUNCH=1
else
  EFFECTIVE_DRY_RUN=1
  EFFECTIVE_LAUNCH=0
fi

# Configurable roots (override DATA_ROOT / OUTPUT_ROOT or AUTOVISION_ROOT for cluster paths)
AUTOVISION_ROOT="${AUTOVISION_ROOT:-/data/autovision-cfs/autovision-notebook}"
DATA_ROOT="${DATA_ROOT:-${AUTOVISION_ROOT}/algo4/tpz/data/v-CPT/pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${AUTOVISION_ROOT}/algo4/tpz/fine-tunning}"

case "$PROFILE" in
  lite)
    PHASE="lite"
    DATASET_VAL="${CLI_DATASET:-${DATA_ROOT}/bundle_lite_compat/stage1_train.jsonl}"
    VAL_DATASET_VAL="${CLI_VAL_DATASET:-}"
    OUTPUT_DIR_VAL="${CLI_OUTPUT_DIR:-${OUTPUT_ROOT}/cpt-qwen3_5-27b-lite-v1.0.0/stage1}"
    EPOCHS_VAL="${CLI_EPOCHS:-${LITE_EPOCHS:-2}}"
    LR_VAL="${LR:-1e-5}"
    MIN_LR_VAL="${MIN_LR:-1e-6}"
    SCHED_VAL="${LR_DECAY_STYLE:-constant}"
    WARMUP_VAL="${LR_WARMUP_FRACTION:-0.06}"
    NO_SAVE_OPTIM_VAL="${NO_SAVE_OPTIM:-false}"
    NO_SAVE_RNG_VAL="${NO_SAVE_RNG:-false}"
    ;;
  stage1)
    PHASE="stage1"
    DATASET_VAL="${CLI_DATASET:-${DATA_ROOT}/bundle/stage1_train.jsonl}"
    VAL_DATASET_VAL="${CLI_VAL_DATASET:-${DATA_ROOT}/bundle/val.jsonl}"
    OUTPUT_DIR_VAL="${CLI_OUTPUT_DIR:-${OUTPUT_ROOT}/cpt-qwen3_5-27b-flash-v1.0.0/stage1}"
    EPOCHS_VAL="${CLI_EPOCHS:-1}"
    LR_VAL="${LR:-1e-5}"
    MIN_LR_VAL="${MIN_LR:-1e-6}"
    SCHED_VAL="${LR_DECAY_STYLE:-constant}"
    WARMUP_VAL="${LR_WARMUP_FRACTION:-0.06}"
    NO_SAVE_OPTIM_VAL="${NO_SAVE_OPTIM:-false}"
    NO_SAVE_RNG_VAL="${NO_SAVE_RNG:-false}"
    ;;
  stage2)
    PHASE="stage2"
    DATASET_VAL="${CLI_DATASET:-${DATA_ROOT}/bundle/stage2_cooldown.jsonl}"
    VAL_DATASET_VAL="${CLI_VAL_DATASET:-${DATA_ROOT}/bundle/val.jsonl}"
    OUTPUT_DIR_VAL="${CLI_OUTPUT_DIR:-${OUTPUT_ROOT}/cpt-qwen3_5-27b-flash-v1.0.0/stage2}"
    EPOCHS_VAL="${CLI_EPOCHS:-1}"
    LR_VAL="${LR:-1e-5}"
    MIN_LR_VAL="${MIN_LR:-1e-6}"
    SCHED_VAL="${LR_DECAY_STYLE:-WSD}"
    WARMUP_VAL="${LR_WARMUP_FRACTION:-0}"
    NO_SAVE_OPTIM_VAL="${NO_SAVE_OPTIM:-false}"
    NO_SAVE_RNG_VAL="${NO_SAVE_RNG:-false}"
    CLI_STAGE2_DECAY_ITERS="${CLI_STAGE2_DECAY_ITERS:-28}"
    ;;
  pro)
    PHASE="pro"
    DATASET_VAL="${CLI_DATASET:-${DATA_ROOT}/bundle_pro_inc/pro_train.jsonl}"
    VAL_DATASET_VAL="${CLI_VAL_DATASET:-${DATA_ROOT}/bundle_pro_inc/pro_val.jsonl}"
    OUTPUT_DIR_VAL="${CLI_OUTPUT_DIR:-${OUTPUT_ROOT}/cpt-qwen3_5-27b-pro-inc/stage1}"
    EPOCHS_VAL="${CLI_EPOCHS:-1}"
    LR_VAL="${LR:-5e-6}"
    MIN_LR_VAL="${MIN_LR:-1e-6}"
    SCHED_VAL="${LR_DECAY_STYLE:-cosine}"
    WARMUP_VAL="${LR_WARMUP_FRACTION:-0.05}"
    NO_SAVE_OPTIM_VAL="${NO_SAVE_OPTIM:-true}"
    NO_SAVE_RNG_VAL="${NO_SAVE_RNG:-true}"
    ;;
esac

export MODEL="$CLI_MODEL"
export DATASET="$DATASET_VAL"
export VAL_DATASET="$VAL_DATASET_VAL"
export OUTPUT_DIR="$OUTPUT_DIR_VAL"
export PHASE="$PHASE"
export TUNER_TYPE="${CLI_TUNER_TYPE:-full}"
export EPOCHS="$EPOCHS_VAL"
export LR="$LR_VAL"
export MIN_LR="$MIN_LR_VAL"
export LR_DECAY_STYLE="$SCHED_VAL"
export LR_WARMUP_FRACTION="$WARMUP_VAL"
export NO_SAVE_OPTIM="$NO_SAVE_OPTIM_VAL"
export NO_SAVE_RNG="$NO_SAVE_RNG_VAL"
export SMOKE="$CLI_SMOKE"
export DRY_RUN="$EFFECTIVE_DRY_RUN"
export LAUNCH="$EFFECTIVE_LAUNCH"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-$DEF_MICRO_BATCH_SIZE}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$DEF_GLOBAL_BATCH_SIZE}"
export MAX_LENGTH="${MAX_LENGTH:-$DEF_MAX_LENGTH}"
export TP_SIZE="${TP_SIZE:-$DEF_TP_SIZE}"
export GRADIENT_ACCUMULATION_FUSION="${GRADIENT_ACCUMULATION_FUSION:-$DEF_GRADIENT_ACCUMULATION_FUSION}"

if [[ -n "$CLI_TRAIN_ITERS" ]]; then
  export TRAIN_ITERS="$CLI_TRAIN_ITERS"
fi
if [[ -n "$CLI_STAGE2_DECAY_ITERS" ]]; then
  export STAGE2_DECAY_ITERS="$CLI_STAGE2_DECAY_ITERS"
fi

if (( ${#EXTRA_ARGS[@]} > 0 )); then
  exec bash "$SCRIPT_DIR/train_cpt_qwen3_5_27b_megatron.sh" "${EXTRA_ARGS[@]}"
else
  exec bash "$SCRIPT_DIR/train_cpt_qwen3_5_27b_megatron.sh"
fi
