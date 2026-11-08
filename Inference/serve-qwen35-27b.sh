#!/usr/bin/env bash
set -euo pipefail

# FP8, single GPU starting point for B300 and RTX PRO 6000 (96 GB).
export MODEL=${MODEL:-Qwen/Qwen3.5-27B-FP8}
export TP_SIZE=${TP_SIZE:-1}
export EXPERT_PARALLEL=${EXPERT_PARALLEL:-0}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen35-27b}
if [[ $EXPERT_PARALLEL != 0 ]]; then
  echo 'Qwen3.5-27B is dense; EXPERT_PARALLEL must be 0.' >&2; exit 2
fi
exec "$(dirname "$0")/serve-common.sh" "$@"
