#!/usr/bin/env bash
set -euo pipefail

# Default: one B300/GB300 with the official FP8 checkpoint.
export MODEL=${MODEL:-Qwen/Qwen3.8-Flash-Next-FP8}
export TP_SIZE=${TP_SIZE:-1}
export EXPERT_PARALLEL=${EXPERT_PARALLEL:-0}
exec "$(dirname "$0")/serve_common.sh" "$@"
