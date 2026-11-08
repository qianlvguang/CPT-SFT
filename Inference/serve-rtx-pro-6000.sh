#!/usr/bin/env bash
set -euo pipefail

# Conservative starting point: four 96 GB RTX PRO 6000 cards, FP8, TP+EP.
# This combination must be verified on the target machine.
export MODEL=${MODEL:-Qwen/Qwen3.8-Flash-Next-FP8}
export TP_SIZE=${TP_SIZE:-4}
export EXPERT_PARALLEL=${EXPERT_PARALLEL:-1}
exec "$(dirname "$0")/serve-common.sh" "$@"
