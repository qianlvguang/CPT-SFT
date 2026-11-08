#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?Set MODEL to a Hugging Face model ID or /models subdirectory}"
: "${TP_SIZE:?Set TP_SIZE}"
IMAGE=${IMAGE:-vllm/vllm-openai:v0.32.0}
PORT=${PORT:-8000}
BIND_ADDRESS=${BIND_ADDRESS:-127.0.0.1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
HF_CACHE=${HF_CACHE:-"$HOME/.cache/huggingface"}
MODEL_DIR=${MODEL_DIR:-"$HOME/models"}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen-flash-next}
EXPERT_PARALLEL=${EXPERT_PARALLEL:-0}

if ! [[ $TP_SIZE =~ ^[1-9][0-9]*$ ]]; then echo 'TP_SIZE must be a positive integer' >&2; exit 2; fi
if [[ $EXPERT_PARALLEL != 0 && $EXPERT_PARALLEL != 1 ]]; then echo 'EXPERT_PARALLEL must be 0 or 1' >&2; exit 2; fi
if [[ $TP_SIZE == 8 && $EXPERT_PARALLEL == 0 && $MODEL == *Qwen3.8-Flash-Next-FP8* ]]; then
  echo 'Official FP8 checkpoint: TP8 requires expert parallelism; set EXPERT_PARALLEL=1.' >&2; exit 2
fi

mkdir -p "$HF_CACHE" "$MODEL_DIR"
args=(run --rm --gpus all --ipc=host -p "$BIND_ADDRESS:$PORT:8000"
  -v "$HF_CACHE:/root/.cache/huggingface"
  -v "$MODEL_DIR:/models:ro")
if [[ -n ${HF_TOKEN:-} ]]; then args+=(-e HF_TOKEN); fi
if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then args+=(-e CUDA_VISIBLE_DEVICES); fi
args+=("$IMAGE" --model "$MODEL" --served-model-name "$SERVED_MODEL_NAME"
  --host 0.0.0.0 --port 8000 --tensor-parallel-size "$TP_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS" --enable-prefix-caching)
if [[ $EXPERT_PARALLEL == 1 ]]; then args+=(--enable-expert-parallel); fi
# Extra vLLM arguments can be passed directly, preserving shell quoting.
args+=("$@")
if [[ ${DRY_RUN:-0} == 1 ]]; then printf 'docker '; printf '%q ' "${args[@]}"; printf '\n'; exit 0; fi
exec docker "${args[@]}"
