#!/usr/bin/env bash
# AutoVision 8xB300 (sm_103 Blackwell Ultra, 268.6 GiB/GPU) hardware profile defaults.

if [[ -n ${_CPT_SFT_HW_B300_SH_LOADED:-} ]]; then
  return 0 2>/dev/null || exit 0
fi
_CPT_SFT_HW_B300_SH_LOADED=1

# Topology & allocation defaults (Runbook §3 & §4)
DEF_CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
DEF_NPROC_PER_NODE=8
DEF_TP_SIZE=4
DEF_PP_SIZE=1
DEF_CP_SIZE=1

# Batch & sequence defaults (Calibrated B300 profile: MBS=2, GBS=16, MAX_LENGTH=8192 -> ~131K tok/step)
# Keeps 4K-8K KB text intact (unlike 16K over-packing which starved short sch/VLM rows to 7 steps)
DEF_MICRO_BATCH_SIZE=2
DEF_GLOBAL_BATCH_SIZE=16
DEF_MAX_LENGTH=8192
DEF_IMAGE_MAX_TOKEN_NUM=16384
DEF_VIDEO_MAX_TOKEN_NUM=128
DEF_FPS_MAX_FRAMES=16

# Kernel & fusion guardrails (Runbook §6.1 & §6.2)
DEF_ATTENTION_BACKEND="flash"
# Fix APEX fused_weight_gradient_mlp_cuda crash in pure-python apex environments
DEF_GRADIENT_ACCUMULATION_FUSION="false"

validate_b300_27b_tp_gqa() {
  local tp_size="${1:?tp_size required}"
  # Qwen3.5-27B FullAttn uses num_key_value_heads=4; TP=8 violates 4 % TP == 0.
  if ! [[ "$tp_size" =~ ^[1-9][0-9]*$ ]] || (( 4 % tp_size != 0 )); then
    die 2 "Qwen3.5-27B has num_key_value_heads=4; TP_SIZE=$tp_size cannot divide 4 (TP_SIZE=8 fails GQA check). Use TP_SIZE=4, 2, or 1."
  fi
}

export_b300_runtime_env() {
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$DEF_CUDA_VISIBLE_DEVICES}"
  export NPROC_PER_NODE="${NPROC_PER_NODE:-$DEF_NPROC_PER_NODE}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
  export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-$DEF_IMAGE_MAX_TOKEN_NUM}"
  export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-$DEF_VIDEO_MAX_TOKEN_NUM}"
  export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-$DEF_FPS_MAX_FRAMES}"
}

verify_te_sm103_preflight() {
  # Runbook §6.2 lesson: top-level `import transformer_engine` can false-pass;
  # must import `transformer_engine.pytorch` to catch PyTorch ABI / cubin mismatches.
  if [[ "${PREFLIGHT_TE:-0}" != "1" ]]; then
    return 0
  fi
  local py_bin="${PYTHON_BIN:-python3}"
  if ! "$py_bin" -c "import transformer_engine.pytorch as te_torch; print('TE_PYTORCH_OK')" >/dev/null 2>&1; then
    die 1 "TransformerEngine PyTorch preflight failed (`import transformer_engine.pytorch`). Check sm_103 cubin and PyTorch ABI compatibility."
  fi
  log_info "TransformerEngine PyTorch preflight passed."
}
