#!/usr/bin/env bash
# =============================================================================
# vis-double: heterogeneous LoRA SFT of Qwen3.5-27B (dense VLM) — ms-swift +
# Megatron-Core, on top of the vis-opt recipe (same skeleton / H200 tuning).
# Target hardware: NVIDIA H200 (Hopper, sm90, 141 GB/card, 8 GPUs)
#
# DOUBLE = TWO heterogeneous dimensions (vs vis-opt's single LLM LoRA):
#   1. module-heterogeneous adapters (S1 perception/domain adaptation):
#        | module  | mode        | rank/α   | LR    |
#        |---------|-------------|----------|-------|
#        | ViT     | LoRA        | 16/32    | 1e-5  |  (rank_pattern via plugin)
#        | aligner | full-param  | —        | 5e-6  |  (freeze_aligner+modules_to_save)
#        | LLM     | LoRA        | 64/128   | 3e-5  |  (global --lora_rank/--lora_alpha)
#   2. per-prompt-channel loss logging (V1..V5, --enable_channel_loss):
#      the five ultra_x parquets carry a `variant` column (1/2/3/4/5); the
#      plugin translates variant -> 'V%d' channel per sample (LOGGING ONLY — zero
#      effect on the training loss or throughput).
#
# INPUT DATA FORMAT (learned from the ultra_x parquets — v1/v2/v3/v4 verified
# on disk, v5 governance QA added 2026-09-23; see maid-3's ultra_x build):
#   columns: ['messages', 'images', 'enable_thinking', 'sample_id', 'cate_id',
#             'variant']
#     * messages: std chat format, assistant = pure JSON caption (V3 is
#       reason-stripped upstream: bak_reason backup kept on disk)
#     * images  : relative path list (base = /data/autovision-cfs/autovision-notebook)
#     * variant : int 1/2/3/4/5 — which prompt version produced this row
#   Each parquet is fed INDEPENDENTLY (no merging; swift --dataset takes a
#   list — data_args.py:75 `dataset: List[str]`); V4 joins by appending its
#   path to TRAIN_FILES when ready.
#
# SINGLE-FILE DELIVERY: the rank-pattern injection plugin is EMBEDDED below
# (section 1b, heredoc) and self-generated at launch next to this script as
# _vis_double_plugin.py — nothing else to upload/deploy. The plugin does:
#   patch A: RowPreprocessor.batched_preprocess wrap — variant -> 'V%d' channel
#     (per-sample column translation at dataset-encode time; full verified
#     chain: preprocess passthrough (core.py:514-530) -> StdTemplateInputs.
#     from_dict picks 'channel' (template_inputs.py:62) -> encode (base.py:648)
#     -> packing_row keeps per-sample list (base.py:688) -> collator emits
#     flat list (base.py:1874) -> megatron trainer pops it (trainer.py:110)
#     -> _compute_channel_loss cu_seqlens branch -> loss_V1..V5 metrics)
#   patch B: utils.LoraConfig wrap — inject rank_pattern/alpha_pattern
#     (swift CLI exposes only a GLOBAL --lora_rank/--lora_alpha; peft
#     suffix-matches pattern keys against module names: key r'visual\..*'
#     covers every visual.visual.* linear, LLM keys fall back to global
#     r64/a128. Verified against installed swift + live RTX-LoRA logs.)
#
# S1 PARAMETER SET (external-plan values, master-approved 2026-09-22;
# epochs 1->2, ckpt every 1/3 epoch, master-ordered 2026-09-23):
#   warmup 3%, cosine, weight_decay 0.01, lora_dropout 0.05, TWO epochs with
#   SIX checkpoints at 1/3-epoch marks (1/3, 2/3, 1, 4/3, 5/3, 2 epochs;
#   SAVE_STEPS = iters_per_epoch/3, fixed after SMOKE reports the real
#   iters_per_epoch; SAVE_TOTAL_LIMIT=6 keeps ALL six).
#
# GLOBAL BATCH NOTE: plan's "global batch 128" assumes unpacked sequences.
# With packing=true + MAX_LENGTH=8192 each step consumes 32 windows ~=
# 150+ samples (avg ~4.8 samples/window), already >128. GBS stays 32 so
# S1/S2/S3 and control arms share one consistent GBS.
#
# TP=4 NOTE (master-ordered 2026-09-23): 8 GPUs / TP=4 -> DP=2, throughput
# ~2x vs TP=8 (per-step wall-time ~halved, step count unchanged — GBS is
# global). Per-card weight shard 27B/4 ~= 13.5GB bf16 (vs 6.75GB @TP8),
# easily absorbed by the MAX_LENGTH=8192 activation savings on 141GB H200s.
# kv_heads divide 4 cleanly. channel loss handles DP=2 natively
# (trainer.py all_gather_object key-sync across DP group).
#
# MEMORY LEDGER (H200 141GB, TP=8, vs vis_opt baseline):
#   LoRA static per card @TP=4: weights 13.5GB + ViT LoRA r16 (tiny)
#   + merger FULL-PARAM (bf16 grads + fp32 optim, TP-sharded) ~3GB -> ~17GB
#   static; activations halved by MAX_LENGTH=8192 (master-ordered
#   2026-09-23). If step-0 autotune still OOMs:
#   RECOMPUTE_GRANULARITY=full -> warm triton cache (see vis_opt header).
#
# DISK LEDGER (NFS ~98% full!): MERGE_LORA=true saves merged full weights per
# checkpoint (~54GB x SAVE_TOTAL_LIMIT). The rotation grinder (save_total_limit)
# eats the oldest checkpoint ~2min after a new save — rescue anything you care
# about DURING the run (battle scar: cpt-2560, 2026-09-22).
#
# SMOKE / RESUME (mechanisms battle-proven in the cpt_recovery campaign):
#   SMOKE=1   bash this-script   # 10 iters, throwaway dir, no saves
#   RESUME=1  nohup bash this-script > vis_double.log 2>&1 &
#   (SMOKE must OMIT --num_train_epochs: megatron_args.py:934 overwrites
#    train_iters whenever num_train_epochs is not None.)
#
# VERIFICATION CHECKLIST (grep the log after launch):
#   1. '[vis_double_plugin] variant->channel patch installed'  (patch A live)
#   2. '[vis_double_plugin] rank_pattern=...'                  (patch B live)
#   3. lora_config: LoraConfig(... rank_pattern={'visual\\..*': 16} ...)
#   4. target_modules regex: merger excluded via (?!(visual.visual.merger))
#   5. trainable params: ViT adapters r16-sized; merger full-param
#      (modules_to_save); vit_lr/aligner_lr param groups present
#   6. 'loss_V1' 'loss_V2' 'loss_V3' 'loss_V4' 'loss_V5' metric keys  (channel loss)
#
# Every knob is env-overridable, e.g.:
#   SMOKE=1 bash train_sft_qwen3_5_27b_megatron_lora_h200_vis_double.sh
#   SAVE_STEPS=650 RESUME=1 nohup bash ... > vis_double.log 2>&1 &
# =============================================================================
# -----------------------------------------------------------------------------
# 0. Environment activation & working dir
# -----------------------------------------------------------------------------
source /data/autovision-cfs/autovision-notebook/tpz_xayah/bin/activate
cd /data/autovision-cfs/autovision-notebook/
set -xeuo pipefail

# Memory-friendly defaults
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_COMPILE_DISABLE=1
export WANDB_MODE=${WANDB_MODE:-disabled}

# CPU-side throughput guards
export TOKENIZERS_PARALLELISM=False
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS=1

# VLM token budgets
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-16384}
export VIDEO_MAX_TOKEN_NUM=${VIDEO_MAX_TOKEN_NUM:-128}
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-16}

# Dataset preprocess cache on NFS (master-ordered 2026-09-23): swift writes
# encoded-dataset cache (datasets.map cache_file_name) under
# modelscope get_cache_dir()/datasets/map_cache. Default /root/.cache/... is
# POD-LOCAL (lost on pod rebuild) — pin it to NFS so the FIRST launch pays
# the full encode (hours) and every later launch / pod starts fast.
export MODELSCOPE_CACHE_HOME=${MODELSCOPE_CACHE_HOME:-/data/autovision-cfs/autovision-notebook/logos-remote/sft/cache}

# -----------------------------------------------------------------------------
# 1a. Model & data (ultra_x v1/v2/v3 INDEPENDENT parquets — no merging)
# -----------------------------------------------------------------------------
MODEL_PATH=${MODEL_PATH:-/data/autovision-cfs/autovision-notebook/algo4/models/Qwen3.5-27B}
[ -f "${MODEL_PATH}/config.json" ] || { echo "[error] missing ${MODEL_PATH}/config.json" >&2; exit 1; }

TEST_DIR=/data/autovision-cfs/autovision-notebook/algo4/tpz/test
# Each parquet: messages/images/enable_thinking/sample_id/cate_id/variant.
# The variant column (1/2/3/4) is translated to the channel column ('V%d') by
# the plugin at encode time — the parquets stay untouched on disk.
# S1 mini recipe (external plan 800k proportions, e2e-probed 2026-09-23;
# v5 added 2026-09-23): V1 360k / V2 200k / V3 120k / V4 120k / V5 60k
# (governance QA: fenced-json {"res","reason"} assistant BY DESIGN, 194
# text-only rows without images are intentional; variant=5 column injected
# 2026-09-23, md5 20537b5c5b1a843495ca398b8b2c9d62). Full-scale v1/v2/v3
# parquets stay on disk for the later stage — switch paths back when
# master orders.
TRAIN_FILES=(
    "${TEST_DIR}/train_caption_ultra_x_v1mini.parquet"  # 360,000 rows
    "${TEST_DIR}/train_caption_ultra_x_v2mini.parquet"  # 200,000 rows
    "${TEST_DIR}/train_caption_ultra_x_v3mini.parquet"  # 120,000 rows
    "${TEST_DIR}/train_caption_ultra_x_v4mini.parquet"  # 120,000 rows
    "${TEST_DIR}/train_caption_ultra_x_v5mini.parquet"  # 60,000 rows (governance QA)
)
for _f in "${TRAIN_FILES[@]}"; do
    [ -f "${_f}" ] || { echo "[error] TRAIN file not found: ${_f}" >&2; exit 1; }
done
SPLIT_DATASET_RATIO=${SPLIT_DATASET_RATIO:-0}

# -----------------------------------------------------------------------------
# 1b. Self-generate the vis_double plugin (rank_pattern + variant->channel)
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_PATH=${PLUGIN_PATH:-${SCRIPT_DIR}/_vis_double_plugin.py}

cat > "${PLUGIN_PATH}" <<'VIS_DOUBLE_PLUGIN_EOF'
r"""vis_double plugin (auto-generated by train_sft_qwen3_5_27b_megatron_lora_h200_vis_double.sh).

Loaded at ARGS-PARSING time via --external_plugins (arguments/base_args/
base_args.py:137-152 imports the plugin before the trainer is built).

patch A: variant -> channel translation.
    The ultra_x parquets carry a `variant` column (int 1/2/3/4/5) but swift's
    channel loss expects a per-sample `channel` column (string, whitelisted in
    preprocessor standard_keys). This patch wraps RowPreprocessor.
    batched_preprocess — the single entry point that receives the RAW columns
    (before remove_useless_columns drops non-standard ones) and returns the
    encoded rows — and translates variant to 'V%d' in place. The channel
    column then flows through swift's official path untouched:
    preprocess passthrough -> StdTemplateInputs.from_dict (template_inputs.py
    :62 picks 'channel') -> encode (template/base.py:648) -> packing_row keeps
    the per-sample list (base.py:688) -> collator emits the flat list
    (base.py:1874) -> megatron trainer pops 'channel' (trainer.py:110) ->
    _compute_channel_loss cu_seqlens branch -> loss_V1..V5 metrics.

patch B: ViT rank_pattern injection (r16/a32; LLM stays global r64/a128).
    swift's CLI exposes only a single global --lora_rank/--lora_alpha; peft's
    LoraConfig natively supports rank_pattern/alpha_pattern. This patch wraps
    the constructor referenced (as a module global) inside swift.megatron.
    utils.utils.prepare_adapter. Suffix-match semantics (lora_layers.py:
    388-390): pattern key r'visual\..*' matches every visual.visual.* linear
    (the `.*\.` prefix consumes the first 'visual.' segment); LLM keys like
    language_model.layers.* match nothing -> global r64/a128 fallback.
    DO NOT use r'visual\.visual\..*' — never matches (same reason).
"""

# --- patch A: variant -> channel -------------------------------------------
from swift.dataset.preprocessor.core import RowPreprocessor as _RowPreprocessor

_orig_batched_preprocess = _RowPreprocessor.batched_preprocess
_vdc_translated_once = [False]


def _vis_double_batched_preprocess(self, batched_row, *, strict,
                                   ignore_max_length_error):
    batched_row = dict(batched_row)
    if 'channel' not in batched_row and 'variant' in batched_row:
        try:
            ch = ['V%d' % v if v is not None else None
                  for v in batched_row['variant']]
            batched_row['channel'] = ch
            if not _vdc_translated_once[0]:
                _vdc_translated_once[0] = True
                print('[vis_double_plugin] first variant->channel batch: '
                      f'{len(ch)} rows, e.g. {ch[:8]}')
        except Exception as e:
            print(f'[vis_double_plugin] variant->channel translation failed: {e!r}')
    return _orig_batched_preprocess(self, batched_row, strict=strict,
                                    ignore_max_length_error=ignore_max_length_error)


if not getattr(getattr(_RowPreprocessor, 'batched_preprocess', None),
               '_vis_double_patched', False):
    _vis_double_batched_preprocess._vis_double_patched = True
    _RowPreprocessor.batched_preprocess = _vis_double_batched_preprocess
    print('[vis_double_plugin] variant->channel patch installed '
          '(RowPreprocessor.batched_preprocess)')

# --- patch B: ViT rank_pattern (r16/a32) ------------------------------------
from swift.megatron.utils import utils as _swift_megatron_utils

VIS_DOUBLE_VIT_RANK = 16
VIS_DOUBLE_VIT_ALPHA = 32
_RANK_PATTERN_KEY = r'visual\..*'

_OrigLoraConfig = _swift_megatron_utils.LoraConfig


def _vis_double_lora_config(*args, **kwargs):
    kwargs.setdefault('rank_pattern', {_RANK_PATTERN_KEY: VIS_DOUBLE_VIT_RANK})
    kwargs.setdefault('alpha_pattern', {_RANK_PATTERN_KEY: VIS_DOUBLE_VIT_ALPHA})
    cfg = _OrigLoraConfig(*args, **kwargs)
    print(f'[vis_double_plugin] rank_pattern={dict(cfg.rank_pattern)} '
          f'alpha_pattern={dict(cfg.alpha_pattern)} '
          f'(global r={cfg.r} alpha={cfg.lora_alpha})')
    return cfg


# Idempotency guard: importlib caches modules, so a re-import in the same
# process normally does NOT re-run this top-level code; the guard protects
# against any non-cached reload path double-wrapping the constructor.
if not getattr(getattr(_swift_megatron_utils, 'LoraConfig', None),
               '_vis_double_patched', False):
    _vis_double_lora_config._vis_double_patched = True
    _swift_megatron_utils.LoraConfig = _vis_double_lora_config
VIS_DOUBLE_PLUGIN_EOF
[ -f "${PLUGIN_PATH}" ] || { echo "[error] failed to generate ${PLUGIN_PATH}" >&2; exit 1; }
echo "[vis_double] plugin generated: ${PLUGIN_PATH}"

# -----------------------------------------------------------------------------
# 2. Parallelism (TP * PP * DP * CP * EP == NUM_GPUS; H200 8x141GB)
# -----------------------------------------------------------------------------
NUM_GPUS=${NUM_GPUS:-8}
TP_SIZE=${TP_SIZE:-4}             # master-ordered 2026-09-23: TP=4 -> DP=2,
                                  # ~2x throughput; kv_heads>=4 divides OK
PP_SIZE=${PP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}            # 27B dense -> EP=1
SEQUENCE_PARALLEL=${SEQUENCE_PARALLEL:-true}
USE_DISTRIBUTED_OPTIMIZER=${USE_DISTRIBUTED_OPTIMIZER:-false}  # LoRA+merger optimizer state tiny

# -----------------------------------------------------------------------------
# 3. Training hyper-params (S1 plan values)
# -----------------------------------------------------------------------------
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}       # vis_opt-proven on H200 141GB
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}    # packed windows; see header note
MAX_LENGTH=${MAX_LENGTH:-8192}                 # master-ordered 2026-09-23
                                               # (was 16384; activation memory
                                               # halves. All mini rows fit:
                                               # vision ~196 tok/img + short
                                               # prompts/answers, p50 far
                                               # below 8192)
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}        # 2 epochs (master-ordered
                                               # 2026-09-23; was 1). Cosine
                                               # spans the FULL 2-epoch
                                               # iters; min_lr reached at
                                               # the very end.

LR=${LR:-3e-5}                                # LLM LoRA peak LR (plan)
MIN_LR=${MIN_LR:-3e-6}                        # cosine floor = LR/10
LR_WARMUP_FRACTION=${LR_WARMUP_FRACTION:-0.03}  # plan: 3% per stage
LR_DECAY_STYLE=${LR_DECAY_STYLE:-cosine}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}            # plan: 0.01
CLIP_GRAD=${CLIP_GRAD:-1.0}

# Per-module LRs (megatron-native; 'visual.*' prefix groups, aligner split out)
VIT_LR=${VIT_LR:-1e-5}                        # ViT LoRA adapters
ALIGNER_LR=${ALIGNER_LR:-5e-6}                # merger full-param

RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-selective}
RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-uniform}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flash}   # sm90 flash-attn OK (head_dim 256)

PACKING=${PACKING:-true}
PADDING_FREE=${PADDING_FREE:-false}           # packing=true internally forces
                                              # padding_free=True (verified:
                                              # args.json of live runs) ->
                                              # channel loss takes the correct
                                              # cu_seqlens branch. Keep false.

# -----------------------------------------------------------------------------
# 4. Tuner config (double-heterogeneous: ViT LoRA r16 / aligner full / LLM LoRA r64)
# -----------------------------------------------------------------------------
TUNER_TYPE=${TUNER_TYPE:-lora}
LORA_RANK=${LORA_RANK:-64}                    # global default = LLM (ViT overridden by plugin)
LORA_ALPHA=${LORA_ALPHA:-128}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}            # plan: 0.05 (also swift's default)
MERGE_LORA=${MERGE_LORA:-true}                # RTX-LoRA-proven path (adapter-only
                                              # save on VLM has error history)
FREEZE_LLM=${FREEZE_LLM:-false}               # MUST be false (LoRA semantics)
FREEZE_VIT=${FREEZE_VIT:-false}               # vis_double: ViT gets LoRA (r16 via plugin)
FREEZE_ALIGNER=${FREEZE_ALIGNER:-true}        # EXCLUDES merger from LoRA targets;
                                              # full-param via modules_to_save below
MODULES_TO_SAVE=${MODULES_TO_SAVE:-visual.visual.merger}  # megatron-side full path
                                              # (peft suffix-match; identical to the
                                              # name in the live target_modules regex)

ENABLE_CHANNEL_LOSS=${ENABLE_CHANNEL_LOSS:-true}  # logging-only, per-channel metrics

# -----------------------------------------------------------------------------
# 5. MTP head (S1 default OFF to keep the stage clean)
# -----------------------------------------------------------------------------
MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-0}
MTP_LOSS_SCALING_FACTOR=${MTP_LOSS_SCALING_FACTOR:-0.1}

# -----------------------------------------------------------------------------
# 6. Output & checkpointing
# -----------------------------------------------------------------------------
OUTPUT_DIR=${OUTPUT_DIR:-/data/autovision-cfs/autovision-notebook/algo4/tpz/ft/sft-qwen3_5-27b-vis-double}
SAVE_STRATEGY=${SAVE_STRATEGY:-steps}
# SAVE_STEPS = one-third of an epoch (SIX checkpoints at 1/3, 2/3, 1, 4/3,
#   5/3, 2 epochs = every ~16.7% of the 2-epoch run; save_strategy=epoch only
#   supports INTEGER epochs, so every non-integer point REQUIRES steps mode):
#   SAVE_STEPS = iters_per_epoch / 3, where iters_per_epoch is printed by the
#   dataloader build at SMOKE — fill the number below after SMOKE, BEFORE the
#   live launch. Rough estimate @8192, GBS=32, ~4.8 samples/window:
#   860,000 rows (incl. v5 60k) -> ~1350-2700 iters/epoch -> SAVE_STEPS ~= 450-900.
#   Full-scale v1+v2+v3 = 1,733,586 rows (use when switching back).
#   Disk note: 6 merged ckpts x ~54GB = ~324GB on the 98%-full NFS — rescue
#   per-field-eval samples DURING the run if disk pressure rises.
SAVE_STEPS=${SAVE_STEPS:-512}                 # PLACEHOLDER — set to
                                               # iters_per_epoch/3 after SMOKE
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-6}       # exactly 6 ckpts (1/3..2 ep)
                                               # — all six survive; the
                                               # rotation grinder only bites
                                               # at the 7th (never reached)
SAVE_SAFETENSORS=${SAVE_SAFETENSORS:-true}
NO_SAVE_OPTIM=${NO_SAVE_OPTIM:-true}          # merged ckpts are weights anyway
NO_SAVE_RNG=${NO_SAVE_RNG:-true}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-32}
DATASET_NUM_PROC=${DATASET_NUM_PROC:-16}
PACKING_NUM_PROC=${PACKING_NUM_PROC:-16}
LOAD_FROM_CACHE_FILE=${LOAD_FROM_CACHE_FILE:-true}  # read preprocessed cache
                                               # when present (swift default
                                               # true; flag explicit here).
                                               # Cache WRITES are automatic
                                               # (datasets.map cache_file_name
                                               # under MODELSCOPE_CACHE_HOME
                                               # above) — first launch
                                               # re-encodes all images (hours)
                                               # and PERSISTS the cache on
                                               # NFS; later launches fast.

# -----------------------------------------------------------------------------
# 7. SMOKE / RESUME modes (cpt-campaign-proven)
# -----------------------------------------------------------------------------
SMOKE=${SMOKE:-0}
RESUME=${RESUME:-0}
TRAIN_ITERS=${TRAIN_ITERS:-}

if [ "${SMOKE}" = "1" ]; then
    TRAIN_ITERS=${TRAIN_ITERS:-10}
    NUM_TRAIN_EPOCHS=                # MUST be empty: megatron_args.py:934 would
                                     # overwrite train_iters with the epoch count
    OUTPUT_DIR=/data/autovision-cfs/autovision-notebook/algo4/tpz/ft/sft-vis-double-smoke-h200
    rm -rf "${OUTPUT_DIR}"
    SAVE_STEPS=1000000
    echo "[smoke] TRAIN_ITERS=${TRAIN_ITERS} OUTPUT_DIR=${OUTPUT_DIR} SAVE_STEPS=${SAVE_STEPS}"
fi

resume_args=()
FINETUNE=true
if [ "${RESUME}" = "1" ] && [ "${SMOKE}" != "1" ]; then
    [ -f "${OUTPUT_DIR}/latest_checkpointed_iteration.txt" ] || {
        echo "[error] RESUME=1 but no checkpoint tracker in ${OUTPUT_DIR}" >&2; exit 1; }
    _resume_iter=$(cat "${OUTPUT_DIR}/latest_checkpointed_iteration.txt")
    resume_args+=(--mcore_adapter "${OUTPUT_DIR}")
    FINETUNE=false
    echo "[resume] loading checkpoint-${_resume_iter} from ${OUTPUT_DIR}"
fi

# -----------------------------------------------------------------------------
# 8. Build optional flag lists
# -----------------------------------------------------------------------------
mtp_args=()
if [ "${MTP_NUM_LAYERS}" != "0" ]; then
    mtp_args+=(--mtp_num_layers "${MTP_NUM_LAYERS}" --mtp_loss_scaling_factor "${MTP_LOSS_SCALING_FACTOR}")
fi

recompute_extra_args=()
if [ "${RECOMPUTE_GRANULARITY}" = "full" ]; then
    recompute_extra_args+=(--recompute_method "${RECOMPUTE_METHOD}" --recompute_num_layers "${RECOMPUTE_NUM_LAYERS}")
fi

train_iters_args=()
if [ -n "${TRAIN_ITERS}" ]; then
    train_iters_args+=(--train_iters "${TRAIN_ITERS}")
fi

num_train_epochs_args=()
if [ -n "${NUM_TRAIN_EPOCHS}" ]; then
    num_train_epochs_args+=(--num_train_epochs "${NUM_TRAIN_EPOCHS}")
fi

module_lr_args=()
if [ -n "${VIT_LR}" ]; then
    module_lr_args+=(--vit_lr "${VIT_LR}")
fi
if [ -n "${ALIGNER_LR}" ]; then
    module_lr_args+=(--aligner_lr "${ALIGNER_LR}")
fi

modules_to_save_args=()
if [ -n "${MODULES_TO_SAVE}" ]; then
    modules_to_save_args+=(--modules_to_save "${MODULES_TO_SAVE}")
fi

plugin_args=()
if [ -n "${PLUGIN_PATH}" ]; then
    plugin_args+=(--external_plugins "${PLUGIN_PATH}")
fi

channel_loss_args=()
if [ "${ENABLE_CHANNEL_LOSS}" = "true" ]; then
    channel_loss_args+=(--enable_channel_loss true)
fi

# -----------------------------------------------------------------------------
# 9. Launch (swift megatron entrypoint; NPROC_PER_NODE drives torchrun)
# -----------------------------------------------------------------------------
NPROC_PER_NODE=${NUM_GPUS} \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
megatron sft \
    --model "${MODEL_PATH}" \
    --dataset "${TRAIN_FILES[@]}" \
    --split_dataset_ratio "${SPLIT_DATASET_RATIO}" \
    --load_from_cache_file "${LOAD_FROM_CACHE_FILE}" \
    --save_safetensors "${SAVE_SAFETENSORS}" \
    --tensor_model_parallel_size "${TP_SIZE}" \
    --pipeline_model_parallel_size "${PP_SIZE}" \
    --context_parallel_size "${CP_SIZE}" \
    --expert_model_parallel_size "${EP_SIZE}" \
    --sequence_parallel "${SEQUENCE_PARALLEL}" \
    --use_distributed_optimizer "${USE_DISTRIBUTED_OPTIMIZER}" \
    --recompute_granularity "${RECOMPUTE_GRANULARITY}" \
    ${recompute_extra_args[@]} \
    --attention_backend "${ATTENTION_BACKEND}" \
    --packing "${PACKING}" \
    --padding_free "${PADDING_FREE}" \
    --tuner_type "${TUNER_TYPE}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --merge_lora "${MERGE_LORA}" \
    --freeze_llm "${FREEZE_LLM}" \
    --freeze_vit "${FREEZE_VIT}" \
    --freeze_aligner "${FREEZE_ALIGNER}" \
    ${modules_to_save_args[@]} \
    ${module_lr_args[@]} \
    ${plugin_args[@]} \
    ${channel_loss_args[@]} \
    --micro_batch_size "${MICRO_BATCH_SIZE}" \
    --global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --max_length "${MAX_LENGTH}" \
    --lr "${LR}" \
    --min_lr "${MIN_LR}" \
    --lr_warmup_fraction "${LR_WARMUP_FRACTION}" \
    --lr_decay_style "${LR_DECAY_STYLE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --clip_grad "${CLIP_GRAD}" \
    "${num_train_epochs_args[@]}" \
    --cross_entropy_loss_fusion true \
    --finetune "${FINETUNE}" \
    --no_save_optim "${NO_SAVE_OPTIM}" \
    --no_save_rng "${NO_SAVE_RNG}" \
    --save_strategy "${SAVE_STRATEGY}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --add_version false \
    --output_dir "${OUTPUT_DIR}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --dataset_num_proc "${DATASET_NUM_PROC}" \
    --packing_num_proc "${PACKING_NUM_PROC}" \
    "${train_iters_args[@]}" "${resume_args[@]}" "$@"

# -----------------------------------------------------------------------------
# Notes
# -----------------------------------------------------------------------------
# 1. rank_pattern/alpha_pattern are NOT swift CLI args — the ONLY injection
#    channel is --external_plugins (verified: base_args.py:137-152 imports the
#    plugin at args-parsing time, before the trainer; the plugin wraps the
#    module-global LoraConfig used by prepare_adapter).
# 2. modules_to_save uses the MEGATRON-side path (visual.visual.merger), not
#    the HF-side 'model.visual.merger' — megatron model tree prefixes modules
#    with the extra 'visual.' segment (live-regex evidence). peft matches
#    modules_to_save by suffix, so the full path is exact.
# 3. aligner_lr applies to the merger's full-param weights via megatron's
#    _get_param_groups visual-prefix grouping (aligner split from ViT group).
# 4. MERGE_LORA=true doubles disk per save (adapter + merged weights, ~54GB
#    each on 27B) — watch the NFS headroom (98% full at script time).
# 5. First launch on a NEW dataset re-encodes all images (hours) and writes
#    the cache to NFS (MODELSCOPE_CACHE_HOME=.../logos-remote/sft/cache ->
#    .../hub/datasets/map_cache); later launches load it directly.
# 6. New-variant arrival (v4/v5 both landed this way): append the parquet path
#    to TRAIN_FILES (one line) — the plugin's variant->channel translation
#    covers any variant int automatically (variant=N -> 'VN'); SAVE_STEPS needs
#    recomputing for the new total.
# 7. S2 transition (plan §5): after picking the S1 checkpoint, merge & export
#    the full model, then start S2 with a FRESH LLM LoRA, rebuild optimizer/
#    scheduler, and do NOT pass --modules_to_save (aligner frozen in S2).
# =============================================================================