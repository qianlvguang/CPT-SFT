#!/usr/bin/env bash
# =============================================================================
# vis-tripple: S2 STAGE-2 KNOWLEDGE SFT of Qwen3.5-27B (dense VLM) — ms-swift +
# Megatron-Core, on top of the S1 vis-double recipe (same skeleton / H200 tuning).
# Target hardware: NVIDIA H200 (Hopper, sm90, 141 GB/card, 8 GPUs)
#
# TRIPPLE = S2 (external-plan 600k knowledge stage, master-approved):
#   start from the SELECTED S1 merged checkpoint, FREEZE the whole visual
#   stack, train a FRESH LLM LoRA only:
#        | module  | mode    | rank/α | LR    |
#        |---------|---------|--------|-------|
#        | ViT     | FROZEN  | —      | —     |  (freeze_vit=true; S1 patch B removed)
#        | aligner | FROZEN  | —      | —     |  (freeze_aligner=true + NO modules_to_save)
#        | LLM     | LoRA    | 64/128 | 2e-5  |  (global --lora_rank/--lora_alpha)
#   + per-prompt-channel loss logging (V1..V5, --enable_channel_loss):
#     the five S2 parquets carry a `variant` column (1..5); the plugin
#     translates variant -> 'V%d' channel per sample (LOGGING ONLY — zero
#     effect on the training loss or throughput).
#
# INPUT DATA (S2 600k mix — external-plan proportions; S1-unused pools):
#   V1  120k  fine-grained structured recognition   (v1 fresh pool, maid-1)
#   V2   90k  summary description & association     (v2 fresh pool, maid-1)
#   V3  150k  structured + summary + association    (maid-3, screened)
#   V4  180k  directed knowledge description        (maid-2 pilot, screened)
#   V5   60k  rule binary judgment                  (v5mini REPLAY: S1-seen
#             by design — never carve a val set out of it)
#   Same schema as the S1 ultra_x parquets: columns ['messages', 'images',
#   'enable_thinking', 'sample_id', 'cate_id', 'variant'].
#
# SINGLE-FILE DELIVERY: the variant->channel plugin is EMBEDDED below
# (section 1b, heredoc) and self-generated at launch next to this script as
# _vis_tripple_plugin.py — nothing else to upload/deploy. The plugin does:
#   patch A: RowPreprocessor.batched_preprocess wrap — variant -> 'V%d'
#     channel (per-sample column translation at dataset-encode time; full
#     verified chain: preprocess passthrough -> StdTemplateInputs.from_dict
#     picks 'channel' -> encode -> packing_row keeps per-sample list ->
#     collator emits flat list -> megatron trainer pops it ->
#     _compute_channel_loss cu_seqlens branch -> loss_V1..V5 metrics).
#   (S1's patch B — ViT LoRA rank_pattern injection — is REMOVED: with
#    freeze_vit=true the visual stack is out of the LoRA target set, so the
#    global --lora_rank/--lora_alpha lands on the LLM only.)
#
# S2 PARAMETER SET (external-plan values):
#   warmup 3%, cosine, weight_decay 0.01, lora_dropout 0.05, peak LR 2e-5,
#   floor 2e-6, max TWO epochs — checkpoint selection decides; SIX saves at
#   1/3-epoch marks (SAVE_STEPS = iters_per_epoch/3, fixed after SMOKE
#   reports the real iters_per_epoch; SAVE_TOTAL_LIMIT=6 keeps ALL six).
#   FRESH START: RESUME stays 0 — enter S2 from the S1 MERGED model with a
#   new LoRA + optimizer/scheduler, never via the S1 script's RESUME=1
#   (megatron --finetune=true restarts the finetune stage without restoring
#   iteration/optimizer/RNG state; plan: do NOT enter stage 2 via RESUME=1).
#
# GLOBAL BATCH NOTE: GBS=32 counts PACKED windows (MAX_LENGTH=8192), not raw
# samples — same convention as S1, so the stages stay comparable. S1 measured:
# 860k rows -> 2631 iters/epoch (~10.2 rows/window); S2 600k rows ->
# ~1840 iters/epoch ESTIMATE — SMOKE fixes the exact number.
#
# TP=4 NOTE (unchanged): 8 GPUs / TP=4 -> DP=2. kv_heads divide 4 cleanly;
# channel loss handles DP=2 natively (all_gather_object key-sync).
#
# MEMORY LEDGER (H200 141GB, TP=4): LIGHTER than S1 — no ViT LoRA, no
# merger full-param grads/optimizer (S1 carried merger fp32 optim ~3GB/card);
# activations unchanged (MAX_LENGTH=8192 + selective recompute).
#
# DISK LEDGER: MERGE_LORA=true saves a merged full model per checkpoint
# (~51GB x 6 = ~306GB; ckpt adapter dirs are ~1GB each). NFS had ~1.78TB
# free at S1 finish (2026-09-24) — still, the rotation grinder
# (save_total_limit=6) eats the oldest pair ~2min after every new save:
# rescue anything you care about DURING the run (battle scar: cpt-2560).
#
# SMOKE / RESUME (mechanisms battle-proven in the cpt_recovery campaign):
#   SMOKE=1 MODEL_PATH=<s1_merged_ckpt> bash vis_tripple.sh   # 10 iters
#   RESUME=1 nohup bash vis_tripple.sh > vis_tripple.log 2>&1 &
#   (SMOKE must OMIT --num_train_epochs: megatron_args.py:934 overwrites
#    train_iters whenever num_train_epochs is not None.)
#
# VERIFICATION CHECKLIST (grep the log after launch):
#   1. '[vis_tripple_plugin] variant->channel patch installed'  (patch A live)
#   2. NO 'rank_pattern' anywhere (patch B removed) — LoraConfig global only
#   3. trainable params: LLM LoRA ONLY — no ViT adapters, no merger
#      full-param, no vit_lr/aligner_lr param groups
#   4. 'loss_V1' .. 'loss_V5' metric keys  (channel loss, 5 channels)
#   5. sanity echo: MODULES_TO_SAVE/VIT_LR/ALIGNER_LR all empty
#
# Every knob is env-overridable, e.g.:
#   SMOKE=1 MODEL_PATH=... bash vis_tripple.sh
#   SAVE_STEPS=612 RESUME=1 nohup bash vis_tripple.sh > vis_tripple.log 2>&1 &
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
# S2 base = the SELECTED S1 merged checkpoint (master picks after the S1
# finish + knowledge eval; plan: 1-epoch + best knowledge metric). No
# default on purpose — a bare launch must fail loudly, not train the wrong
# base. Candidates: algo4/tpz/ft/sft-qwen3_5-27b-vis-double/checkpoint-*-merged
MODEL_PATH=${MODEL_PATH:-/data/autovision-cfs/autovision-notebook/algo4/tpz/ft/sft-qwen3_5-27b-vis-double-L/checkpoint-4608-merged}
if [ -z "${MODEL_PATH}" ]; then
    echo "[error] S2 needs the selected S1 merged ckpt:" >&2
    echo "  MODEL_PATH=<...>/sft-qwen3_5-27b-vis-double/checkpoint-XXXX-merged bash $0" >&2
    echo "  available merged ckpts:" >&2
    ls -d /data/autovision-cfs/autovision-notebook/algo4/tpz/ft/sft-qwen3_5-27b-vis-double/*-merged 2>/dev/null >&2 || true
    exit 1
fi
[ -f "${MODEL_PATH}/config.json" ] || { echo "[error] missing ${MODEL_PATH}/config.json" >&2; exit 1; }

TEST_DIR=/data/autovision-cfs/autovision-notebook/algo4/tpz/test
# Each parquet: messages/images/enable_thinking/sample_id/cate_id/variant.
# The variant column (1..5) is translated to the channel column ('V%d') by
# the plugin at encode time — the parquets stay untouched on disk.
# S2 600k recipe (external-plan proportions; built from S1-unused pools,
# manifest at TEST_DIR/train_caption_s2_manifest.json):
TRAIN_FILES=(
    "${TEST_DIR}/train_caption_ultra_x_v1medium.parquet"  # 120,000 rows (v1 fresh pool)
    "${TEST_DIR}/train_caption_ultra_x_v2medium.parquet"  #  90,000 rows (v2 fresh pool)
    "${TEST_DIR}/train_caption_ultra_x_v3medium.parquet"  # 150,000 rows (maid-3, screened)
    "${TEST_DIR}/train_caption_ultra_x_v4medium.parquet"  # 180,000 rows (maid-2 pilot)
    "${TEST_DIR}/train_rough_templete.parquet"  #  60,000 rows (v5mini replay)
)
for _f in "${TRAIN_FILES[@]}"; do
    [ -f "${_f}" ] || { echo "[error] TRAIN file not found: ${_f}" >&2; exit 1; }
done
SPLIT_DATASET_RATIO=${SPLIT_DATASET_RATIO:-0}

# -----------------------------------------------------------------------------
# 1b. Self-generate the vis_tripple plugin (variant->channel only)
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_PATH=${PLUGIN_PATH:-${SCRIPT_DIR}/_vis_tripple_plugin.py}

cat > "${PLUGIN_PATH}" <<'VIS_TRIPPLE_PLUGIN_EOF'
r"""vis_tripple plugin (auto-generated by vis_tripple.sh).

Loaded at ARGS-PARSING time via --external_plugins (arguments/base_args/
base_args.py:137-152 imports the plugin before the trainer is built).

patch A: variant -> channel translation.
    The S2 parquets carry a `variant` column (int 1..5) but swift's
    channel loss expects a per-sample `channel` column (string, whitelisted
    in preprocessor standard_keys). This patch wraps RowPreprocessor.
    batched_preprocess — the single entry point that receives the RAW columns
    (before remove_useless_columns drops non-standard ones) and returns the
    encoded rows — and translates variant to 'V%d' in place. The channel
    column then flows through swift's official path untouched:
    preprocess passthrough -> StdTemplateInputs.from_dict (template_inputs.py
    :62 picks 'channel') -> encode (template/base.py:648) -> packing_row keeps
    the per-sample list (base.py:688) -> collator emits the flat list
    (base.py:1874) -> megatron trainer pops 'channel' (trainer.py:110) ->
    _compute_channel_loss cu_seqlens branch -> loss_V1..loss_V5 metrics.

S2 note: S1's patch B (ViT rank_pattern injection r16/a32) is intentionally
    REMOVED — the whole visual stack (ViT + aligner) is frozen in S2
    (freeze_vit + freeze_aligner + NO modules_to_save), so the global
    --lora_rank/--lora_alpha applies to the LLM only.
"""

# --- patch A: variant -> channel -------------------------------------------
from swift.dataset.preprocessor.core import RowPreprocessor as _RowPreprocessor

_orig_batched_preprocess = _RowPreprocessor.batched_preprocess
_vdc_translated_once = [False]


def _vis_tripple_batched_preprocess(self, batched_row, *, strict,
                                    ignore_max_length_error):
    batched_row = dict(batched_row)
    if 'channel' not in batched_row and 'variant' in batched_row:
        try:
            ch = ['V%d' % v if v is not None else None
                  for v in batched_row['variant']]
            batched_row['channel'] = ch
            if not _vdc_translated_once[0]:
                _vdc_translated_once[0] = True
                print('[vis_tripple_plugin] first variant->channel batch: '
                      f'{len(ch)} rows, e.g. {ch[:8]}')
        except Exception as e:
            print(f'[vis_tripple_plugin] variant->channel translation failed: {e!r}')
    return _orig_batched_preprocess(self, batched_row, strict=strict,
                                    ignore_max_length_error=ignore_max_length_error)


if not getattr(getattr(_RowPreprocessor, 'batched_preprocess', None),
               '_vis_tripple_patched', False):
    _vis_tripple_batched_preprocess._vis_tripple_patched = True
    _RowPreprocessor.batched_preprocess = _vis_tripple_batched_preprocess
    print('[vis_tripple_plugin] variant->channel patch installed '
          '(RowPreprocessor.batched_preprocess)')

# --- patch B: REMOVED in S2 --------------------------------------------------
# S1 injected ViT rank_pattern/alpha_pattern here (r16/a32 via a LoraConfig
# wrap in swift.megatron.utils.utils). S2 freezes the whole visual stack —
# no ViT LoRA, nothing to inject.
VIS_TRIPPLE_PLUGIN_EOF
[ -f "${PLUGIN_PATH}" ] || { echo "[error] failed to generate ${PLUGIN_PATH}" >&2; exit 1; }
echo "[vis_tripple] plugin generated: ${PLUGIN_PATH}"

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

LR=${LR:-2e-5}                                # LLM LoRA peak LR (S2 plan)
MIN_LR=${MIN_LR:-2e-6}                        # cosine floor = LR/10 (S2 plan)
LR_WARMUP_FRACTION=${LR_WARMUP_FRACTION:-0.03}  # plan: 3% per stage
LR_DECAY_STYLE=${LR_DECAY_STYLE:-cosine}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}            # plan: 0.01
CLIP_GRAD=${CLIP_GRAD:-1.0}

# Per-module LRs (megatron-native): EMPTY in S2 — the visual stack is frozen
# (ViT LoRA and merger full-param are S1-only; empty = no vit_lr/aligner_lr
# flags -> no extra param groups).
VIT_LR=${VIT_LR:-}                            # S2: unset (ViT frozen)
ALIGNER_LR=${ALIGNER_LR:-}                    # S2: unset (aligner frozen)

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
# 4. Tuner config (S2: ViT+aligner FROZEN, fresh LLM LoRA r64/a128)
# -----------------------------------------------------------------------------
TUNER_TYPE=${TUNER_TYPE:-lora}
LORA_RANK=${LORA_RANK:-64}                    # LLM-only in S2 (visual excluded by freeze flags)
LORA_ALPHA=${LORA_ALPHA:-128}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}            # plan: 0.05 (also swift's default)
MERGE_LORA=${MERGE_LORA:-true}                # RTX-LoRA-proven path (adapter-only
                                              # save on VLM has error history)
FREEZE_LLM=${FREEZE_LLM:-false}               # MUST be false (LoRA semantics)
FREEZE_VIT=${FREEZE_VIT:-true}                # S2: ViT FROZEN — keep the
                                              # S1-learned visual weights
FREEZE_ALIGNER=${FREEZE_ALIGNER:-true}        # EXCLUDES merger from LoRA targets;
                                              # S2: frozen outright (NO modules_to_save)
# MODULES_TO_SAVE — MUST STAY EMPTY in S2 (merger full-param is S1-only).
#   NOTE THE MISSING COLON in the default below: `${VAR:-...}` refills the
#   merger path even when the launcher exports an EMPTY string (external-
#   plan trap, S1 line 319). `${VAR-}` only substitutes when UNSET, so
#   MODULES_TO_SAVE="" survives and --modules_to_save is omitted at launch.
MODULES_TO_SAVE=${MODULES_TO_SAVE-}           # S2: empty (aligner frozen)

ENABLE_CHANNEL_LOSS=${ENABLE_CHANNEL_LOSS:-true}  # logging-only, per-channel metrics (V1..V5)

# -----------------------------------------------------------------------------
# 5. MTP head (S1 default OFF to keep the stage clean)
# -----------------------------------------------------------------------------
MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-0}
MTP_LOSS_SCALING_FACTOR=${MTP_LOSS_SCALING_FACTOR:-0.1}

# -----------------------------------------------------------------------------
# 6. Output & checkpointing
# -----------------------------------------------------------------------------
OUTPUT_DIR=${OUTPUT_DIR:-/data/autovision-cfs/autovision-notebook/algo4/tpz/ft/sft-qwen3_5-27b-vis-tripple-L}
SAVE_STRATEGY=${SAVE_STRATEGY:-steps}
# SAVE_STEPS = one-third of an epoch (SIX checkpoints at 1/3, 2/3, 1, 4/3,
#   5/3, 2 epochs = every ~16.7% of the 2-epoch run; save_strategy=epoch only
#   supports INTEGER epochs, so every non-integer point REQUIRES steps mode):
#   SAVE_STEPS = iters_per_epoch / 3, where iters_per_epoch is printed by the
#   dataloader build at SMOKE — fill the number below after SMOKE, BEFORE the
#   live launch. Estimate from S1 measured (860k rows -> 2631 iters/epoch
#   -> ~10.2 rows/window @ GBS=32): 600k rows -> ~1836 iters/epoch ->
#   SAVE_STEPS ~= 612.
#   Disk note: 6 merged ckpts x ~51GB = ~306GB (adapter dirs ~1GB each) —
#   NFS had ~1.78TB free at S1 finish; still rescue eval samples DURING the
#   run if disk pressure rises.
SAVE_STEPS=${SAVE_STEPS:-612}                 # ESTIMATE — set to
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
    OUTPUT_DIR=/data/autovision-cfs/autovision-notebook/algo4/tpz/ft/sft-vis-tripple-smoke-h200
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

# S2 sanity preamble — the first three MUST be empty for the main experiment
echo "[vis_tripple] MODULES_TO_SAVE='${MODULES_TO_SAVE}' VIT_LR='${VIT_LR}' ALIGNER_LR='${ALIGNER_LR}'"
echo "[vis_tripple] FREEZE_VIT=${FREEZE_VIT} FREEZE_ALIGNER=${FREEZE_ALIGNER} LORA_RANK=${LORA_RANK} LR=${LR} MIN_LR=${MIN_LR}"
echo "[vis_tripple] MODEL_PATH=${MODEL_PATH}"
echo "[vis_tripple] OUTPUT_DIR=${OUTPUT_DIR} SAVE_STEPS=${SAVE_STEPS} RESUME=${RESUME} SMOKE=${SMOKE}"

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
# 1. S2 entry: MODEL_PATH = S1 MERGED ckpt; fresh LLM LoRA + fresh optimizer
#    (RESUME=0, --finetune true). Never enter S2 via the S1 run's RESUME=1 —
#    that restores S1's iteration/optimizer/RNG state and keeps the S1 LoRA.
# 2. NO modules_to_save in S2: `${MODULES_TO_SAVE-}` (no colon) lets an
#    explicitly-exported EMPTY string stay empty. The S1 form
#    `${MODULES_TO_SAVE:-visual.visual.merger}` would refill the merger path
#    and silently put the aligner back into full-param training.
# 3. No vit_lr/aligner_lr either (frozen modules carry no param groups);
#    grep the launch log for 'vit_lr'/'aligner_lr' — expect NOTHING.
# 4. rank_pattern/alpha_pattern are NOT swift CLI args and NOT injected here
#    (S1's plugin patch B removed) — the global --lora_rank/--lora_alpha
#    applies to the LLM only because freeze_vit/freeze_aligner exclude the
#    visual stack from the LoRA target set (target_modules regex still
#    excludes merger via the freeze_aligner negative-lookahead).
# 5. MERGE_LORA=true: merged full weights ~51GB per save (adapter ~1GB) —
#    watch NFS headroom; rescue eval samples DURING the run.
# 6. First launch on the S2 parquets re-encodes all images (hours) and
#    PERSISTS the cache on NFS (MODELSCOPE_CACHE_HOME=.../logos-remote/sft/
#    cache); later launches/pods start fast.
# 7. V5 is S1-seen REPLAY by design — never split it as a validation set
#    (external plan). In-run val stays OFF (split_dataset_ratio=0) as in S1;
#    evaluation stays external (eval_* scripts) unless master orders a
#    fixed val_dataset.
# 8. Loss weights: file sizes lie — per-task loss weight = per-variant
#    assistant-token share of the packed windows. loss_V1..V5 are per-channel
#    LOGGING; the S2 manifest carries per-variant assistant-token estimates
#    for the real accounting (plan: count assistant tokens that actually
#    enter the loss).
# =============================================================================