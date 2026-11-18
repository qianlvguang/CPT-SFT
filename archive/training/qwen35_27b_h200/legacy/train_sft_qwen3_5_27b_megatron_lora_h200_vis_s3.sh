#!/usr/bin/env bash
# =============================================================================
# vis-s3: S3 STAGE-3 FINAL-DATA SFT of Qwen3.5-27B (dense VLM) — ms-swift +
# Megatron-Core, on top of the S2 vis-tripple recipe (same skeleton / H200
# tuning). Target hardware: NVIDIA H200 (Hopper, sm90, 141 GB/card, 8 GPUs)
#
# S3 = third stage on the ~1.1M "final" pools (zero overlap with mini/medium):
#   start from the SELECTED S2 / S2-L merged checkpoint, FREEZE the visual
#   stack, train a FRESH LLM LoRA at a GENTLER LR than S2:
#        | module  | mode    | rank/α | LR           |
#        |---------|---------|--------|--------------|
#        | ViT     | FROZEN  | —      | —            |  (freeze_vit=true)
#        | aligner | FROZEN  | —      | —            |  (default; MERGER_LORA=1 -> LoRA 32/64 @ 5e-6)
#        | LLM     | LoRA    | 64/128 | 1e-5 -> 1e-6 |  (S2 was 2e-5 -> 2e-6)
#   Why gentler: every stage merges its LoRA into the base; S2's correlation
#   to its parent fell 0.94 -> 0.86 over training (drift) and S2 saturated at
#   ~1 epoch. S3 stacks a third merged LoRA, so it steps more softly.
#
# DELTA vs vis-tripple (S2 / S2-L):
#   - MODEL_PATH: NO default — pick the S3 parent explicitly (S2-L is the
#     preferred lineage: it beats S2 at matched ckpts). A bare launch fails.
#   - data: v1..v4 final (1,097,530 rows) + V5 replay (60k, same file as S2)
#     + optional EXTRA_TRAIN_FILES (replay slices, e.g. V1 mini/medium).
#   - LR 1e-5 / MIN_LR 1e-6 (was 2e-5 / 2e-6).
#   - preflight: row counts + schema + enable_thinking check on every parquet,
#     and SAVE_STEPS auto-estimated from the row total (override after SMOKE).
#   - optional ablation arm MERGER_LORA=1: merger gets LoRA r32/a64 @ 5e-6
#     (S1-L showed the merger-LoRA route is what fixed the misleading
#     cluster); ViT stays frozen. Output dir gets a '-M' suffix.
#
# INPUT DATA (S3 ~1.16M mix; TEST_DIR, schema = medium versions):
#   V1   40,113  fine-grained structured recognition   (v1final)
#   V2  230,100  summary description & association     (v2final)
#   V3  423,370  structured + summary + association    (v3final)
#   V4  403,947  directed knowledge description        (v4final, enable_thinking=False)
#   V5   60,000  rule binary judgment REPLAY           (S1-seen by design —
#                never carve a val set out of it)
#   NOTE: raw row mix is V1-light (3.5%) and V3/V4-heavy vs S1/S2. If V1
#   tasks regress, add a replay slice via EXTRA_TRAIN_FILES.
#
# SINGLE-FILE DELIVERY: the plugin is EMBEDDED below (section 1c, heredoc)
# and self-generated at launch next to this script as _vis_s3_plugin.py:
#   patch A: variant -> 'V%d' channel (logging only, same as S1/S2).
#   patch B: ONLY when MERGER_LORA=1 — rank_pattern {merger: 32} / alpha
#     {merger: 64} (same LoraConfig wrap as the S1-L plugin, merger key only,
#     since the ViT is frozen and out of the LoRA target set).
#
# SAVE PLAN: SIX saves at 1/3-epoch marks over 2 epochs (SAVE_TOTAL_LIMIT=6
# keeps all six). ~1.16M rows @ ~10.18 rows/window (S2 measured 600k ->
# 58,936 windows) @ GBS=32 -> ~3,550 iters/epoch -> SAVE_STEPS ~1,180 (the
# preflight prints the estimate; SMOKE prints the real iters_per_epoch).
# 2 epochs ~7,100 iters ~26h @ 13.3 s/it. For a 1-epoch run (S2 saturated
# at ~1 epoch): NUM_TRAIN_EPOCHS=1 — the preflight then saves every 1/6 epoch.
#
# DISK LEDGER: 6 merged ckpts x ~51GB = ~306GB + adapters. The rotation
# grinder is idle at exactly 6 saves, but rescue anything you evaluate
# DURING the run if you ever raise the save count.
#
# SMOKE / RESUME:
#   SMOKE=1 MODEL_PATH=<s2_merged_ckpt> bash vis_s3.sh          # 10 iters
#   MODEL_PATH=<...> nohup bash vis_s3.sh > vis_s3.log 2>&1 &
#   MODEL_PATH=<...> RESUME=1 nohup bash vis_s3.sh > vis_s3.log 2>&1 &
#   (SMOKE must OMIT --num_train_epochs: megatron_args.py:934 overwrites
#    train_iters whenever num_train_epochs is not None.)
#   FIRST launch encodes ~1.16M rows (hours; cache persists on NFS).
#
# VERIFICATION CHECKLIST (grep the log after launch):
#   1. '[vis_s3_plugin] variant->channel patch installed'  (patch A live)
#   2. MERGER_LORA=0: NO 'rank_pattern' anywhere; trainable = LLM LoRA only
#      MERGER_LORA=1: '[vis_s3_plugin] rank_pattern={'visual\\.merger\\..*': 32}'
#      AND merger adapters actually created (target_modules regex must NOT
#      exclude visual.merger — freeze_vit=true + freeze_aligner=false is a
#      NEW combination in this family: VERIFY AT SMOKE before the live run)
#   3. 'loss_V1' .. 'loss_V5' metric keys
#   4. sanity echo: MODULES_TO_SAVE/VIT_LR empty; ALIGNER_LR empty unless
#      MERGER_LORA=1
#   5. '[vis_s3 preflight]' total rows == 1,157,530 (default file set)
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

# Dataset preprocess cache pinned on NFS (pod-rebuild safe; see S1/S2 notes)
export MODELSCOPE_CACHE_HOME=${MODELSCOPE_CACHE_HOME:-/data/autovision-cfs/autovision-notebook/logos-remote/sft/cache}

FT_ROOT=/data/autovision-cfs/autovision-notebook/algo4/tpz/ft

# -----------------------------------------------------------------------------
# 1a. Model (NO default — the S3 parent must be chosen explicitly)
# -----------------------------------------------------------------------------
# Candidates: best S2-L ckpt (preferred lineage) or S2-3060 (S2 champion).
MODEL_PATH=${MODEL_PATH:-}
if [ -z "${MODEL_PATH}" ]; then
    echo "[error] S3 needs the selected S2 / S2-L merged ckpt:" >&2
    echo "  MODEL_PATH=${FT_ROOT}/sft-qwen3_5-27b-vis-tripple-L/checkpoint-XXXX-merged bash $0" >&2
    echo "  available merged ckpts:" >&2
    ls -d "${FT_ROOT}"/sft-qwen3_5-27b-vis-tripple*/*-merged 2>/dev/null >&2 || true
    exit 1
fi
[ -f "${MODEL_PATH}/config.json" ] || { echo "[error] missing ${MODEL_PATH}/config.json" >&2; exit 1; }

# -----------------------------------------------------------------------------
# 1b. Data (v1..v4 final + V5 replay + optional extra replay files)
# -----------------------------------------------------------------------------
TEST_DIR=/data/autovision-cfs/autovision-notebook/algo4/tpz/test
# V5 replay: same file S2/S2-L trained on. Override with
#   V5_FILE=${TEST_DIR}/train_caption_ultra_x_v5mini.parquet
# to replay the exact S1 v5mini instead (6 cols, no variant column -> its
# rows land outside loss_V5; training loss unaffected).
V5_FILE=${V5_FILE:-${TEST_DIR}/train_rough_templete.parquet}
TRAIN_FILES=(
    "${TEST_DIR}/train_caption_ultra_x_v1final.parquet"   #  40,113 rows
    "${TEST_DIR}/train_caption_ultra_x_v2final.parquet"   # 230,100 rows
    "${TEST_DIR}/train_caption_ultra_x_v3final.parquet"   # 423,370 rows
    "${TEST_DIR}/train_caption_ultra_x_v4final.parquet"   # 403,947 rows
    "${V5_FILE}"                                          #  60,000 rows (V5 replay)
)
# Optional replay slices, space-separated absolute paths, e.g.
#   EXTRA_TRAIN_FILES="${TEST_DIR}/train_caption_ultra_x_v1medium.parquet"
EXTRA_TRAIN_FILES=${EXTRA_TRAIN_FILES:-}
if [ -n "${EXTRA_TRAIN_FILES}" ]; then
    read -r -a _extra <<< "${EXTRA_TRAIN_FILES}"
    TRAIN_FILES+=("${_extra[@]}")
fi
for _f in "${TRAIN_FILES[@]}"; do
    [ -f "${_f}" ] || { echo "[error] TRAIN file not found: ${_f}" >&2; exit 1; }
done
SPLIT_DATASET_RATIO=${SPLIT_DATASET_RATIO:-0}   # V5 is replay: never split it

# -----------------------------------------------------------------------------
# 1c. Arm selection + self-generated plugin
# -----------------------------------------------------------------------------
MERGER_LORA=${MERGER_LORA:-0}          # 1 = ablation arm: merger LoRA r32/a64
MERGER_RANK=${MERGER_RANK:-32}
MERGER_ALPHA=${MERGER_ALPHA:-64}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_PATH=${PLUGIN_PATH:-${SCRIPT_DIR}/_vis_s3_plugin.py}

cat > "${PLUGIN_PATH}" <<'VIS_S3_PLUGIN_EOF'
r"""vis_s3 plugin (auto-generated by train_sft_qwen3_5_27b_megatron_lora_h200_vis_s3.sh).

Loaded at ARGS-PARSING time via --external_plugins.

patch A: variant -> channel translation (identical to the S2 plugin):
    wraps RowPreprocessor.batched_preprocess and writes 'V%d' into the
    per-sample `channel` column -> loss_V1..loss_V5 metrics (LOGGING ONLY).

patch B: ONLY when env VIS_S3_MERGER_LORA=1 — LoraConfig wrap injecting
    rank_pattern/alpha_pattern for the merger (same mechanism as the S1-L
    plugin). The ViT is frozen in S3, so only the merger key is needed; LLM
    keys match nothing -> global r64/a128 fallback.
"""
import os

# --- patch A: variant -> channel -------------------------------------------
from swift.dataset.preprocessor.core import RowPreprocessor as _RowPreprocessor

_orig_batched_preprocess = _RowPreprocessor.batched_preprocess
_vdc_translated_once = [False]


def _vis_s3_batched_preprocess(self, batched_row, *, strict,
                               ignore_max_length_error):
    batched_row = dict(batched_row)
    if 'channel' not in batched_row and 'variant' in batched_row:
        try:
            ch = ['V%d' % v if v is not None else None
                  for v in batched_row['variant']]
            batched_row['channel'] = ch
            if not _vdc_translated_once[0]:
                _vdc_translated_once[0] = True
                print('[vis_s3_plugin] first variant->channel batch: '
                      f'{len(ch)} rows, e.g. {ch[:8]}')
        except Exception as e:
            print(f'[vis_s3_plugin] variant->channel translation failed: {e!r}')
    return _orig_batched_preprocess(self, batched_row, strict=strict,
                                    ignore_max_length_error=ignore_max_length_error)


if not getattr(getattr(_RowPreprocessor, 'batched_preprocess', None),
               '_vis_s3_patched', False):
    _vis_s3_batched_preprocess._vis_s3_patched = True
    _RowPreprocessor.batched_preprocess = _vis_s3_batched_preprocess
    print('[vis_s3_plugin] variant->channel patch installed '
          '(RowPreprocessor.batched_preprocess)')

# --- patch B: merger LoRA rank_pattern (MERGER_LORA=1 arm only) --------------
if os.environ.get('VIS_S3_MERGER_LORA', '0') == '1':
    from swift.megatron.utils import utils as _swift_megatron_utils

    _MERGER_PATTERN_KEY = r'visual\.merger\..*'
    _MERGER_RANK = int(os.environ.get('VIS_S3_MERGER_RANK', '32'))
    _MERGER_ALPHA = int(os.environ.get('VIS_S3_MERGER_ALPHA', '64'))
    _OrigLoraConfig = _swift_megatron_utils.LoraConfig

    def _vis_s3_lora_config(*args, **kwargs):
        kwargs.setdefault('rank_pattern', {_MERGER_PATTERN_KEY: _MERGER_RANK})
        kwargs.setdefault('alpha_pattern', {_MERGER_PATTERN_KEY: _MERGER_ALPHA})
        cfg = _OrigLoraConfig(*args, **kwargs)
        print(f'[vis_s3_plugin] rank_pattern={dict(cfg.rank_pattern)} '
              f'alpha_pattern={dict(cfg.alpha_pattern)} '
              f'(global r={cfg.r} alpha={cfg.lora_alpha})')
        return cfg

    if not getattr(getattr(_swift_megatron_utils, 'LoraConfig', None),
                   '_vis_s3_patched', False):
        _vis_s3_lora_config._vis_s3_patched = True
        _swift_megatron_utils.LoraConfig = _vis_s3_lora_config
        print('[vis_s3_plugin] merger LoRA rank_pattern patch installed')
VIS_S3_PLUGIN_EOF
[ -f "${PLUGIN_PATH}" ] || { echo "[error] failed to generate ${PLUGIN_PATH}" >&2; exit 1; }
export VIS_S3_MERGER_LORA=${MERGER_LORA}
export VIS_S3_MERGER_RANK=${MERGER_RANK}
export VIS_S3_MERGER_ALPHA=${MERGER_ALPHA}
echo "[vis_s3] plugin generated: ${PLUGIN_PATH} (MERGER_LORA=${MERGER_LORA})"

# -----------------------------------------------------------------------------
# 2. Parallelism (TP * PP * DP * CP * EP == NUM_GPUS; H200 8x141GB)
# -----------------------------------------------------------------------------
NUM_GPUS=${NUM_GPUS:-8}
TP_SIZE=${TP_SIZE:-4}             # TP=4 -> DP=2 (S1/S2-proven)
PP_SIZE=${PP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}             # 27B dense -> EP=1
SEQUENCE_PARALLEL=${SEQUENCE_PARALLEL:-true}
USE_DISTRIBUTED_OPTIMIZER=${USE_DISTRIBUTED_OPTIMIZER:-false}

# -----------------------------------------------------------------------------
# 3. Training hyper-params (S3: gentler LR than S2)
# -----------------------------------------------------------------------------
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}    # packed windows (family convention)
MAX_LENGTH=${MAX_LENGTH:-8192}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}       # S2 saturated at ~1 ep; set 1
                                               # for a shorter/cheaper run

LR=${LR:-1e-5}                                # S3 peak (S2: 2e-5)
MIN_LR=${MIN_LR:-1e-6}                        # S3 floor = LR/10 (S2: 2e-6)
LR_WARMUP_FRACTION=${LR_WARMUP_FRACTION:-0.03}
LR_DECAY_STYLE=${LR_DECAY_STYLE:-cosine}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
CLIP_GRAD=${CLIP_GRAD:-1.0}

VIT_LR=${VIT_LR:-}                            # ViT frozen in every S3 arm
if [ "${MERGER_LORA}" = "1" ]; then
    ALIGNER_LR=${ALIGNER_LR:-5e-6}            # merger LoRA arm: half of LLM LR
else
    ALIGNER_LR=${ALIGNER_LR:-}                # main arm: aligner frozen
fi

RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-selective}
RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-uniform}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flash}

PACKING=${PACKING:-true}
PADDING_FREE=${PADDING_FREE:-false}           # packing=true forces padding_free internally

# -----------------------------------------------------------------------------
# 4. Tuner config (fresh LLM LoRA r64/a128; visual stack frozen by default)
# -----------------------------------------------------------------------------
TUNER_TYPE=${TUNER_TYPE:-lora}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
MERGE_LORA=${MERGE_LORA:-true}
FREEZE_LLM=${FREEZE_LLM:-false}               # MUST be false (LoRA semantics)
FREEZE_VIT=${FREEZE_VIT:-true}
if [ "${MERGER_LORA}" = "1" ]; then
    FREEZE_ALIGNER=${FREEZE_ALIGNER:-false}   # merger stays a LoRA target
else
    FREEZE_ALIGNER=${FREEZE_ALIGNER:-true}
fi
MODULES_TO_SAVE=${MODULES_TO_SAVE-}           # NO colon: explicit "" stays empty

ENABLE_CHANNEL_LOSS=${ENABLE_CHANNEL_LOSS:-true}

# -----------------------------------------------------------------------------
# 5. MTP head (OFF, as in S1/S2)
# -----------------------------------------------------------------------------
MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-0}
MTP_LOSS_SCALING_FACTOR=${MTP_LOSS_SCALING_FACTOR:-0.1}

# -----------------------------------------------------------------------------
# 6. Output, preflight & checkpointing
# -----------------------------------------------------------------------------
if [ "${MERGER_LORA}" = "1" ]; then
    OUTPUT_DIR=${OUTPUT_DIR:-${FT_ROOT}/sft-qwen3_5-27b-vis-s3-M}
else
    OUTPUT_DIR=${OUTPUT_DIR:-${FT_ROOT}/sft-qwen3_5-27b-vis-s3}
fi

# Preflight: parquet metadata only (no image decode) — row counts, schema,
# enable_thinking values; estimates iters/epoch from S2's measured density.
ROWS_PER_WINDOW=${ROWS_PER_WINDOW:-10.18}     # S2: 600,000 rows -> 58,936 windows
_est_save_steps=$(python3 - "${GLOBAL_BATCH_SIZE}" "${ROWS_PER_WINDOW}" \
                  "${NUM_TRAIN_EPOCHS:-1}" "${TRAIN_FILES[@]}" <<'PREFLIGHT_EOF'
import math, sys
import pyarrow.parquet as pq

gbs, rpw, epochs = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
need = {'messages', 'images', 'sample_id', 'cate_id'}
total = 0
for f in sys.argv[4:]:
    pf = pq.ParquetFile(f)
    cols = set(pf.schema_arrow.names)
    n = pf.metadata.num_rows
    total += n
    miss = need - cols
    think = '-'
    if 'enable_thinking' in cols:
        think = sorted(map(str, set(pq.read_table(f, columns=['enable_thinking'])
                                    .column(0).to_pylist())))
    print(f"[vis_s3 preflight] {f.rsplit('/', 1)[-1]}: rows={n:,} "
          f"cols={len(cols)} variant={'variant' in cols} "
          f"enable_thinking={think}" + (f" MISSING={sorted(miss)}" if miss else ''),
          file=sys.stderr)
    if miss:
        sys.exit(f'[vis_s3 preflight] schema error in {f}')
ipe = math.ceil(total / rpw / gbs)
# six saves over the whole run: 1/3 epoch for 2 ep, 1/6 epoch for 1 ep
save = max(1, round(ipe * epochs / 6))
print(f'[vis_s3 preflight] total rows={total:,} -> ~{ipe} iters/epoch '
      f'(est. @ {rpw} rows/window, GBS={gbs}) -> SAVE_STEPS~{save}',
      file=sys.stderr)
print(save)
PREFLIGHT_EOF
)
SAVE_STRATEGY=${SAVE_STRATEGY:-steps}
SAVE_STEPS=${SAVE_STEPS:-${_est_save_steps}}  # ESTIMATE — re-set to the real
                                               # iters_per_epoch/3 after SMOKE
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-6}
SAVE_SAFETENSORS=${SAVE_SAFETENSORS:-true}
NO_SAVE_OPTIM=${NO_SAVE_OPTIM:-true}
NO_SAVE_RNG=${NO_SAVE_RNG:-true}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-32}
DATASET_NUM_PROC=${DATASET_NUM_PROC:-16}
PACKING_NUM_PROC=${PACKING_NUM_PROC:-16}
LOAD_FROM_CACHE_FILE=${LOAD_FROM_CACHE_FILE:-true}

# -----------------------------------------------------------------------------
# 7. SMOKE / RESUME modes
# -----------------------------------------------------------------------------
SMOKE=${SMOKE:-0}
RESUME=${RESUME:-0}
TRAIN_ITERS=${TRAIN_ITERS:-}

if [ "${SMOKE}" = "1" ]; then
    TRAIN_ITERS=${TRAIN_ITERS:-10}
    NUM_TRAIN_EPOCHS=                # MUST be empty (megatron_args.py:934)
    OUTPUT_DIR=${FT_ROOT}/sft-vis-s3-smoke-h200
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

echo "[vis_s3] MODULES_TO_SAVE='${MODULES_TO_SAVE}' VIT_LR='${VIT_LR}' ALIGNER_LR='${ALIGNER_LR}' MERGER_LORA=${MERGER_LORA}"
echo "[vis_s3] FREEZE_VIT=${FREEZE_VIT} FREEZE_ALIGNER=${FREEZE_ALIGNER} LORA_RANK=${LORA_RANK} LR=${LR} MIN_LR=${MIN_LR} EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "[vis_s3] MODEL_PATH=${MODEL_PATH}"
echo "[vis_s3] OUTPUT_DIR=${OUTPUT_DIR} SAVE_STEPS=${SAVE_STEPS} RESUME=${RESUME} SMOKE=${SMOKE}"

# -----------------------------------------------------------------------------
# 9. Launch
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
    ${mtp_args[@]} \
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
# 1. S3 entry: MODEL_PATH = S2/S2-L MERGED ckpt; fresh LLM LoRA + optimizer
#    (RESUME=0, --finetune true). Never enter S3 via the S2 run's RESUME=1.
# 2. v4final: (sample_id, cate_id, image) is the unique row key — one
#    (task, sid) can carry several images. Never dedup by (task, sid).
# 3. Selection: 5658 subset + FULL 9450 side by side; guard per-cluster
#    regressions (S2 lost inconsistent-quantity -30pp, PillOrganizer, PPE).
# 4. MERGER_LORA=1 is an ablation arm, not the main run: freeze_vit=true +
#    freeze_aligner=false has not been launched in this family before — at
#    SMOKE confirm the merger adapters exist and the aligner_lr group shows.
# 5. The mtp_args list was built but never passed in S1/S2; it is wired
#    here (still a no-op at the default MTP_NUM_LAYERS=0).
# =============================================================================
