#!/usr/bin/env bash
# =============================================================================
# pipeline S1 -> S2 -> S3: three-stage LoRA SFT of Qwen3.5-27B (dense VLM) in
# ONE launch — ms-swift + Megatron-Core, same skeleton / H200 tuning as the
# vis_double_L / vis_tripple / vis_s3 scripts. Every stage knob is an env var,
# so sweep_pipeline.sh can launch many variants of this one script.
# Target hardware: NVIDIA H200 (Hopper, sm90, 141 GB/card, 8 GPUs)
#
# WHAT IT DOES (defaults):
#   base Qwen3.5-27B
#     └─ S1  (S1-L recipe)  mini   v1..v4  -> merged model
#         └─ S2 (tripple)   medium v1..v4  -> merged model
#             └─ S3 (s3)    final  v1..v4  -> merged model
#   - NO V5 by default: channels V1..V4 only.
#   - ONE epoch per stage, NO intermediate checkpoints: SAVE_STEPS is far
#     beyond the run, so megatron only writes the end-of-training save
#     (adapter + merged). One merged model per stage.
#   - each stage enters from the previous stage's MERGED final model with a
#     FRESH LoRA + optimizer (--finetune true), the family's stage rule.
#
# STAGE CACHE (why sweeps are cheap):
#   every stage gets an ID = md5 of EVERYTHING that determines its weights
#   (parent stage ID, data files, LRs, epochs, ranks, where LoRA goes, global
#   hyper-params, seed). Finished stages live in
#     ${STAGE_ROOT}/s<k>-<id>/          (default ${FT_ROOT}/pipe-stages)
#   and are REUSED by any run that asks for the same stage. Two runs that
#   only differ in S3 share one S1 and one S2 — only S3 is trained again.
#   Each run gets a small folder ${PIPE_ROOT} (logs, manifest, result.env,
#   symlinks s1/s2/s3 -> the stage dirs it used).
#   A stage being trained holds <stage_dir>.lock; another run/pod that needs
#   the SAME stage waits for it instead of training it twice. If a pod dies
#   hard (kill -9 / node loss) the lock stays: remove it by hand
#   (rm -rf <stage_dir>.lock) — the waiting run prints its owner every poll.
#
# PER-STAGE KNOBS (k = 1, 2, 3; defaults in section 3):
#   Sk_ENABLE        1 | 0        0 = skip the stage (parent passes through)
#   Sk_DATA          comma/space list; bare names resolve under TEST_DIR
#   Sk_EPOCHS        epochs of this stage (default NUM_TRAIN_EPOCHS=1)
#   Sk_LR / Sk_MIN_LR          LLM LoRA peak / cosine floor
#   Sk_LLM_RANK / Sk_LLM_ALPHA LLM LoRA size (global --lora_rank/alpha)
#   Sk_VIT           lora | frozen
#   Sk_VIT_RANK / Sk_VIT_ALPHA / Sk_VIT_LR          (used when VIT=lora)
#   Sk_MERGER        lora | full | frozen
#   Sk_MERGER_RANK / Sk_MERGER_ALPHA                (used when MERGER=lora)
#   Sk_ALIGNER_LR                                   (used when MERGER!=frozen)
#   Global (all stages): SEED GLOBAL_BATCH_SIZE MAX_LENGTH LR_WARMUP_FRACTION
#   LR_DECAY_STYLE WEIGHT_DECAY CLIP_GRAD LORA_DROPOUT BASE_MODEL
#
#   ViT x merger combinations already run in this family (safe):
#     VIT=lora   MERGER=lora    (S1-L)     VIT=lora MERGER=full (S1)
#     VIT=frozen MERGER=frozen  (S2)
#   Anything else is NEW — the script warns; verify with SMOKE that the
#   intended adapters / param groups appear in the log before a real run.
#
# OTHER MODES:
#   PLAN_ONLY=1   resolve all stages, print 'PLAN <stage> <status> <dir>'
#                 (status: done / locked / todo / disabled / external), exit.
#   SMOKE=1       10 iters per stage, separate stage cache + run root.
#   START_STAGE=2 S1_MODEL=<merged dir>   start from an existing model
#   START_STAGE=3 S2_MODEL=<merged dir>   (e.g. an S2-L checkpoint)
#
# TIME (defaults, ~10.2 rows/window, S1 16.1 s/it, S2/S3 13.3 s/it):
#   S1 ~11h | S2 ~6h | S3 ~12.5h, plus dataset encoding on the first launch.
# DISK: ~52GB per trained stage (merged ~51GB + adapter ~1GB).
#
# USAGE:
#   bash pipeline_s1_s3.sh                                 # defaults
#   S3_LR=5e-6 S3_MIN_LR=5e-7 RUN_TAG=s3_lr5e-6 bash pipeline_s1_s3.sh
#   PLAN_ONLY=1 S3_MERGER=lora bash pipeline_s1_s3.sh      # what would run?
#   many variants: see sweep_pipeline.sh + sweep_configs.txt
#
# VERIFICATION (grep <PIPE_ROOT>/logs/s{1,2,3}.log):
#   1. '[pipeline_plugin] variant->channel patch installed'
#   2. '[pipeline_plugin] rank_pattern=' only on stages with a visual LoRA,
#      merger key FIRST when both keys are present
#   3. frozen-visual stages: NO 'rank_pattern'; trainable = LLM LoRA only
#   4. 'loss_V1' .. 'loss_V4' keys (no loss_V5 unless you add V5 data)
# =============================================================================
PLAN_ONLY=${PLAN_ONLY:-0}
SMOKE=${SMOKE:-0}

# -----------------------------------------------------------------------------
# 0. Environment activation & working dir
# -----------------------------------------------------------------------------
VENV_ACTIVATE=${VENV_ACTIVATE:-/data/autovision-cfs/autovision-notebook/tpz_xayah/bin/activate}
WORK_DIR=${WORK_DIR:-/data/autovision-cfs/autovision-notebook}
# shellcheck disable=SC1090
source "${VENV_ACTIVATE}"
cd "${WORK_DIR}"
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_COMPILE_DISABLE=1
export WANDB_MODE=${WANDB_MODE:-disabled}

export TOKENIZERS_PARALLELISM=False
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS=1

export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-16384}
export VIDEO_MAX_TOKEN_NUM=${VIDEO_MAX_TOKEN_NUM:-128}
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-16}

export MODELSCOPE_CACHE_HOME=${MODELSCOPE_CACHE_HOME:-/data/autovision-cfs/autovision-notebook/logos-remote/sft/cache}

# -----------------------------------------------------------------------------
# 1. Paths & run layout
# -----------------------------------------------------------------------------
BASE_MODEL=${BASE_MODEL:-/data/autovision-cfs/autovision-notebook/algo4/models/Qwen3.5-27B}
FT_ROOT=${FT_ROOT:-/data/autovision-cfs/autovision-notebook/algo4/tpz/ft}
TEST_DIR=${TEST_DIR:-/data/autovision-cfs/autovision-notebook/algo4/tpz/test}

_sfx=""
if [ "${SMOKE}" = "1" ]; then _sfx="-smoke"; fi
RUN_TAG=${RUN_TAG:-default}
STAGE_ROOT=${STAGE_ROOT:-${FT_ROOT}/pipe-stages${_sfx}}
PIPE_ROOT=${PIPE_ROOT:-${FT_ROOT}/pipe-runs${_sfx}/${RUN_TAG}}
LOG_DIR=${PIPE_ROOT}/logs
MANIFEST=${PIPE_ROOT}/pipeline_manifest.txt
LOCK_POLL_SEC=${LOCK_POLL_SEC:-600}

START_STAGE=${START_STAGE:-1}
S1_MODEL=${S1_MODEL:-}          # only read when START_STAGE=2
S2_MODEL=${S2_MODEL:-}          # only read when START_STAGE=3

# -----------------------------------------------------------------------------
# 2. Global hyper-params (identical across stages; part of every stage ID)
# -----------------------------------------------------------------------------
NUM_GPUS=${NUM_GPUS:-8}
TP_SIZE=${TP_SIZE:-4}
PP_SIZE=${PP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}
SEQUENCE_PARALLEL=${SEQUENCE_PARALLEL:-true}
USE_DISTRIBUTED_OPTIMIZER=${USE_DISTRIBUTED_OPTIMIZER:-false}

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}       # throughput knob (not in stage ID)
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}    # packed windows
MAX_LENGTH=${MAX_LENGTH:-8192}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}       # default epochs for every stage
LR_WARMUP_FRACTION=${LR_WARMUP_FRACTION:-0.03}
LR_DECAY_STYLE=${LR_DECAY_STYLE:-cosine}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
CLIP_GRAD=${CLIP_GRAD:-1.0}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
SEED=${SEED:-42}

RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-selective}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flash}
PACKING=${PACKING:-true}
PADDING_FREE=${PADDING_FREE:-false}           # packing forces padding_free internally

# NO intermediate checkpoints: only megatron's end-of-training save is
# written (the S2 run proved it: final checkpoint-3683 was saved although
# 3683 is not a multiple of its 612 save interval).
SAVE_STEPS=1000000
SAVE_TOTAL_LIMIT=1

DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-32}
DATASET_NUM_PROC=${DATASET_NUM_PROC:-16}
PACKING_NUM_PROC=${PACKING_NUM_PROC:-16}

# -----------------------------------------------------------------------------
# 3. Per-stage defaults (any of these can be overridden from the env)
# -----------------------------------------------------------------------------
: "${S1_ENABLE:=1}" "${S2_ENABLE:=1}" "${S3_ENABLE:=1}"
: "${S1_DATA:=train_caption_ultra_x_v1mini.parquet,train_caption_ultra_x_v2mini.parquet,train_caption_ultra_x_v3mini.parquet,train_caption_ultra_x_v4mini.parquet}"
: "${S2_DATA:=train_caption_ultra_x_v1medium.parquet,train_caption_ultra_x_v2medium.parquet,train_caption_ultra_x_v3medium.parquet,train_caption_ultra_x_v4medium.parquet}"
: "${S3_DATA:=train_caption_ultra_x_v1final.parquet,train_caption_ultra_x_v2final.parquet,train_caption_ultra_x_v3final.parquet,train_caption_ultra_x_v4final.parquet}"
: "${S1_EPOCHS:=${NUM_TRAIN_EPOCHS}}" "${S2_EPOCHS:=${NUM_TRAIN_EPOCHS}}" "${S3_EPOCHS:=${NUM_TRAIN_EPOCHS}}"
# LLM LoRA
: "${S1_LR:=3e-5}" "${S1_MIN_LR:=3e-6}"
: "${S2_LR:=2e-5}" "${S2_MIN_LR:=2e-6}"
: "${S3_LR:=1e-5}" "${S3_MIN_LR:=1e-6}"
: "${S1_LLM_RANK:=64}" "${S1_LLM_ALPHA:=128}"
: "${S2_LLM_RANK:=64}" "${S2_LLM_ALPHA:=128}"
: "${S3_LLM_RANK:=64}" "${S3_LLM_ALPHA:=128}"
# ViT: S1 LoRA r16/a32 @ 1e-5; S2/S3 frozen (LRs below apply if switched on)
: "${S1_VIT:=lora}"   "${S2_VIT:=frozen}"   "${S3_VIT:=frozen}"
: "${S1_VIT_RANK:=16}" "${S2_VIT_RANK:=16}" "${S3_VIT_RANK:=16}"
: "${S1_VIT_ALPHA:=32}" "${S2_VIT_ALPHA:=32}" "${S3_VIT_ALPHA:=32}"
: "${S1_VIT_LR:=1e-5}" "${S2_VIT_LR:=1e-5}" "${S3_VIT_LR:=5e-6}"
# merger (aligner): S1 LoRA r32/a64 @ 1e-5 (S1-L recipe); S2/S3 frozen
: "${S1_MERGER:=lora}" "${S2_MERGER:=frozen}" "${S3_MERGER:=frozen}"
: "${S1_MERGER_RANK:=32}" "${S2_MERGER_RANK:=32}" "${S3_MERGER_RANK:=32}"
: "${S1_MERGER_ALPHA:=64}" "${S2_MERGER_ALPHA:=64}" "${S3_MERGER_ALPHA:=64}"
: "${S1_ALIGNER_LR:=1e-5}" "${S2_ALIGNER_LR:=1e-5}" "${S3_ALIGNER_LR:=5e-6}"

# -----------------------------------------------------------------------------
# 4. Helpers
# -----------------------------------------------------------------------------
die() { echo "[error] $*" >&2; exit 1; }
sv() { local n="S$1_$2"; printf '%s' "${!n-}"; }       # stage var: sv 3 LR
short_hash() { python3 -c 'import hashlib,sys;print(hashlib.md5(sys.stdin.buffer.read()).hexdigest()[:10])'; }

# Echo the final merged model dir of a stage output dir (empty if none).
final_merged() {
    local out=$1 it d
    if [ -f "${out}/latest_checkpointed_iteration.txt" ]; then
        it=$(cat "${out}/latest_checkpointed_iteration.txt")
        d="${out}/checkpoint-${it}-merged"
        if [ -f "${d}/config.json" ]; then echo "${d}"; return 0; fi
    fi
    for d in $(ls -d "${out}"/checkpoint-*-merged 2>/dev/null | sort -V -r); do
        if [ -f "${d}/config.json" ]; then echo "${d}"; return 0; fi
    done
    return 0
}

stage_status() {
    local dir=$1
    if [ -n "$(final_merged "${dir}")" ]; then echo done
    elif [ -d "${dir}.lock" ]; then echo locked
    else echo todo; fi
}

CURRENT_LOCK=""
release_lock() {
    if [ -n "${CURRENT_LOCK}" ]; then rm -rf "${CURRENT_LOCK}"; CURRENT_LOCK=""; fi
}
trap release_lock EXIT

# -----------------------------------------------------------------------------
# 5. Resolve all stages up front (validation + stage IDs; nothing runs yet)
# -----------------------------------------------------------------------------
# R_*[k] hold the resolved config of stage k.
R_ENABLE=(); R_FILES=(); R_SIG=(); R_DIR=(); R_PKIND=(); R_PVAL=()

case "${START_STAGE}" in
    1) parent_id="base:${BASE_MODEL}"; parent_kind=path; parent_val=${BASE_MODEL} ;;
    2) parent_id="external:${S1_MODEL}"; parent_kind=path; parent_val=${S1_MODEL} ;;
    3) parent_id="external:${S2_MODEL}"; parent_kind=path; parent_val=${S2_MODEL} ;;
    *) die "START_STAGE must be 1, 2 or 3" ;;
esac
[ -f "${parent_val}/config.json" ] || die "missing ${parent_val}/config.json (BASE_MODEL / S1_MODEL / S2_MODEL)"

for k in 1 2 3; do
    if [ "${k}" -lt "${START_STAGE}" ]; then R_ENABLE[k]=external; continue; fi
    en=$(sv $k ENABLE)
    case "${en}" in 0|1) ;; *) die "S${k}_ENABLE must be 0 or 1 (got '${en}')" ;; esac
    R_ENABLE[k]=${en}
    [ "${en}" = "1" ] || continue

    vit=$(sv $k VIT); merger=$(sv $k MERGER)
    case "${vit}" in lora|frozen) ;; *) die "S${k}_VIT must be lora|frozen (got '${vit}')" ;; esac
    case "${merger}" in lora|full|frozen) ;; *) die "S${k}_MERGER must be lora|full|frozen (got '${merger}')" ;; esac
    case "${vit}/${merger}" in
        lora/lora|lora/full|frozen/frozen) ;;
        *) echo "[warn] S${k}: VIT=${vit} MERGER=${merger} has never been run in this family — SMOKE it first" >&2 ;;
    esac

    # data: comma/space list, bare names under TEST_DIR
    files=""
    for f in $(printf '%s' "$(sv $k DATA)" | tr ',' ' '); do
        case "${f}" in /*) ;; *) f="${TEST_DIR}/${f}" ;; esac
        [ -f "${f}" ] || die "S${k} data file not found: ${f}"
        files="${files:+${files} }${f}"
    done
    [ -n "${files}" ] || die "S${k}_DATA is empty"
    R_FILES[k]=${files}

    for key in LR MIN_LR EPOCHS LLM_RANK LLM_ALPHA; do
        [ -n "$(sv $k ${key})" ] || die "S${k}_${key} is empty"
    done

    # stage ID: only the knobs that actually change this stage's weights
    sig="stage=s${k}
parent=${parent_id}
data=${files}
lr=$(sv $k LR) min_lr=$(sv $k MIN_LR) epochs=$(sv $k EPOCHS)
llm_lora=r$(sv $k LLM_RANK)/a$(sv $k LLM_ALPHA) dropout=${LORA_DROPOUT}
vit=${vit}"
    if [ "${vit}" = "lora" ]; then
        sig="${sig} r$(sv $k VIT_RANK)/a$(sv $k VIT_ALPHA) lr=$(sv $k VIT_LR)"
    fi
    sig="${sig}
merger=${merger}"
    case "${merger}" in
        lora) sig="${sig} r$(sv $k MERGER_RANK)/a$(sv $k MERGER_ALPHA) lr=$(sv $k ALIGNER_LR)" ;;
        full) sig="${sig} lr=$(sv $k ALIGNER_LR)" ;;
    esac
    sig="${sig}
gbs=${GLOBAL_BATCH_SIZE} max_length=${MAX_LENGTH} warmup=${LR_WARMUP_FRACTION} decay=${LR_DECAY_STYLE} wd=${WEIGHT_DECAY} clip=${CLIP_GRAD} seed=${SEED} packing=${PACKING} smoke=${SMOKE}"
    id=$(printf '%s' "${sig}" | short_hash)

    R_SIG[k]=${sig}
    R_DIR[k]="${STAGE_ROOT}/s${k}-${id}"
    R_PKIND[k]=${parent_kind}
    R_PVAL[k]=${parent_val}
    parent_id="s${k}-${id}"; parent_kind=stage; parent_val=${R_DIR[k]}
done

# -----------------------------------------------------------------------------
# 6. PLAN_ONLY: report and exit
# -----------------------------------------------------------------------------
if [ "${PLAN_ONLY}" = "1" ]; then
    for k in 1 2 3; do
        case "${R_ENABLE[k]}" in
            external) echo "PLAN s${k} external -" ;;
            0)        echo "PLAN s${k} disabled -" ;;
            *)        echo "PLAN s${k} $(stage_status "${R_DIR[k]}") ${R_DIR[k]}" ;;
        esac
    done
    exit 0
fi

# -----------------------------------------------------------------------------
# 7. Self-generate the pipeline plugin (shared by all stages)
# -----------------------------------------------------------------------------
mkdir -p "${PIPE_ROOT}" "${LOG_DIR}" "${STAGE_ROOT}"
PLUGIN_PATH=${PLUGIN_PATH:-${PIPE_ROOT}/_pipeline_plugin.py}

cat > "${PLUGIN_PATH}" <<'PIPELINE_PLUGIN_EOF'
r"""pipeline plugin (auto-generated by train_sft_qwen3_5_27b_megatron_lora_h200_pipeline_s1_s3.sh).

Loaded at ARGS-PARSING time via --external_plugins. Each stage is its own
`megatron sft` process, so the env vars below are read per stage.

patch A (always): variant -> 'V%d' channel (logging only).
patch B (when PIPELINE_MERGER_RANK and/or PIPELINE_VIT_RANK are set):
    LoraConfig wrap injecting rank_pattern/alpha_pattern. peft takes the
    FIRST suffix-match in dict order and the general visual key also matches
    the merger, so the merger key is inserted FIRST. LLM modules match no
    key -> global --lora_rank/--lora_alpha. Same mechanism as the S1-L plugin.
"""
import os

# --- patch A: variant -> channel -------------------------------------------
from swift.dataset.preprocessor.core import RowPreprocessor as _RowPreprocessor

_orig_batched_preprocess = _RowPreprocessor.batched_preprocess
_vdc_translated_once = [False]


def _pipeline_batched_preprocess(self, batched_row, *, strict,
                                 ignore_max_length_error):
    batched_row = dict(batched_row)
    if 'channel' not in batched_row and 'variant' in batched_row:
        try:
            ch = ['V%d' % v if v is not None else None
                  for v in batched_row['variant']]
            batched_row['channel'] = ch
            if not _vdc_translated_once[0]:
                _vdc_translated_once[0] = True
                print('[pipeline_plugin] first variant->channel batch: '
                      f'{len(ch)} rows, e.g. {ch[:8]}')
        except Exception as e:
            print(f'[pipeline_plugin] variant->channel translation failed: {e!r}')
    return _orig_batched_preprocess(self, batched_row, strict=strict,
                                    ignore_max_length_error=ignore_max_length_error)


if not getattr(getattr(_RowPreprocessor, 'batched_preprocess', None),
               '_pipeline_patched', False):
    _pipeline_batched_preprocess._pipeline_patched = True
    _RowPreprocessor.batched_preprocess = _pipeline_batched_preprocess
    print('[pipeline_plugin] variant->channel patch installed '
          '(RowPreprocessor.batched_preprocess)')

# --- patch B: visual rank_pattern (only for stages with a visual LoRA) ------
_rank, _alpha = {}, {}
if os.environ.get('PIPELINE_MERGER_RANK'):                 # MUST come first
    _rank[r'visual\.merger\..*'] = int(os.environ['PIPELINE_MERGER_RANK'])
    _alpha[r'visual\.merger\..*'] = int(os.environ['PIPELINE_MERGER_ALPHA'])
if os.environ.get('PIPELINE_VIT_RANK'):
    _rank[r'visual\..*'] = int(os.environ['PIPELINE_VIT_RANK'])
    _alpha[r'visual\..*'] = int(os.environ['PIPELINE_VIT_ALPHA'])

if _rank:
    from swift.megatron.utils import utils as _swift_megatron_utils

    _OrigLoraConfig = _swift_megatron_utils.LoraConfig

    def _pipeline_lora_config(*args, **kwargs):
        kwargs.setdefault('rank_pattern', dict(_rank))
        kwargs.setdefault('alpha_pattern', dict(_alpha))
        cfg = _OrigLoraConfig(*args, **kwargs)
        print(f'[pipeline_plugin] rank_pattern={dict(cfg.rank_pattern)} '
              f'alpha_pattern={dict(cfg.alpha_pattern)} '
              f'(global r={cfg.r} alpha={cfg.lora_alpha})')
        return cfg

    if not getattr(getattr(_swift_megatron_utils, 'LoraConfig', None),
                   '_pipeline_patched', False):
        _pipeline_lora_config._pipeline_patched = True
        _swift_megatron_utils.LoraConfig = _pipeline_lora_config
PIPELINE_PLUGIN_EOF
[ -f "${PLUGIN_PATH}" ] || die "failed to generate ${PLUGIN_PATH}"

# -----------------------------------------------------------------------------
# 8. Run one stage (reuse / wait on lock / train)
# -----------------------------------------------------------------------------
STAGE_RESULT=""
run_stage() {
    local k=$1
    local out=${R_DIR[k]} log="${LOG_DIR}/s${k}.log" model done_dir

    if [ "${R_PKIND[k]}" = "stage" ]; then
        model=$(final_merged "${R_PVAL[k]}")
        [ -n "${model}" ] || die "s${k}: parent stage ${R_PVAL[k]} has no merged model"
    else
        model=${R_PVAL[k]}
    fi

    # reuse a finished stage, or take its lock, or wait for whoever holds it
    while true; do
        done_dir=$(final_merged "${out}")
        if [ -n "${done_dir}" ]; then break; fi
        if mkdir "${out}.lock" 2>/dev/null; then
            CURRENT_LOCK="${out}.lock"
            echo "host=$(hostname) pid=$$ run=${RUN_TAG} since=$(date '+%F %T')" > "${CURRENT_LOCK}/owner"
            done_dir=$(final_merged "${out}")   # finished while we raced?
            break
        fi
        echo "[pipeline] s${k} is being trained elsewhere ($(cat "${out}.lock/owner" 2>/dev/null || echo 'owner unknown')); waiting ${LOCK_POLL_SEC}s"
        echo "[pipeline]   stale lock? rm -rf ${out}.lock"
        sleep "${LOCK_POLL_SEC}"
    done

    if [ -n "${done_dir}" ]; then
        release_lock
        echo "[pipeline] s${k} reused -> ${done_dir}"
        echo "s${k} REUSED $(date '+%F %T') -> ${done_dir}" >> "${MANIFEST}"
        ln -sfn "${out}" "${PIPE_ROOT}/s${k}"
        STAGE_RESULT=${done_dir}
        return 0
    fi

    if [ -e "${out}" ]; then
        local aside
        aside="${out}.failed-$(date +%Y%m%d_%H%M%S)"
        echo "[pipeline] s${k}: partial output found, moving aside -> ${aside}"
        mv "${out}" "${aside}"
    fi
    mkdir -p "${out}"
    printf '%s\n' "${R_SIG[k]}" > "${out}/stage_config.txt"
    ln -sfn "${out}" "${PIPE_ROOT}/s${k}"

    local vit merger visual_args=() len_args=() files=()
    vit=$(sv $k VIT); merger=$(sv $k MERGER)
    export PIPELINE_VIT_RANK="" PIPELINE_VIT_ALPHA="" PIPELINE_MERGER_RANK="" PIPELINE_MERGER_ALPHA=""
    if [ "${vit}" = "lora" ]; then
        visual_args+=(--freeze_vit false --vit_lr "$(sv $k VIT_LR)")
        export PIPELINE_VIT_RANK="$(sv $k VIT_RANK)" PIPELINE_VIT_ALPHA="$(sv $k VIT_ALPHA)"
    else
        visual_args+=(--freeze_vit true)
    fi
    case "${merger}" in
        lora)
            visual_args+=(--freeze_aligner false --aligner_lr "$(sv $k ALIGNER_LR)")
            export PIPELINE_MERGER_RANK="$(sv $k MERGER_RANK)" PIPELINE_MERGER_ALPHA="$(sv $k MERGER_ALPHA)" ;;
        full)   # S1 recipe: excluded from LoRA targets, trained full-param
            visual_args+=(--freeze_aligner true --modules_to_save visual.visual.merger
                          --aligner_lr "$(sv $k ALIGNER_LR)") ;;
        frozen)
            visual_args+=(--freeze_aligner true) ;;
    esac

    if [ "${SMOKE}" = "1" ]; then
        len_args+=(--train_iters 10)         # NO --num_train_epochs in SMOKE
    else                                     # (megatron_args.py:934 override)
        len_args+=(--num_train_epochs "$(sv $k EPOCHS)")
    fi
    read -r -a files <<< "${R_FILES[k]}"
    local llm_r llm_a lr min_lr
    llm_r=$(sv $k LLM_RANK); llm_a=$(sv $k LLM_ALPHA); lr=$(sv $k LR); min_lr=$(sv $k MIN_LR)

    echo "[pipeline] s${k} START $(date '+%F %T') model=${model} out=${out}"
    sed 's/^/[pipeline]   /' "${out}/stage_config.txt"
    echo "s${k} START $(date '+%F %T') model=${model} out=${out}" >> "${MANIFEST}"

    set -x
    NPROC_PER_NODE=${NUM_GPUS} \
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
    megatron sft \
        --model "${model}" \
        --dataset "${files[@]}" \
        --split_dataset_ratio 0 \
        --load_from_cache_file true \
        --save_safetensors true \
        --tensor_model_parallel_size "${TP_SIZE}" \
        --pipeline_model_parallel_size "${PP_SIZE}" \
        --context_parallel_size "${CP_SIZE}" \
        --expert_model_parallel_size "${EP_SIZE}" \
        --sequence_parallel "${SEQUENCE_PARALLEL}" \
        --use_distributed_optimizer "${USE_DISTRIBUTED_OPTIMIZER}" \
        --recompute_granularity "${RECOMPUTE_GRANULARITY}" \
        --attention_backend "${ATTENTION_BACKEND}" \
        --packing "${PACKING}" \
        --padding_free "${PADDING_FREE}" \
        --tuner_type lora \
        --lora_rank "${llm_r}" \
        --lora_alpha "${llm_a}" \
        --lora_dropout "${LORA_DROPOUT}" \
        --merge_lora true \
        --freeze_llm false \
        "${visual_args[@]}" \
        --external_plugins "${PLUGIN_PATH}" \
        --enable_channel_loss true \
        --micro_batch_size "${MICRO_BATCH_SIZE}" \
        --global_batch_size "${GLOBAL_BATCH_SIZE}" \
        --max_length "${MAX_LENGTH}" \
        --lr "${lr}" \
        --min_lr "${min_lr}" \
        --lr_warmup_fraction "${LR_WARMUP_FRACTION}" \
        --lr_decay_style "${LR_DECAY_STYLE}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --clip_grad "${CLIP_GRAD}" \
        --seed "${SEED}" \
        "${len_args[@]}" \
        --cross_entropy_loss_fusion true \
        --finetune true \
        --no_save_optim true \
        --no_save_rng true \
        --save_strategy steps \
        --save_steps "${SAVE_STEPS}" \
        --save_total_limit "${SAVE_TOTAL_LIMIT}" \
        --add_version false \
        --output_dir "${out}" \
        --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
        --dataset_num_proc "${DATASET_NUM_PROC}" \
        --packing_num_proc "${PACKING_NUM_PROC}" \
        2>&1 | tee "${log}" "${out}/pipeline_stage.log"
    { set +x; } 2>/dev/null

    done_dir=$(final_merged "${out}")
    [ -n "${done_dir}" ] || die "s${k} finished but no merged model found in ${out}"
    release_lock
    echo "[pipeline] s${k} done -> ${done_dir}"
    echo "s${k} DONE  $(date '+%F %T') -> ${done_dir}" >> "${MANIFEST}"
    STAGE_RESULT=${done_dir}
}

# -----------------------------------------------------------------------------
# 9. The chain
# -----------------------------------------------------------------------------
echo "[pipeline] RUN_TAG=${RUN_TAG} PIPE_ROOT=${PIPE_ROOT} STAGE_ROOT=${STAGE_ROOT} START_STAGE=${START_STAGE} SMOKE=${SMOKE}"
echo "=== run ${RUN_TAG} $(date '+%F %T') ===" >> "${MANIFEST}"

RESULT=("" "" "" "")
for k in 1 2 3; do
    case "${R_ENABLE[k]}" in
        external)
            if [ "${k}" -eq $((START_STAGE - 1)) ]; then
                if [ "${k}" -eq 1 ]; then RESULT[k]=${S1_MODEL}; else RESULT[k]=${S2_MODEL}; fi
            else
                RESULT[k]=EXTERNAL
            fi ;;
        0)  RESULT[k]=DISABLED
            echo "[pipeline] s${k} disabled (parent passes through)" ;;
        1)  run_stage "${k}"
            RESULT[k]=${STAGE_RESULT} ;;
    esac
done

{
    echo "S1_RESULT=${RESULT[1]}"
    echo "S2_RESULT=${RESULT[2]}"
    echo "S3_RESULT=${RESULT[3]}"
} > "${PIPE_ROOT}/result.env"

echo "================================================================"
echo "[pipeline] ${RUN_TAG}: ALL STAGES DONE"
echo "  S1: ${RESULT[1]}"
echo "  S2: ${RESULT[2]}"
echo "  S3: ${RESULT[3]}"
echo "  result.env / manifest / logs under ${PIPE_ROOT}"
echo "================================================================"
