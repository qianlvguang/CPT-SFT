#!/usr/bin/env bash
# =============================================================================
# sweep_pipeline.sh — run many variants of the S1->S2->S3 pipeline
# (train_sft_qwen3_5_27b_megatron_lora_h200_pipeline_s1_s3.sh) from ONE list
# of experiments, then hand every resulting model to your eval.
#
# EXPERIMENT LIST (default: sweep_configs.txt next to this script):
#   one experiment per line:  <name>  KEY=VALUE KEY=VALUE ...
#   '#' starts a comment; a name with no overrides = pipeline defaults.
#   Allowed keys (typos are rejected, never silently ignored):
#     S1_* / S2_* / S3_*:  ENABLE DATA EPOCHS LR MIN_LR LLM_RANK LLM_ALPHA
#                          VIT VIT_RANK VIT_ALPHA VIT_LR
#                          MERGER MERGER_RANK MERGER_ALPHA ALIGNER_LR
#     global:              SEED GLOBAL_BATCH_SIZE MAX_LENGTH LR_WARMUP_FRACTION
#                          LR_DECAY_STYLE WEIGHT_DECAY CLIP_GRAD LORA_DROPOUT
#                          BASE_MODEL START_STAGE S1_MODEL S2_MODEL
#   Values contain no spaces; Sk_DATA uses commas between files.
#
# STAGE SHARING: the pipeline caches every finished stage by a hash of its
# full config, so experiments that share S1 (or S1+S2) train it ONCE.
# Order the list so the shared prefix is trained first (the baseline).
# `plan` shows exactly which stages are shared and what is left to train.
#
# COMMANDS:
#   bash sweep_pipeline.sh list                 # experiments + overrides
#   bash sweep_pipeline.sh plan   [names...]    # per-stage status, sharing,
#                                               #   unique stages left, disk
#   bash sweep_pipeline.sh run    [names...]    # run them one after another
#   bash sweep_pipeline.sh results [names...]   # model paths (for eval)
#   (no names = every experiment in the list)
#
#   several pods: give each pod different names, e.g.
#     pod A: nohup bash sweep_pipeline.sh run baseline s3_lr5e-6 > a.log 2>&1 &
#     pod B: nohup bash sweep_pipeline.sh run s1_merger_full      > b.log 2>&1 &
#   if two pods need the same stage, the second one waits for the first.
#
# EVAL HOOK (optional): EVAL_CMD runs after each successful experiment, with
#   {name} {s1} {s2} {s3} replaced by the experiment name / model paths:
#     EVAL_CMD='bash /path/eval_oms.sh {s3} {name}' bash sweep_pipeline.sh run
#   Its output goes to logs/<name>.eval.log; a failing eval never stops the
#   sweep.
#
# ENV:
#   SWEEP_FILE   experiment list        (default ./sweep_configs.txt)
#   SWEEP_NAME   folder name for runs   (default: SWEEP_FILE basename)
#   SMOKE=1      10 iters per stage, separate smoke caches
#   FT_ROOT      same default as the pipeline
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE=${PIPELINE:-${SCRIPT_DIR}/train_sft_qwen3_5_27b_megatron_lora_h200_pipeline_s1_s3.sh}
SWEEP_FILE=${SWEEP_FILE:-${SCRIPT_DIR}/sweep_configs.txt}
SWEEP_NAME=${SWEEP_NAME:-$(basename "${SWEEP_FILE}" .txt)}
FT_ROOT=${FT_ROOT:-/data/autovision-cfs/autovision-notebook/algo4/tpz/ft}
SMOKE=${SMOKE:-0}
EVAL_CMD=${EVAL_CMD:-}

_sfx=""
if [ "${SMOKE}" = "1" ]; then _sfx="-smoke"; fi
RUNS_ROOT=${FT_ROOT}/pipe-runs${_sfx}/${SWEEP_NAME}
SWEEP_LOGS=${RUNS_ROOT}/logs
STATUS_FILE=${RUNS_ROOT}/sweep_status.tsv

GB_PER_STAGE=52        # merged ~51GB + adapter ~1GB

die() { echo "[sweep][error] $*" >&2; exit 1; }
[ -f "${PIPELINE}" ] || die "pipeline script not found: ${PIPELINE}"
[ -f "${SWEEP_FILE}" ] || die "experiment list not found: ${SWEEP_FILE}"

# Stage knobs left in the calling shell would leak into EVERY experiment —
# clear them so each experiment is exactly defaults + its own overrides.
for _v in $(compgen -v | grep -E '^S[123]_' || true); do
    echo "[sweep] note: ignoring ${_v} from the calling shell" >&2
    unset "${_v}"
done

KEY_RE='^((S[123]_(ENABLE|DATA|EPOCHS|LR|MIN_LR|LLM_RANK|LLM_ALPHA|VIT|VIT_RANK|VIT_ALPHA|VIT_LR|MERGER|MERGER_RANK|MERGER_ALPHA|ALIGNER_LR))|SEED|GLOBAL_BATCH_SIZE|MAX_LENGTH|LR_WARMUP_FRACTION|LR_DECAY_STYLE|WEIGHT_DECAY|CLIP_GRAD|LORA_DROPOUT|BASE_MODEL|START_STAGE|S1_MODEL|S2_MODEL)$'

# -----------------------------------------------------------------------------
# Parse the experiment list into NAMES[i] / OVR[i]
# -----------------------------------------------------------------------------
NAMES=(); OVR=()
_lineno=0
while IFS= read -r _line || [ -n "${_line}" ]; do
    _lineno=$((_lineno + 1))
    _line=${_line%%#*}
    read -r _name _rest <<< "${_line}" || true
    [ -n "${_name:-}" ] || continue
    [[ "${_name}" =~ ^[A-Za-z0-9_.-]+$ ]] || die "${SWEEP_FILE}:${_lineno}: bad name '${_name}'"
    for _n in "${NAMES[@]+"${NAMES[@]}"}"; do
        [ "${_n}" != "${_name}" ] || die "${SWEEP_FILE}:${_lineno}: duplicate name '${_name}'"
    done
    for _kv in ${_rest:-}; do
        [[ "${_kv}" == *=* ]] || die "${SWEEP_FILE}:${_lineno}: '${_kv}' is not KEY=VALUE"
        [[ "${_kv%%=*}" =~ ${KEY_RE} ]] || die "${SWEEP_FILE}:${_lineno}: unknown key '${_kv%%=*}'"
        [ -n "${_kv#*=}" ] || die "${SWEEP_FILE}:${_lineno}: empty value for '${_kv%%=*}'"
    done
    NAMES+=("${_name}"); OVR+=("${_rest:-}")
done < "${SWEEP_FILE}"
[ "${#NAMES[@]}" -gt 0 ] || die "no experiments in ${SWEEP_FILE}"

# select experiments by name (default: all), keeping list order
CMD=${1:-}
[ -n "${CMD}" ] || die "usage: $0 {list|plan|run|results} [names...]"
shift
SEL=()
if [ "$#" -eq 0 ]; then
    for _i in "${!NAMES[@]}"; do SEL+=("${_i}"); done
else
    for _want in "$@"; do
        _found=""
        for _i in "${!NAMES[@]}"; do
            if [ "${NAMES[_i]}" = "${_want}" ]; then _found=${_i}; fi
        done
        [ -n "${_found}" ] || die "no experiment named '${_want}' in ${SWEEP_FILE}"
        SEL+=("${_found}")
    done
fi

# pipeline_env <i> -> KEY=VALUE words for env(1)
pipeline_env() {
    local i=$1
    local -a kv=()
    read -r -a kv <<< "${OVR[i]}"
    printf '%s\n' "${kv[@]+"${kv[@]}"}" \
        "RUN_TAG=${NAMES[i]}" "PIPE_ROOT=${RUNS_ROOT}/${NAMES[i]}" \
        "FT_ROOT=${FT_ROOT}" "SMOKE=${SMOKE}"
}

# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------
case "${CMD}" in
list)
    for _i in "${SEL[@]}"; do
        printf '%-24s %s\n' "${NAMES[_i]}" "${OVR[_i]:-(defaults)}"
    done
    ;;

plan)
    _all=$(mktemp)
    for _i in "${SEL[@]}"; do
        _envs=(); while IFS= read -r _e; do _envs+=("${_e}"); done < <(pipeline_env "${_i}")
        echo "== ${NAMES[_i]}  ${OVR[_i]:-(defaults)}"
        _out=$(env "${_envs[@]}" PLAN_ONLY=1 bash "${PIPELINE}" 2>&1) || { echo "${_out}"; die "plan failed for ${NAMES[_i]}"; }
        echo "${_out}" | grep -v '^PLAN' | sed 's/^/   /' || true
        echo "${_out}" | grep '^PLAN' | while read -r _p _stage _status _dir; do
            printf '   %s  %-8s %s\n' "${_stage}" "${_status}" "${_dir}"
            if [ "${_dir}" != "-" ]; then printf '%s %s %s\n' "${_stage}" "${_status}" "${_dir}" >> "${_all}"; fi
        done
    done
    echo "== summary"
    for _s in s1 s2 s3; do
        _uniq=$(awk -v s="${_s}" '$1==s {print $3}' "${_all}" | sort -u | wc -l | tr -d ' ')
        _todo=$(awk -v s="${_s}" '$1==s && $2!="done" {print $3}' "${_all}" | sort -u | wc -l | tr -d ' ')
        echo "   ${_s}: ${_uniq} unique stage(s), ${_todo} still to train"
    done
    _todo_all=$(awk '$2!="done" {print $3}' "${_all}" | sort -u | wc -l | tr -d ' ')
    echo "   total stages to train: ${_todo_all}  (~$((_todo_all * GB_PER_STAGE)) GB new disk)"
    echo "   rough time @ default data: S1 ~11h, S2 ~6h, S3 ~12.5h per stage"
    rm -f "${_all}"
    ;;

run)
    mkdir -p "${SWEEP_LOGS}"
    [ -f "${STATUS_FILE}" ] || printf 'time\tname\tstatus\ts1\ts2\ts3\n' > "${STATUS_FILE}"
    _n_ok=0; _n_fail=0
    for _i in "${SEL[@]}"; do
        _name=${NAMES[_i]}
        _log=${SWEEP_LOGS}/${_name}.log
        _envs=(); while IFS= read -r _e; do _envs+=("${_e}"); done < <(pipeline_env "${_i}")
        echo "[sweep] $(date '+%F %T') START ${_name}  ${OVR[_i]:-(defaults)}  log=${_log}"
        if env "${_envs[@]}" bash "${PIPELINE}" > "${_log}" 2>&1; then
            _status=OK; _n_ok=$((_n_ok + 1))
        else
            _status=FAIL; _n_fail=$((_n_fail + 1))
            echo "[sweep] ${_name} FAILED — tail of ${_log}:"; tail -n 20 "${_log}" | sed 's/^/    /'
        fi
        S1_RESULT=""; S2_RESULT=""; S3_RESULT=""
        _res=${RUNS_ROOT}/${_name}/result.env
        if [ "${_status}" = "OK" ] && [ -f "${_res}" ]; then
            # shellcheck disable=SC1090
            source "${_res}"
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "${_name}" "${_status}" \
            "${S1_RESULT:--}" "${S2_RESULT:--}" "${S3_RESULT:--}" >> "${STATUS_FILE}"
        echo "[sweep] $(date '+%F %T') ${_status} ${_name}  s3=${S3_RESULT:--}"

        if [ "${_status}" = "OK" ] && [ -n "${EVAL_CMD}" ]; then
            _cmd=${EVAL_CMD//\{name\}/${_name}}
            _cmd=${_cmd//\{s1\}/${S1_RESULT}}
            _cmd=${_cmd//\{s2\}/${S2_RESULT}}
            _cmd=${_cmd//\{s3\}/${S3_RESULT}}
            echo "[sweep] eval: ${_cmd}"
            if bash -c "${_cmd}" > "${SWEEP_LOGS}/${_name}.eval.log" 2>&1; then
                echo "[sweep] eval OK  ${_name}"
            else
                echo "[sweep] eval FAILED ${_name} (see ${SWEEP_LOGS}/${_name}.eval.log)"
            fi
        fi
    done
    echo "[sweep] finished: ${_n_ok} OK, ${_n_fail} FAILED  (status: ${STATUS_FILE})"
    [ "${_n_fail}" -eq 0 ]
    ;;

results)
    printf '%-24s %-6s %s\n' name stage model
    for _i in "${SEL[@]}"; do
        _res=${RUNS_ROOT}/${NAMES[_i]}/result.env
        if [ ! -f "${_res}" ]; then
            printf '%-24s %-6s %s\n' "${NAMES[_i]}" "-" "(not finished)"
            continue
        fi
        while IFS='=' read -r _k _v; do
            printf '%-24s %-6s %s\n' "${NAMES[_i]}" "${_k%_RESULT}" "${_v}"
        done < "${_res}"
    done
    ;;

*)
    die "unknown command '${CMD}' (use list | plan | run | results)"
    ;;
esac
