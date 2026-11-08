#!/usr/bin/env bash
# 唯一推荐路线；历史脚本和 sweep 保留作实验记录。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# plan 不需要激活远端环境；preflight/smoke/run 使用训练环境。
if [[ "${1:-plan}" != plan && "${SKIP_ACTIVATE:-0}" != 1 ]]; then
    WORK_DIR=${WORK_DIR:-/data/autovision-cfs/autovision-notebook}
    VENV_ACTIVATE=${VENV_ACTIVATE:-${WORK_DIR}/tpz_xayah/bin/activate}
    [[ -f "${VENV_ACTIVATE}" ]] || { echo "训练环境不存在：${VENV_ACTIVATE}" >&2; exit 1; }
    source "${VENV_ACTIVATE}"
fi
exec python3 "${SCRIPT_DIR}/train_s1_s3_recommended.py" "${@:-plan}"
