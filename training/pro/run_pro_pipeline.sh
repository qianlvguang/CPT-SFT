#!/usr/bin/env bash
set -euo pipefail

# Phase C (Pro Closed-Loop Evolution) Orchestrator:
#   Step 1: Triage error clusters & build 50/30/20 incremental CPT + sft_overlap=0 isolated SFT
#   Step 2: Launch / Dry-Run B300 27B Pro incremental CPT (`cpt_27b/run_cpt_27b_b300.sh pro`)
#   Step 3: Run Montessori-Instruct local influence & API preference alignment (`montessori/run_montessori_pipeline.sh`)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CPT27B_DIR="$TRAINING_DIR/cpt_27b"

WORK_DIR=${WORK_DIR:-"${TMPDIR:-/tmp}/cpt_sft_pro_pipeline"}
MODEL=${MODEL:-/data/autovision-cfs/autovision-notebook/algo4/models/Qwen3.5-27B}
DEMO=${DEMO:-0}
DRY_RUN=${DRY_RUN:-1}

mkdir -p "$WORK_DIR"

MINER_DIR="$WORK_DIR/step1_error_miner"
CPT_OUT_DIR="$WORK_DIR/step2_pro_cpt_ckpt"
MON_DIR="$WORK_DIR/step3_montessori"

echo "=== [1/3] Mining Error Clusters & Building 50/30/20 Incremental CPT + sft_overlap=0 ==="
miner_cmd=(python3 "$SCRIPT_DIR/error_cluster_miner.py" --output-dir "$MINER_DIR")
if [[ "$DEMO" == "1" || ( -z ${ERROR_LOGS:-} && "$DRY_RUN" == "1" ) ]]; then
  miner_cmd+=(--demo --total-cpt-rows 20)
else
  : "${ERROR_LOGS:?Set ERROR_LOGS JSONL path (or pass DEMO=1)}"
  miner_cmd+=(--error-logs "$ERROR_LOGS")
  [[ -n ${ACCEPTED_TEMPLATES:-} ]] && miner_cmd+=(--accepted-templates "$ACCEPTED_TEMPLATES")
  [[ -n ${V_REPLAY:-} ]] && miner_cmd+=(--v-replay "$V_REPLAY")
  [[ -n ${SFT_CANDIDATES:-} ]] && miner_cmd+=(--sft-candidates "$SFT_CANDIDATES")
  [[ -n ${TOTAL_CPT_ROWS:-} ]] && miner_cmd+=(--total-cpt-rows "$TOTAL_CPT_ROWS")
fi
"${miner_cmd[@]}" >/dev/null

echo "=== [2/3] B300 Pro Incremental CPT (cosine 5e-6, 50/30/20, DRY_RUN=$DRY_RUN) ==="
launch_flag="--dry-run"
[[ "$DRY_RUN" == "0" ]] && launch_flag="--launch"

bash "$CPT27B_DIR/run_cpt_27b_b300.sh" pro \
  --model "$MODEL" \
  --dataset "$MINER_DIR/bundle_pro_inc/pro_train.jsonl" \
  --val-dataset "$MINER_DIR/bundle_pro_inc/pro_val.jsonl" \
  --output-dir "$CPT_OUT_DIR" \
  "$launch_flag"

echo "=== [3/3] Running Montessori-Instruct Local Influence & Preference Alignment (DRY_RUN=$DRY_RUN) ==="
WORK_DIR="$MON_DIR" DRY_RUN="$DRY_RUN" \
  bash "$SCRIPT_DIR/montessori/run_montessori_pipeline.sh"
