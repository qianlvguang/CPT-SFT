#!/usr/bin/env bash
set -euo pipefail

# Phase C (Pro Same-Source Enhancement & Multi-Round Closed-Loop Evolution) Orchestrator:
#   Step 1: Ingest Flash's exact same raw/midtrain pool (or run Flash data prep in DEMO/DRY_RUN)
#   Step 2: Run 4 programmatic enhancements (`enhance_flash_to_pro.py`):
#             curation -> cross-modal alignment -> WiSE-FT + Thinking-OFF retention gate -> Student DPO/KTO + OPD BSD
#   Step 3: Launch / Dry-Run B300 27B Pro Enhanced Mid-Training (`run_cpt_27b_b300.sh pro`, WSD 6e-6)
#   Step 4: Mine multi-round error clusters (`error_cluster_miner.py` -> `iteration_ledger.json`, monotonic sft_overlap=0)
#             & launch / dry-run `run_cpt_27b_b300.sh pro_inc` (50/30/20 cosine 5e-6)
#   Step 5: Run Montessori-Instruct Local Influence & OPD/Preference Alignment (`montessori/run_montessori_pipeline.sh`)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FLASH_DIR="$TRAINING_DIR/flash"
CPT27B_DIR="$TRAINING_DIR/cpt_27b"

WORK_DIR=${WORK_DIR:-"${TMPDIR:-/tmp}/cpt_sft_pro_pipeline"}
MODEL=${MODEL:-/data/autovision-cfs/autovision-notebook/algo4/models/Qwen3.5-27B}
ROUND_ID=${ROUND_ID:-round_01}
PARENT_CKPT=${PARENT_CKPT:-"$MODEL"}
PREV_LEDGER=${PREV_LEDGER:-""}
DEMO=${DEMO:-0}
DRY_RUN=${DRY_RUN:-1}

mkdir -p "$WORK_DIR"

FLASH_SRC_DIR="$WORK_DIR/step0_flash_source"
PRO_ENH_DIR="$WORK_DIR/step1_pro_enhanced"
PRO_CPT_OUT_DIR="$WORK_DIR/step2_pro_midtrain_ckpt"
MINER_DIR="$WORK_DIR/step3_error_miner"
PRO_INC_OUT_DIR="$WORK_DIR/step4_pro_inc_ckpt"
MON_DIR="$WORK_DIR/step5_montessori"

launch_flag="--dry-run"
[[ "$DRY_RUN" == "0" ]] && launch_flag="--launch"

echo "=== [1/5] Preparing Same-Source Flash Pool for Pro Enhancement ==="
if [[ -z ${FLASH_MIDTRAIN_TRAIN:-} ]]; then
  if [[ "$DEMO" == "1" || "$DRY_RUN" == "1" ]]; then
    mkdir -p "$FLASH_SRC_DIR"
    python3 "$FLASH_DIR/gen_cot_from_qa.py" \
      --input "$FLASH_DIR/examples/qa_kb_v04_example.jsonl" \
      --cot-mode internalized \
      --output-dir "$FLASH_SRC_DIR/cot" >/dev/null
    python3 "$FLASH_DIR/build_v2_image_pipeline.py" \
      --entities "$FLASH_DIR/examples/v2_image_entities_sample.jsonl" \
      --v1-audit "$FLASH_DIR/examples/v2_image_v1_audit_sample.jsonl" \
      --output-dir "$FLASH_SRC_DIR/v2_img" >/dev/null
    python3 "$FLASH_DIR/refactor_v1_v4_pipeline.py" \
      --v1-v4-inputs "$FLASH_DIR/examples/v1_v4_legacy_sample.jsonl" \
      --new-qa-inputs "$FLASH_SRC_DIR/cot/qa_sft_mixed_8_2.jsonl" "$FLASH_DIR/examples/style_sft_example.jsonl" \
      --output-dir "$FLASH_SRC_DIR/refactor" \
      --val-ratio 0.25 >/dev/null
    python3 "$FLASH_DIR/build_mm_wiki_cpt.py" \
      --mode midtrain \
      --inputs \
        "$FLASH_DIR/examples/cpt_mm_6ch_example.jsonl" \
        "$FLASH_SRC_DIR/refactor/cpt_v1_v4_replay.jsonl" \
        "$FLASH_SRC_DIR/v2_img/v2_packed_cpt_docs.jsonl" \
        "$FLASH_SRC_DIR/v2_img/v2_cf_neg_pairs.jsonl" \
        "$FLASH_SRC_DIR/cot/qa_midtrain_prose.jsonl" \
      --exclude-families-from "$FLASH_SRC_DIR/refactor/sft_unified_val.jsonl" \
      --output-dir "$FLASH_SRC_DIR/midtrain" \
      --val-ratio 0.20 \
      --total-rows 28 \
      --max-tokens 8192 \
      --max-row-repeats 2 >/dev/null

    FLASH_MIDTRAIN_TRAIN="$FLASH_SRC_DIR/midtrain/midtrain_train.jsonl"
    FLASH_MIDTRAIN_VAL="$FLASH_SRC_DIR/midtrain/midtrain_val.jsonl"
    COT_PASSED="$FLASH_SRC_DIR/cot/qa_cot_passed.jsonl"
    COT_REJECTED="$FLASH_SRC_DIR/cot/qa_dpo_rejected_cots.jsonl"
    V2_PACKED_DOCS="$FLASH_SRC_DIR/v2_img/v2_packed_cpt_docs.jsonl"
  else
    echo "Set FLASH_MIDTRAIN_TRAIN and FLASH_MIDTRAIN_VAL to share Flash's raw/midtrain pool with Pro (or pass DEMO=1)." >&2
    exit 2
  fi
fi

echo "=== [2/5] Running 4 Programmatic Pro Upgrades (curation, cross-modal alignment, WiSE-FT/Think-OFF gate, OPD/DPO) ==="
enh_cmd=(
  python3 "$SCRIPT_DIR/enhance_flash_to_pro.py"
  --flash-midtrain-train "$FLASH_MIDTRAIN_TRAIN"
  --flash-midtrain-val "${FLASH_MIDTRAIN_VAL:-$FLASH_SRC_DIR/midtrain/midtrain_val.jsonl}"
  --output-dir "$PRO_ENH_DIR"
)
[[ -n ${COT_PASSED:-} ]] && enh_cmd+=(--cot-passed "$COT_PASSED")
[[ -n ${COT_REJECTED:-} ]] && enh_cmd+=(--cot-rejected "$COT_REJECTED")
[[ -n ${V2_PACKED_DOCS:-} ]] && enh_cmd+=(--v2-packed-docs "$V2_PACKED_DOCS")
[[ -n ${CANARY_METRICS:-} ]] && enh_cmd+=(--canary-metrics "$CANARY_METRICS")
"${enh_cmd[@]}" >/dev/null

echo "=== [3/5] B300 Pro Enhanced Mid-Training (WSD linear 6e-6, DRY_RUN=$DRY_RUN) ==="
bash "$CPT27B_DIR/run_cpt_27b_b300.sh" pro \
  --model "$MODEL" \
  --dataset "$PRO_ENH_DIR/bundle_pro/pro_midtrain_train.jsonl" \
  --val-dataset "$PRO_ENH_DIR/bundle_pro/pro_midtrain_val.jsonl" \
  --output-dir "$PRO_CPT_OUT_DIR" \
  "$launch_flag"

echo "=== [4/5] Mining Multi-Round Error Clusters + Monotonic sft_overlap=0 Ledger ($ROUND_ID) & Incremental CPT ==="
miner_cmd=(
  python3 "$SCRIPT_DIR/error_cluster_miner.py"
  --output-dir "$MINER_DIR"
  --round-id "$ROUND_ID"
  --parent-ckpt "$PARENT_CKPT"
)
if [[ -n "$PREV_LEDGER" ]]; then
  miner_cmd+=(--prev-ledger "$PREV_LEDGER")
fi
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

bash "$CPT27B_DIR/run_cpt_27b_b300.sh" pro_inc \
  --model "$PRO_CPT_OUT_DIR/latest" \
  --dataset "$MINER_DIR/bundle_pro_inc/pro_train.jsonl" \
  --val-dataset "$MINER_DIR/bundle_pro_inc/pro_val.jsonl" \
  --output-dir "$PRO_INC_OUT_DIR" \
  "$launch_flag"

echo "=== [5/5] Running Montessori-Instruct & OPD Think-ON->OFF Context Distillation (DRY_RUN=$DRY_RUN) ==="
WORK_DIR="$MON_DIR" DRY_RUN="$DRY_RUN" \
  bash "$SCRIPT_DIR/montessori/run_montessori_pipeline.sh"
