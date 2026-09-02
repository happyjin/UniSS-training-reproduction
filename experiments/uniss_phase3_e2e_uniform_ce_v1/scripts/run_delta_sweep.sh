#!/usr/bin/env bash
# Sweep the inference-side continue bias on this run's checkpoint.
#
# The chain already measured the two endpoints and they straddle the target
# text length band from opposite sides: delta=0 gives a median ratio of 0.755
# (under-generating, natural_eos 0.500) and delta=5 gives 1.528 (over-
# generating, natural_eos 1.000, coverage 0.997).  The band is [0.9, 1.2], so
# it lies between them and the bias alone can reach it.  Nothing here trains;
# every delta reuses the established probe runner unchanged, so this cannot
# disturb the training lineage or any existing experiment.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
OWN_NAME=uniss_phase3_e2e_uniform_ce_v1

TRAIN_RUN_ID=${TRAIN_RUN_ID:-uniform_ce_20260902T041721Z}
GATE_RUN_ID=${GATE_RUN_ID:-uniform_ce_gate_20260902T042621Z}
BASE_EXPERIMENT=uniss_phase3_v4_e2e_simuls2st_pilot15_v1
FORMAL_DATA_RUN_ID=${FORMAL_DATA_RUN_ID:-formal_gold_20260818T090515Z}

GATE_ROOT=${GATE_ROOT:-${REPO_ROOT}/reports/${BASE_EXPERIMENT}/${FORMAL_DATA_RUN_ID}/free_running_gates/${GATE_RUN_ID}}
REPORT_ROOT=${REPORT_ROOT:-${REPO_ROOT}/reports/${OWN_NAME}/${TRAIN_RUN_ID}/delta_sweep_${GATE_RUN_ID}}
CANDIDATE_HF=${CANDIDATE_HF:-${REPO_ROOT}/checkpoints/exported_hf/${OWN_NAME}_${TRAIN_RUN_ID}_iter_0001132_hf}

DELTAS=${DELTAS:-"1 2 3 4"}
SEMANTIC_REPETITION=${SEMANTIC_REPETITION:-1.1}
SEMANTIC_WINDOW=${SEMANTIC_WINDOW:-8}
MAX_S2S_SEMANTIC_TOKENS=${MAX_S2S_SEMANTIC_TOKENS:-384}
HOLDBACK=${HOLDBACK:-2}
PACE_MARGIN_MS=${PACE_MARGIN_MS:-1200}
PYTHON_BIN=${PYTHON_BIN:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python}

mkdir -p "${REPORT_ROOT}"
echo "report_root=${REPORT_ROOT}"
echo "candidate_hf=${CANDIDATE_HF}"
echo "deltas=${DELTAS}"

SUMMARY_ARGS=()
for delta in ${DELTAS}; do
  probe_root="${REPORT_ROOT}/delta${delta}"
  if [[ -s "${probe_root}/PROBE_CONFIG.json" ]]; then
    echo "step=probe delta=${delta} (cached)"
  else
    echo "step=probe delta=${delta}"
    PROBE_ROOT="${probe_root}" \
    PROBE_ID="${GATE_RUN_ID}_delta${delta}" \
    PROBE_MODULE=experiments.uniss_phase3_e2e_speak_decision_v1.diagnostics.biased_family_probe \
    FAMILY_MT_BIAS=0 CONTINUE_WRITE_BIAS="${delta}" \
    UNISS_E2E_SEMANTIC_REPETITION_PENALTY="${SEMANTIC_REPETITION}" \
    UNISS_E2E_SEMANTIC_REPETITION_WINDOW="${SEMANTIC_WINDOW}" \
    SELECTION="${GATE_ROOT}/SELECTION.json" \
    CANDIDATE_HF="${CANDIDATE_HF}" \
    CANDIDATE_FINGERPRINT="${GATE_ROOT}/CANDIDATE_HF_FINGERPRINT.json" \
    MAX_S2S_SEMANTIC_TOKENS="${MAX_S2S_SEMANTIC_TOKENS}" \
    UNISS_E2E_MT_HOLDBACK="${HOLDBACK}" \
    UNISS_E2E_SEMANTIC_PACE=1 UNISS_E2E_SEMANTIC_PACE_MARGIN_MS="${PACE_MARGIN_MS}" \
      bash "${REPO_ROOT}/experiments/uniss_phase3_e2e_speak_decision_v1/scripts/run_family_logit_probe.sh"
  fi
  SUMMARY_ARGS+=(--run "delta${delta}=${probe_root}")
done

echo "step=summary"
cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" -m \
  experiments.uniss_phase3_e2e_uniform_ce_v1.evaluation.deploy_summary \
  "${SUMMARY_ARGS[@]}" \
  --run "delta5_from_chain=${REPO_ROOT}/reports/${OWN_NAME}/${TRAIN_RUN_ID}/gate_${GATE_RUN_ID}/deploy_probe" \
  --output "${REPORT_ROOT}/DEPLOY_SUMMARY.json" \
  | tee "${REPORT_ROOT}/DEPLOY_SUMMARY.txt"

echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "summary=${REPORT_ROOT}/DEPLOY_SUMMARY.json"
