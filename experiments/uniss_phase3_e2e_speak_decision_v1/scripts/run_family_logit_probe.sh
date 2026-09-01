#!/usr/bin/env bash
# Runs the family-logit probe over the frozen fixed-16 selection on 8 GPUs.
# The worker argument list is the gate runner's, with only the module swapped:
# see tests/test_family_logit_probe.py, which diffs the two.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/../../uniss_phase3_v4_e2e_simuls2st_pilot15_v1" && pwd)
source "${BASE_EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

: "${PROBE_ID:?PROBE_ID is required}"
: "${CANDIDATE_HF:?CANDIDATE_HF is required}"

FORMAL_DATA_RUN_ID=${FORMAL_DATA_RUN_ID:-formal_gold_20260818T090515Z}
BASELINE_GATE=${BASELINE_GATE:-${REPO_ROOT}/reports/${EXPERIMENT_NAME}/${FORMAL_DATA_RUN_ID}/free_running_gates/stage2_paced_m1200_iter0002264_20260831T180448Z}
SELECTION=${SELECTION:-${BASELINE_GATE}/SELECTION.json}
GOLD=${GOLD:-${REPO_ROOT}/data/processed/${EXPERIMENT_NAME}/${FORMAL_DATA_RUN_ID}/source_events/valid_gold_trajectories.jsonl}
BICODEC_MODEL=${BICODEC_MODEL:-${REPO_ROOT}/pretrained_models/UniSS/bicodec}
NUM_WORKERS=${NUM_WORKERS:-8}
MAX_S2S_SEMANTIC_TOKENS=${MAX_S2S_SEMANTIC_TOKENS:-384}
PROBE_ROOT=${PROBE_ROOT:-${REPO_ROOT}/reports/uniss_phase3_e2e_speak_decision_v1/family_logit_probe/${PROBE_ID}}
PROBE_MODULE=${PROBE_MODULE:-experiments.uniss_phase3_e2e_speak_decision_v1.diagnostics.family_logit_probe}
FAMILY_MT_BIAS=${FAMILY_MT_BIAS:-0}
CONTINUE_WRITE_BIAS=${CONTINUE_WRITE_BIAS:-0}

CANDIDATE_FINGERPRINT=${CANDIDATE_FINGERPRINT:?CANDIDATE_FINGERPRINT is required}
CANDIDATE_SHA=$("${PYTHON_BIN}" -c '
import json,sys
x=json.load(open(sys.argv[1]))
v=x["checkpoints"]["candidate_hf"]
assert v["path"] == str(__import__("pathlib").Path(sys.argv[2]).resolve())
print(v["sha256"])
' "${CANDIDATE_FINGERPRINT}" "${CANDIDATE_HF}")
[[ "${#CANDIDATE_SHA}" == "64" ]] || { echo "malformed candidate HF fingerprint" >&2; exit 5; }

for path in "${SELECTION}" "${GOLD}" "${CANDIDATE_HF}/model.safetensors" "${CANDIDATE_FINGERPRINT}"; do
  [[ -e "${path}" ]] || { echo "missing probe input: ${path}" >&2; exit 2; }
done
[[ ! -e "${PROBE_ROOT}" ]] || { echo "refusing to overwrite ${PROBE_ROOT}" >&2; exit 3; }
mkdir -p "${PROBE_ROOT}/probes" "${PROBE_ROOT}/workers" "${PROBE_ROOT}/audio" "${PROBE_ROOT}/logs"

NVIDIA_LIBRARY_ROOT="$(dirname "${PYTHON_BIN}")/../lib/python3.12/site-packages/nvidia"
NVIDIA_LIBRARY_PATH=
if [[ -d "${NVIDIA_LIBRARY_ROOT}" ]]; then
  NVIDIA_LIBRARY_PATH=$(find "${NVIDIA_LIBRARY_ROOT}" -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd: -)
fi
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib:/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/targets/x86_64-linux/lib:$(dirname "${PYTHON_BIN}")/../lib:${LD_LIBRARY_PATH:-}${NVIDIA_LIBRARY_PATH:+:${NVIDIA_LIBRARY_PATH}}"
export HF_HOME="${USER_ROOT}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TMPDIR="${USER_ROOT}/tmp"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

pids=()
for worker in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=${worker} \
  UNISS_E2E_FAMILY_PROBE_OUTPUT="${PROBE_ROOT}/probes/worker_$(printf '%02d' "${worker}").jsonl" \
  UNISS_E2E_FAMILY_MT_BIAS="${FAMILY_MT_BIAS}" \
  UNISS_E2E_CONTINUE_WRITE_BIAS="${CONTINUE_WRITE_BIAS}" \
  "${PYTHON_BIN}" -m \
    "${PROBE_MODULE}" \
    --selection "${SELECTION}" \
    --gold "${GOLD}" \
    --candidate-hf "${CANDIDATE_HF}" \
    --phase3-hf "${PHASE3_HF_MODEL}" \
    --v1-checkpoint "${V1_CHECKPOINT}" \
    --whispervq-model "${WHISPERVQ_MODEL}" \
    --bicodec-model "${BICODEC_MODEL}" \
    --candidate-hf-sha256 "${CANDIDATE_SHA}" \
    --worker-index "${worker}" \
    --num-workers "${NUM_WORKERS}" \
    --max-s2s-semantic-tokens "${MAX_S2S_SEMANTIC_TOKENS}" \
    --report "${PROBE_ROOT}/workers/worker_$(printf '%02d' "${worker}").json" \
    --audio-dir "${PROBE_ROOT}/audio/worker_$(printf '%02d' "${worker}")" \
    --device cuda:0 > "${PROBE_ROOT}/logs/worker_$(printf '%02d' "${worker}").log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 )) || { echo "a probe worker failed; see ${PROBE_ROOT}/logs" >&2; exit 4; }
"${PYTHON_BIN}" - <<PY
import json,pathlib
pathlib.Path("${PROBE_ROOT}/PROBE_CONFIG.json").write_text(json.dumps({
  "probe_id": "${PROBE_ID}", "module": "${PROBE_MODULE}",
  "family_mt_bias": float("${FAMILY_MT_BIAS}"), "continue_write_bias": float("${CONTINUE_WRITE_BIAS}"),
  "candidate_hf": "${CANDIDATE_HF}", "selection": "${SELECTION}",
  "max_s2s_semantic_tokens": ${MAX_S2S_SEMANTIC_TOKENS},
  "mt_holdback": "${UNISS_E2E_MT_HOLDBACK:-}", "pace_margin_ms": "${UNISS_E2E_SEMANTIC_PACE_MARGIN_MS:-}"
}, indent=1, sort_keys=True))
PY
echo "probe_root=${PROBE_ROOT}"
