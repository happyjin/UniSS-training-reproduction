#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 PHASE3_RETENTION_ROOT" >&2
  exit 2
fi

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EVAL_DIR}/../../.." && pwd)"
USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
ROOT="$(realpath "$1")"
RESULTS="${ROOT}/aggregate/results.jsonl"
METRICS="${ROOT}/aggregate/metrics"
EVAL_ENV="${EVAL_ENV:-${USER_ROOT}/conda_envs/uniss-eval}"
PYTHON="${PYTHON:-${EVAL_ENV}/bin/python}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
OBJECTIVE_RUNNER="${REPO_ROOT}/experiments/evaluation/uniss_full198_phase2_phase3/run_objective_metrics.sh"

for required in "${RESULTS}" "${PYTHON}" "${OBJECTIVE_RUNNER}"; do
  [[ -e "${required}" ]] || { echo "Missing retention metric input: ${required}" >&2; exit 1; }
done
[[ ! -e "${METRICS}/complete.json" ]] || {
  echo "Retention metrics already complete at ${METRICS}" >&2
  exit 1
}

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${EVAL_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-${USER_ROOT}/cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${USER_ROOT}/cache/modelscope}"
export TORCH_HOME="${TORCH_HOME:-${USER_ROOT}/cache/torch}"
export ENV_ROOT="${EVAL_ENV}"
export EVAL_GPU_LIST="${GPU_LIST}"
export METRIC_NUM_GPUS=8
export AUTOPCP_ENCODER="${AUTOPCP_ENCODER:-${USER_ROOT}/evaluation_models/wav2vec2-large-xlsr-53}"

mkdir -p "${METRICS}"
"${PYTHON}" -m evaluation.text_metrics \
  --input "${RESULTS}" --output "${METRICS}/text_bleu.json" \
  --hypothesis-field generated_translation --reference-field translation_ref \
  --score-empty-hypotheses
"${PYTHON}" -m evaluation.slc_metrics --input "${RESULTS}" --output-dir "${METRICS}"

# Reuse the established 8-GPU ASR, Speech-BLEU, UTMOS and AutoPCP protocol.
# The aggregate root already contains results.jsonl and the runner writes into
# its metrics/ child, which is exactly ${METRICS}.
"${OBJECTIVE_RUNNER}" "${ROOT}/aggregate"

"${PYTHON}" - "${METRICS}" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
required = ["text_bleu.json", "speech_bleu.json", "slc.json", "utmos.json", "autopcp.json"]
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise SystemExit(f"retention metrics missing: {missing}")
(root / "complete.json").write_text(
    json.dumps({"status": "complete", "artifacts": required}, indent=2) + "\n",
    encoding="utf-8",
)
PY

printf '%s\n' "${METRICS}"
