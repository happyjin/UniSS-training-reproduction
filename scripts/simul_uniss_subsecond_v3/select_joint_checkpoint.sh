#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${STAGE_B_V3_CONFIG:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v3/stage_b_v3_balanced_hidden_15shard_v1.env}"
# shellcheck source=/dev/null
source "${CONFIG}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

mkdir -p "${V3_PHASE3_RESULT_ROOT}"
if [[ ! -s "${V3_CHECKPOINT_ROOT}/CANDIDATES.json" ]]; then
  echo "Missing ${V3_CHECKPOINT_ROOT}/CANDIDATES.json" >&2
  exit 1
fi
if [[ ! -s "${V3_PHASE3_EVAL_MANIFEST}" ]]; then
  echo "Missing ${V3_PHASE3_EVAL_MANIFEST}" >&2
  exit 1
fi

mapfile -t candidates < <(
  python - "${V3_CHECKPOINT_ROOT}/CANDIDATES.json" <<'PY'
import json
import sys

for row in json.load(open(sys.argv[1], encoding="utf-8"))["candidates"]:
    print(row["checkpoint"])
PY
)

for checkpoint in "${candidates[@]}"; do
  stem="$(basename "${checkpoint}" .pt)"
  output="${V3_PHASE3_RESULT_ROOT}/${stem}.json"
  stream_name="candidate_${stem}"
  if [[ -s "${output}" ]]; then
    echo "Reusing ${output}"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${V3_SELECTION_GPU}" python -m \
    training.simul_uniss.subsecond_v2.evaluate_phase3_token_streams \
    --manifest "${V3_PHASE3_EVAL_MANIFEST}" \
    --whispervq-model "${WHISPERVQ_MODEL}" \
    --student-checkpoint "${checkpoint}" \
    --student-stream-name "${stream_name}" \
    --phase3-model "${V3_PHASE3_MODEL}" \
    --output "${output}" \
    --device cuda:0 \
    --samples "${V3_PHASE3_EVAL_SAMPLES}" \
    --audio-workers "${V3_PHASE3_AUDIO_WORKERS}" \
    --chunk-ms 160 --lookahead-ms 80 \
    --streaming-clone-chunk-ms 160 \
    --streaming-clone-right-context-ms 80 \
    --max-audio-seconds 8 --max-new-tokens 192
done

python -m training.simul_uniss.subsecond_v3.select_joint_checkpoint \
  --candidates "${V3_CHECKPOINT_ROOT}/CANDIDATES.json" \
  --phase3-result-dir "${V3_PHASE3_RESULT_ROOT}" \
  --output-dir "${V3_CHECKPOINT_ROOT}"
