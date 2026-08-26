#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${EXPERIMENT_ROOT}/../.." && pwd)
source "${EXPERIMENT_ROOT}/config.env"

TRAIN_RUN=${1:-formal_train64_g4_v1}
VALID_RUN=${2:-formal_valid16_g4_v1}
ATTRIBUTION_RUN=${3:-reference_attribution_valid16_v1}
PACK_ID=${4:-formal_64x4_train_16x4_valid_v1}
TRAINING_RUN=${5:-episode_grpo_formal_8gpu_v1}
TRAIN_MERGED=${REPO_ROOT}/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/${TRAIN_RUN}/ROLLOUT_MERGED.json
VALID_MERGED=${REPO_ROOT}/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/${VALID_RUN}/ROLLOUT_MERGED.json
ATTR_ROOT=${REPO_ROOT}/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/${ATTRIBUTION_RUN}
ATTR_MERGED=${ATTR_ROOT}/ATTRIBUTION_MERGED.json
ATTR_REPORT=${REPO_ROOT}/reports/uniss_phasea_stateful_longepisode_rl_v1/attribution/${ATTRIBUTION_RUN}/REPORT.zh-CN.md

echo "waiting for train/valid rollout merges"
while [[ ! -f "${TRAIN_MERGED}" || ! -f "${VALID_MERGED}" ]]; do
  sleep 15
done
echo "rollouts complete"

if [[ ! -f "${ATTR_MERGED}" ]]; then
  if [[ ! -d "${ATTR_ROOT}/workers" ]]; then
    bash "${SCRIPT_DIR}/run_reference_attribution_8gpu.sh" "${ATTRIBUTION_RUN}"
  fi
  "${PYTHON}" "${EXPERIMENT_ROOT}/evaluation/merge_attribution.py" \
    --workers-root "${ATTR_ROOT}/workers" \
    --expected-workers 8 \
    --runtime-rollout "${VALID_MERGED}" \
    --output "${ATTR_MERGED}"
fi
if [[ ! -f "${ATTR_REPORT}" ]]; then
  "${PYTHON}" "${EXPERIMENT_ROOT}/evaluation/write_attribution_report.py" \
    --attribution "${ATTR_MERGED}" \
    --output "${ATTR_REPORT}"
fi

PACK_ROOT=${REPO_ROOT}/data/processed/uniss_phasea_stateful_longepisode_rl_v1/${PACK_ID}
if [[ ! -f "${PACK_ROOT}/train_packs.jsonl" || ! -f "${PACK_ROOT}/valid_packs.jsonl" ]]; then
  bash "${SCRIPT_DIR}/pack_formal_rollouts.sh" "${TRAIN_RUN}" "${VALID_RUN}" "${PACK_ID}"
fi

bash "${SCRIPT_DIR}/run_megatron_formal_8gpu.sh" "${TRAINING_RUN}" "${PACK_ID}" 3
