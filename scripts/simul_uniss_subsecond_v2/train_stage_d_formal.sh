#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${STAGE_D_CONFIG:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v2/stage_d_formal_15shard_v1.env}"
MODE="${1:-formal}"
[[ "${MODE}" == "formal" || "${MODE}" == "smoke" ]] || { echo "mode must be formal or smoke" >&2; exit 2; }

if [[ "${MODE}" == "smoke" ]]; then
  export PACKED_TRAIN="${REPO_ROOT}/data/processed/simul_uniss_subsecond_v2/benchmark/stage_d_formal_train_112/packed.jsonl"
  export VALID_PACKED_INTERLEAVED="${REPO_ROOT}/data/processed/simul_uniss_subsecond_v2/benchmark/stage_d_formal_valid_13/packed.jsonl"
  export STAGE_D_SAVE_ROOT="${STAGE_D_SAVE_ROOT:-${REPO_ROOT}/checkpoints/simul_uniss_subsecond_v2/smoke/stage_d_formal_launcher}"
  export STAGE_D_TENSORBOARD_DIR="${STAGE_D_TENSORBOARD_DIR:-${REPO_ROOT}/runs/simul_uniss_subsecond_v2/smoke/stage_d_formal_launcher/tensorboard}"
  exec "${REPO_ROOT}/scripts/simul_uniss/train_qwen_stage.sh" --config "${CONFIG}" --stage interleaved --smoke
fi
exec "${REPO_ROOT}/scripts/simul_uniss/train_qwen_stage.sh" --config "${CONFIG}" --stage interleaved
