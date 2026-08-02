#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
# shellcheck source=/dev/null
source "${USER_ROOT}/env_recovery/uniss-train-20260721/activate_uniss.sh"

MODE="${MODE:-prefix80}"
GPU="${GPU:-0}"
if [[ "${MODE}" == "clone" ]]; then
  CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/checkpoints/simul_uniss_subsecond_v2/stage_b_v2_clone_pretrain_15shard_v1/best.pt}"
  SIDECAR="${SIDECAR:-${REPO_ROOT}/data/processed/simul_uniss_subsecond_v2/stage_a_v3_clone_valid_v1/manifest.jsonl}"
  OUTPUT="${OUTPUT:-${REPO_ROOT}/reports/simul_uniss_subsecond_v2/stage_b_v2_clone_validation.json}"
else
  CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/checkpoints/simul_uniss_subsecond_v2/stage_b_v2_prefix80_finetune_100k_v1/best.pt}"
  SIDECAR="${SIDECAR:-${REPO_ROOT}/data/processed/simul_uniss_subsecond_v2/stage_a_v3_prefix80_valid_v1/manifest.jsonl}"
  OUTPUT="${OUTPUT:-${REPO_ROOT}/reports/simul_uniss_subsecond_v2/stage_b_v2_prefix80_validation.json}"
fi
SOURCE="${SOURCE:-${REPO_ROOT}/data/processed/simul_uniss_subsecond_v2/formal_15shard_v1/stage_a_formal/formal_valid_manifest.jsonl}"
mkdir -p "$(dirname "${OUTPUT}")"

CUDA_VISIBLE_DEVICES="${GPU}" python -m \
  training.simul_uniss.subsecond_v2.validate_stage_b_v2 \
  --checkpoint "${CHECKPOINT}" \
  --sidecar-manifest "${SIDECAR}" \
  --source-manifest "${SOURCE}" \
  --output "${OUTPUT}" \
  --device cuda:0 --samples 128 --latency-samples 16 \
  --chunk-ms 160 --right-context-ms 80 \
  --minimum-target-agreement 0.70 \
  --maximum-rtf 0.25 \
  --minimum-correct-stable-coverage 0.90
