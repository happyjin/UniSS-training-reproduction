#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${STAGE_B_LATENT_CONFIG:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v2/stage_b_latent_formal_15shard_v1.env}"
# shellcheck source=/dev/null
source "${CONFIG}"

mkdir -p "${STAGE_B_LATENT_LOG_ROOT}"
pipeline_log="${STAGE_B_LATENT_LOG_ROOT}/stage_b_latent_pipeline.log"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "${pipeline_log}"; }

while [[ ! -f "${STAGE_A_COMPLETE_MARKER}" ]]; do
  log "waiting for corrected formal Stage A marker"
  sleep 30
done

while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; do
  log "waiting for all eight GPUs to become idle"
  sleep 30
done

tensorboard_session="uniss_stage_b_latent_tensorboard"
tmux has-session -t "${tensorboard_session}" 2>/dev/null || tmux new-session -d \
  -s "${tensorboard_session}" \
  "cd '${REPO_ROOT}' && bash scripts/simul_uniss_subsecond_v2/start_stage_b_latent_tensorboard.sh"

log "running corrected latent Stage-B launcher smoke"
bash scripts/simul_uniss_subsecond_v2/train_stage_b_latent_formal.sh smoke \
  2>&1 | tee -a "${pipeline_log}"
log "starting corrected latent Stage-B 15-shard eight-GPU training"
STAGE_B_LATENT_RESUME=1 bash scripts/simul_uniss_subsecond_v2/train_stage_b_latent_formal.sh formal \
  2>&1 | tee -a "${pipeline_log}"
log "corrected latent Stage-B training and quality gate complete"
