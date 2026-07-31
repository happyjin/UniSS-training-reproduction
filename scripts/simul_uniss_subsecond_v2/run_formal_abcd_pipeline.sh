#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FORMAL_CONFIG="${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v2/formal_15shard.env"
source "${FORMAL_CONFIG}"
PIPELINE_LOG="${LOG_ROOT}/formal_abcd_pipeline.log"
expected_parts=$((SHARD_COUNT * CHUNKS_PER_SHARD))
mkdir -p "${LOG_ROOT}"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "${PIPELINE_LOG}"; }
marker_count() { find "$1" -name "$2" 2>/dev/null | wc -l; }
start_tensorboard() {
  local stage="$1" session="uniss_formal_${1}_tensorboard"
  tmux has-session -t "${session}" 2>/dev/null || tmux new-session -d -s "${session}" \
    "cd '${REPO_ROOT}' && scripts/simul_uniss_subsecond_v2/start_formal_tensorboard.sh '${stage}'"
}

while [[ "$(marker_count "${A45_ROOT}" STAGE_A_A45_COMPLETE.json)" -lt "${expected_parts}" ]]; do
  count="$(marker_count "${A45_ROOT}" STAGE_A_A45_COMPLETE.json)"
  log "waiting for formal A4/A5: ${count}/${expected_parts} parts"
  if ! pgrep -f 'training.simul_uniss.subsecond_v2.prepare_a45' >/dev/null; then
    log "A4/A5 workers are absent; resuming incomplete parts"
    bash "${REPO_ROOT}/scripts/simul_uniss_subsecond_v2/run_formal_stage_a_15shard.sh" "${FORMAL_CONFIG}" a45 \
      2>&1 | tee -a "${PIPELINE_LOG}"
  fi
  sleep 30
done

log "formal A4/A5 complete; starting A6/A8 bilingual alignment"
bash "${REPO_ROOT}/scripts/simul_uniss_subsecond_v2/run_formal_stage_a_15shard.sh" "${FORMAL_CONFIG}" a68 \
  2>&1 | tee -a "${PIPELINE_LOG}"
log "assembling deterministic formal train/valid manifests"
bash "${REPO_ROOT}/scripts/simul_uniss_subsecond_v2/run_formal_stage_a_15shard.sh" "${FORMAL_CONFIG}" assemble \
  2>&1 | tee -a "${PIPELINE_LOG}"

start_tensorboard stage_b
log "running corrected Stage-B launcher smoke"
bash "${REPO_ROOT}/scripts/simul_uniss_subsecond_v2/train_stage_b_formal.sh" smoke 2>&1 | tee -a "${PIPELINE_LOG}"
log "starting corrected Stage-B eight-GPU training"
STAGE_B_RESUME=1 bash "${REPO_ROOT}/scripts/simul_uniss_subsecond_v2/train_stage_b_formal.sh" formal \
  2>&1 | tee -a "${PIPELINE_LOG}"

start_tensorboard stage_c
log "running formal Stage-C launcher smoke"
bash "${REPO_ROOT}/scripts/simul_uniss_subsecond_v2/train_stage_c_formal.sh" smoke 2>&1 | tee -a "${PIPELINE_LOG}"
log "starting formal Stage-C eight-GPU training and calibration"
STAGE_C_RESUME=1 bash "${REPO_ROOT}/scripts/simul_uniss_subsecond_v2/train_stage_c_formal.sh" formal \
  2>&1 | tee -a "${PIPELINE_LOG}"
python -m training.simul_uniss.subsecond_v2.validate_stage_c \
  --calibration "${STAGE_C_ROOT}/calibration.json" \
  --output "${STAGE_C_ROOT}/STAGE_C_QUALITY_GATE.json" \
  2>&1 | tee -a "${PIPELINE_LOG}"

log "preparing formal Stage-D Micro-WRITE plus 30% Phase3 replay"
bash "${REPO_ROOT}/scripts/simul_uniss_subsecond_v2/prepare_stage_d_formal.sh" 2>&1 | tee -a "${PIPELINE_LOG}"
start_tensorboard stage_d
log "running formal Stage-D Megatron smoke"
bash "${REPO_ROOT}/scripts/simul_uniss_subsecond_v2/train_stage_d_formal.sh" smoke 2>&1 | tee -a "${PIPELINE_LOG}"
log "starting formal Stage-D eight-GPU training"
bash "${REPO_ROOT}/scripts/simul_uniss_subsecond_v2/train_stage_d_formal.sh" formal 2>&1 | tee -a "${PIPELINE_LOG}"
log "formal Stage A-D pipeline complete"
