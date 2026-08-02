#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
CONFIG="${STAGE_B_V2_CONFIG:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v2/stage_b_v2_causal_15shard_v1.env}"
# shellcheck source=/dev/null
source "${CONFIG}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

export TMPDIR="${USER_ROOT}/tmp"
export XDG_CACHE_HOME="${USER_ROOT}/cache/xdg"
export HF_HOME="${USER_ROOT}/cache/huggingface"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${REPO_ROOT}/reports/simul_uniss_subsecond_v2"

log() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }

wait_for_idle_gpus() {
  local consecutive=0
  while (( consecutive < 2 )); do
    local active
    active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l)"
    if [[ "${active}" -eq 0 ]]; then
      consecutive=$((consecutive + 1))
      log "GPU idle check ${consecutive}/2"
    else
      consecutive=0
      log "Waiting: ${active} external GPU compute processes are still active"
    fi
    if (( consecutive < 2 )); then
      sleep 30
    fi
  done
}

run_sidecar() {
  local mode="$1"
  local manifest="$2"
  local output="$3"
  local limit="$4"
  log "Stage-A-v3 ${mode}: ${output}"
  MODE="${mode}" WORLD_SIZE=8 LIMIT_RECORDS="${limit}" AUDIO_WORKERS=4 \
    TEACHER_BATCH_SIZE="$([[ "${mode}" == "clone" ]] && echo 32 || echo 8)" \
    RECORDS_PER_SHARD=512 MANIFEST="${manifest}" OUTPUT_ROOT="${output}" \
    bash "${REPO_ROOT}/scripts/simul_uniss_subsecond_v2/run_stage_a_v3_sidecar.sh"
}

wait_for_idle_gpus

run_sidecar clone "${SOURCE_VALID_MANIFEST}" "${CLONE_VALID_ROOT}" 0
run_sidecar prefix80 "${SOURCE_VALID_MANIFEST}" "${PREFIX_VALID_ROOT}" 0
run_sidecar clone "${SOURCE_TRAIN_MANIFEST}" "${CLONE_TRAIN_ROOT}" 0
run_sidecar prefix80 "${SOURCE_TRAIN_MANIFEST}" "${PREFIX_TRAIN_ROOT}" 100000

log "Starting eight-GPU Stage-B-v2 clone pretraining"
bash "${REPO_ROOT}/scripts/simul_uniss_subsecond_v2/train_stage_b_v2_causal.sh" clone

log "Starting eight-GPU Stage-B-v2 prefix-80 fine-tuning"
bash "${REPO_ROOT}/scripts/simul_uniss_subsecond_v2/train_stage_b_v2_causal.sh" prefix80

log "Validating clone-pretrained checkpoint"
MODE=clone GPU=0 bash "${REPO_ROOT}/scripts/simul_uniss_subsecond_v2/validate_stage_b_v2_causal.sh"

log "Validating prefix-fine-tuned checkpoint"
MODE=prefix80 GPU=0 bash "${REPO_ROOT}/scripts/simul_uniss_subsecond_v2/validate_stage_b_v2_causal.sh"

log "Running frozen-Phase3 sensitivity for repaired Student"
SAMPLES=8 AUDIO_WORKERS=8 GPU=0 STUDENT_STREAM_NAME=student_v2_prefix80 \
  STUDENT_CHECKPOINT="${STAGE_B_V2_PREFIX_ROOT}/best.pt" \
  OUTPUT="${REPO_ROOT}/reports/simul_uniss_subsecond_v2/stage_b_v2_prefix80_phase3_sensitivity.json" \
  bash "${REPO_ROOT}/scripts/simul_uniss_subsecond_v2/run_phase3_token_stream_sensitivity.sh"

log "Stage-B-v2 repair pipeline complete"
