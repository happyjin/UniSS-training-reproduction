#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/uniss_phase3_true_subsecond_deadline_full198_v1/config.env"
SESSION="${PIPELINE_TMUX_SESSION:-uniss_true_subsecond_full198_pipeline}"
PIPELINE_LOG="${PIPELINE_LOG:-${REPO_ROOT}/logs/uniss_true_subsecond_full198_pipeline.log}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "pipeline already running: ${SESSION}"
  exit 0
fi
command=(bash -lc '
  set -euo pipefail
  repo="$1"
  source "$repo/experiments/uniss_phase3_true_subsecond_deadline_full198_v1/config.env"
  log() { printf "[%s] %s\n" "$(date -u +%FT%TZ)" "$*"; }
  bash "$repo/experiments/uniss_phase3_true_subsecond_deadline_full198_v1/scripts/launch_pack_full198_tmux.sh"
  while true; do
    cache_count="$(find "$CACHE_ROOT" -mindepth 2 -maxdepth 2 -name PART_COMPLETE.json | wc -l)"
    pack_count="$(find "$PACKED_ROOT/parts" -mindepth 2 -maxdepth 2 -name PACK_COMPLETE.json | wc -l)"
    log "waiting for cache/pack: cache=${cache_count}/198 pack=${pack_count}/198"
    [[ "$cache_count" == 198 && "$pack_count" == 198 ]] && break
    sleep 60
  done
  log "assembling immutable full198 trajectory JSONL and uint64 offsets"
  "$PYTHON" -m experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.assemble_trajectory_packs \
    --parts-root "$PACKED_ROOT/parts" \
    --output "$TRAJECTORY_PACKED" \
    --offsets "$TRAJECTORY_OFFSETS" \
    --marker "$PACKED_ROOT/ASSEMBLY_COMPLETE.json" \
    --shard-count 198 --seq-length 18000
  if [[ ! -f "$VALID_REPLAY_OFFSETS" ]]; then
    log "building immutable Phase3 dev replay offset index"
    "$PYTHON" -m training.phase3_whisper_streamspeech_joint.build_replay_index \
      --source "$VALID_REPLAY_PACKED" --output "$VALID_REPLAY_OFFSETS" --progress-interval 0
  fi
  if [[ ! -f "$PHASE3_FINGERPRINT" ]]; then
    log "building Phase3 native/HF handoff fingerprint"
    "$PYTHON" "$repo/experiments/uniss_phase3_true_subsecond_deadline_full198_v1/scripts/build_phase3_fingerprint.py" \
      --source "$PHASE3_SAFETENSORS" --output "$PHASE3_FINGERPRINT"
  fi
  log "running 1-GPU native checkpoint/objective smoke"
  bash "$repo/experiments/uniss_phase3_true_subsecond_deadline_full198_v1/scripts/run_smoke_1gpu.sh"
  log "running 8-GPU 50-step throughput and resume smoke"
  bash "$repo/experiments/uniss_phase3_true_subsecond_deadline_full198_v1/scripts/run_smoke_8gpu.sh"
  log "starting TensorBoard and formal full198 joint epoch"
  bash "$repo/experiments/uniss_phase3_true_subsecond_deadline_full198_v1/scripts/start_tensorboard.sh"
  bash "$repo/experiments/uniss_phase3_true_subsecond_deadline_full198_v1/scripts/run_megatron_8gpu.sh"
' pipeline "${REPO_ROOT}")
printf -v quoted '%q ' "${command[@]}"
mkdir -p "$(dirname "${PIPELINE_LOG}")"
tmux new-session -d -s "${SESSION}" "${quoted} >> $(printf '%q' "${PIPELINE_LOG}") 2>&1"
echo "started ${SESSION}; log=${PIPELINE_LOG}"
