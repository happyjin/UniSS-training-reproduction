#!/usr/bin/env bash
# Wait until Phase1 recovery finishes cleanly and Phase2 data is packed,
# then start Phase2. A failed check exits without launching training.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${REPO_ROOT}/configs/experiments/uniss_qwen3b_unist198_phase2_qp_replay_v1.env"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

POLL_SECONDS="${POLL_SECONDS:-60}"
LOG_PATH="${REPO_ROOT}/logs/uniss_qwen3b_phase2_waiter.log"
PHASE1_LOG="${REPO_ROOT}/logs/uniss_qwen3b_phase1_unist198_recovery_from7000_v1.log"
MIX_DIR="${REPO_ROOT}/data/processed/phase2_qwen3b_qp_replay_sharded"
ACTIVATE="${ACTIVATE_SCRIPT}"
SESSION="${PHASE2_SESSION:-qwen3b-p2}"

mkdir -p "$(dirname "${LOG_PATH}")" "${RUN_DIR}"

log() {
  echo "[$(date -u +%FT%TZ)] $*" | tee -a "${LOG_PATH}"
}

phase1_running() {
  pgrep -af 'pretrain_uniss_megatron.py' | grep -q 'recovery_from7000_v1' && return 0
  return 1
}

data_ready() {
  [[ -s "${TRAIN_DATA}" && -s "${TRAIN_DATA}.count" ]] || return 1
  local count shards
  count="$(tr -d '[:space:]' < "${TRAIN_DATA}.count")"
  [[ "${count}" =~ ^[1-9][0-9]*$ ]] || return 1
  shards="$(find "${MIX_DIR}" -maxdepth 1 -name 'train-*.jsonl' -size +0c | wc -l)"
  [[ "${shards}" == "198" ]] || return 1
  pgrep -f 'prepare_qwen3b_phase2_qp_replay.sh' >/dev/null && return 1
  return 0
}

checkpoint_ready() {
  local tracker iter_dir shards
  tracker="${SOURCE_CHECKPOINT}/latest_checkpointed_iteration.txt"
  [[ -f "${tracker}" ]] || return 1
  [[ "$(tr -d '[:space:]' < "${tracker}")" == "${EXPECTED_SOURCE_ITERATION}" ]] || return 1
  iter_dir="${SOURCE_CHECKPOINT}/iter_$(printf '%07d' "${EXPECTED_SOURCE_ITERATION}")"
  [[ -f "${iter_dir}/metadata.json" ]] || return 1
  shards="$(find "${iter_dir}" -maxdepth 1 -name '__*_0.distcp' -type f | wc -l)"
  [[ "${shards}" == "8" ]] || return 1
  return 0
}

training_healthy() {
  PHASE1_LOG="${PHASE1_LOG}" EXPECTED="${EXPECTED_SOURCE_ITERATION}" \
    "${USER_ROOT}/conda_envs/uniss-train/bin/python" - <<'PY'
import os, re, sys
from pathlib import Path
log = Path(os.environ["PHASE1_LOG"]).read_text(errors="replace")
expected = int(os.environ["EXPECTED"])
pat = re.compile(
    r"iteration\s+(\d+)/\s+\d+ .*?lm loss: ([0-9.E+-]+) .*?grad norm: ([0-9.E+-]+) .*?number of nan iterations:\s+(\d+)"
)
rows = [(int(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4))) for m in pat.finditer(log)]
if not rows or rows[-1][0] < expected - 10:
    sys.exit(f"last logged iter {rows[-1][0] if rows else 'none'} is short of {expected}")
tail = [row for row in rows if row[0] >= expected - 50]
if len(tail) < 3:
    sys.exit("not enough tail iterations to judge health")
for it, loss, grad, nans in tail:
    if nans or loss >= 10 or grad >= 100:
        sys.exit(f"unhealthy iter={it} loss={loss} grad={grad} nans={nans}")
if f"successfully saved checkpoint from iteration {expected:7d}" not in log and \
   f"successfully saved checkpoint from iteration {expected}" not in log:
    # Megatron pads the iteration field.
    if not re.search(rf"successfully saved checkpoint from iteration\s+{expected}\b", log):
        sys.exit(f"iter {expected} was not saved")
print(f"healthy tail_iters={len(tail)} last={rows[-1][0]} loss={rows[-1][1]:.4f} grad={rows[-1][2]:.4f}")
PY
}

gpus_free() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{
    gsub(/ /, "", $1); if ($1+0 > 2000) bad=1
  } END { exit bad ? 1 : 0 }'
}

start_phase2() {
  # Leading '=' is an exact name match. A prefix match treats qwen3b-p2-wait as qwen3b-p2.
  if tmux has-session -t "=${SESSION}" 2>/dev/null; then
    log "refusing to start: tmux session ${SESSION} already exists"
    exit 1
  fi
  tmux new-session -d -s "${SESSION}" -n train
  tmux new-window -t "${SESSION}" -n tensorboard
  tmux new-window -t "${SESSION}" -n gpu
  tmux new-window -t "${SESSION}" -n s3
  tmux send-keys -t "${SESSION}:train" "cd ${REPO_ROOT} && bash scripts/run_qwen3b_unist198_phase2.sh" C-m
  tmux send-keys -t "${SESSION}:tensorboard" "source ${ACTIVATE} && tensorboard --logdir ${TENSORBOARD_DIR} --host 0.0.0.0 --port ${TENSORBOARD_PORT} --load_fast=false" C-m
  tmux send-keys -t "${SESSION}:gpu" "echo timestamp,gpu_index,power_w,util_gpu,util_mem,mem_used_mb > ${GPU_LOG}; while true; do ts=\$(date -u +%FT%TZ); nvidia-smi --query-gpu=index,power.draw,utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits | awk -v ts=\"\$ts\" -F',' '{gsub(/ /,\"\",\$0); print ts\",\"\$0}' >> ${GPU_LOG}; sleep 10; done" C-m
  tmux send-keys -t "${SESSION}:s3" "cd ${REPO_ROOT} && bash scripts/backup_qwen3b_phase1_ckpts_to_s3.sh ${CONFIG_FILE}" C-m
  log "launched Phase2 in tmux ${SESSION}"
}

log "waiter start expected_iter=${EXPECTED_SOURCE_ITERATION}"
while true; do
  if phase1_running; then
    log "phase1 still running"
    sleep "${POLL_SECONDS}"
    continue
  fi
  if ! checkpoint_ready; then
    tracker="$(tr -d '[:space:]' < "${SOURCE_CHECKPOINT}/latest_checkpointed_iteration.txt" 2>/dev/null || echo missing)"
    log "phase1 stopped but checkpoint is ${tracker}, expected ${EXPECTED_SOURCE_ITERATION}. Not starting Phase2."
    exit 1
  fi
  if ! health="$(training_healthy 2>&1)"; then
    log "phase1 checkpoint ${EXPECTED_SOURCE_ITERATION} exists but the tail looks unhealthy. Not starting Phase2. ${health}"
    exit 1
  fi
  log "${health}"
  if ! data_ready; then
    log "phase1 is healthy; waiting for phase2 packed data"
    sleep "${POLL_SECONDS}"
    continue
  fi
  if ! gpus_free; then
    log "checkpoint and data are ready; waiting for GPUs to drain"
    sleep "${POLL_SECONDS}"
    continue
  fi
  log "checks passed; starting phase2"
  start_phase2
  exit 0
done
