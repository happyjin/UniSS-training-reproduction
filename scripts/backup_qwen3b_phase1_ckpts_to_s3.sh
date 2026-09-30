#!/usr/bin/env bash
# Backup every BACKUP_EVERY-step checkpoint to S3, then keep only the newest
# LOCAL_KEEP checkpoints on disk.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${1:-${REPO_ROOT}/configs/experiments/uniss_qwen3b_unist198_phase1_full_v1.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

SAVE_DIR="${SAVE_DIR:?}"
S3_CHECKPOINT_PREFIX="${S3_CHECKPOINT_PREFIX:?}"
BACKUP_EVERY="${BACKUP_EVERY:-500}"
LOCAL_KEEP="${LOCAL_KEEP:-3}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MARKER_DIR="${MARKER_DIR:-${RUN_DIR}/s3_uploaded}"
BACKUP_LOG="${BACKUP_LOG:-${REPO_ROOT}/logs/${EXPERIMENT_NAME}_s3_backup.log}"

mkdir -p "${MARKER_DIR}" "$(dirname "${BACKUP_LOG}")"

log() {
  echo "[$(date -u +%FT%TZ)] $*" | tee -a "${BACKUP_LOG}"
}

is_complete_ckpt() {
  local iter="$1"
  local dir="$2"
  local tracker="${SAVE_DIR}/latest_checkpointed_iteration.txt"
  [[ -d "${dir}" ]] || return 1
  [[ -f "${dir}/metadata.json" ]] || return 1
  [[ -f "${tracker}" ]] || return 1
  local latest
  latest="$(tr -d '[:space:]' < "${tracker}")"
  # Only treat a dir as complete after Megatron has published it in the tracker.
  [[ "${latest}" =~ ^[0-9]+$ ]] || return 1
  (( latest >= iter ))
}

upload_iter() {
  local iter="$1"
  local name
  printf -v name 'iter_%07d' "${iter}"
  local src="${SAVE_DIR}/${name}"
  local dst="${S3_CHECKPOINT_PREFIX%/}/${name}"
  local marker="${MARKER_DIR}/${name}"
  if [[ -f "${marker}" ]]; then
    return 0
  fi
  if ! is_complete_ckpt "${iter}" "${src}"; then
    return 0
  fi
  log "uploading ${src} -> ${dst}"
  aws s3 sync "${src}/" "${dst}/"
  aws s3 cp "${SAVE_DIR}/latest_checkpointed_iteration.txt" \
    "${S3_CHECKPOINT_PREFIX%/}/latest_checkpointed_iteration.txt" >/dev/null
  date -u +%FT%TZ > "${marker}"
  log "uploaded ${name}"
}

prune_local() {
  local tracker="${SAVE_DIR}/latest_checkpointed_iteration.txt"
  [[ -f "${tracker}" ]] || return 0
  mapfile -t dirs < <(find "${SAVE_DIR}" -maxdepth 1 -type d -name 'iter_*' | sort)
  local count="${#dirs[@]}"
  (( count > LOCAL_KEEP )) || return 0

  local keep_start=$((count - LOCAL_KEEP))
  local i dir name iter marker
  for ((i = 0; i < keep_start; i++)); do
    dir="${dirs[$i]}"
    name="$(basename "${dir}")"
    iter="${name#iter_}"
    iter=$((10#${iter}))
    if (( iter % BACKUP_EVERY == 0 )); then
      marker="${MARKER_DIR}/${name}"
      if [[ ! -f "${marker}" ]]; then
        log "skip prune ${name}: S3 backup not finished"
        continue
      fi
    fi
    log "pruning local ${dir}"
    rm -rf "${dir}"
  done
}

log "s3 backup watcher start save_dir=${SAVE_DIR} prefix=${S3_CHECKPOINT_PREFIX} every=${BACKUP_EVERY} keep=${LOCAL_KEEP}"

while true; do
  if [[ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
    latest="$(tr -d '[:space:]' < "${SAVE_DIR}/latest_checkpointed_iteration.txt" || true)"
    if [[ "${latest}" =~ ^[0-9]+$ ]]; then
      for ((iter = BACKUP_EVERY; iter <= latest; iter += BACKUP_EVERY)); do
        upload_iter "${iter}"
      done
    fi
    prune_local
  fi
  sleep "${POLL_SECONDS}"
done
