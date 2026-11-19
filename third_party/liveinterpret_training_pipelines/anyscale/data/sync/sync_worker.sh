#!/usr/bin/env bash
# =============================================================================
# 通用同步 worker — 支持 S3 / FSx / NVMe 之间的目录同步。
# =============================================================================
# 环境变量:
#   SYNC_SOURCE  源路径 (必需, s3:// 或绝对路径)
#   SYNC_TARGET  目标路径 (必需, s3:// 或绝对路径)
#   SYNC_REGION   S3 region (默认: ap-southeast-3, 仅 S3 操作时使用)
#   SYNC_MARKER   幂等标记文件名 (默认: .sync_done, 写入 TARGET 目录)
#   SYNC_OPTS     额外传递给 aws s3 sync / rsync 的参数
#
# 传输协议自动选择:
#   - 源或目标含 s3:// → aws s3 sync
#   - 都为本地路径       → rsync -a
# =============================================================================
set -euo pipefail

# --- 参数校验 ---
SRC="${SYNC_SOURCE:?SYNC_SOURCE is required}"
DST="${SYNC_TARGET:?SYNC_TARGET is required}"
REGION="${SYNC_REGION:-ap-southeast-3}"
MARKER_NAME="${SYNC_MARKER:-.sync_done}"
EXTRA_OPTS="${SYNC_OPTS:-}"

HOSTNAME="$(hostname)"
MARKER="${DST}/${MARKER_NAME}"

# --- 幂等检查 ---
if [[ -f "${MARKER}" ]]; then
    echo "[sync] ${HOSTNAME} ${MARKER} exists — skipping."
    exit 0
fi

is_s3() { [[ "$1" == s3://* ]]; }

echo "[sync] ${HOSTNAME} start at $(date -Iseconds)"
echo "[sync] source: ${SRC}"
echo "[sync] target: ${DST}"
echo "[sync] region: ${REGION}"
df -h /local 2>/dev/null | tail -1 || true

# --- S3 同步 ---
if is_s3 "${SRC}" || is_s3 "${DST}"; then
    echo "[sync] transport: aws s3 sync"

    aws configure set default.s3.max_concurrent_requests 100
    aws configure set default.s3.max_queue_size 10000

    aws sts get-caller-identity --output text || {
        echo "[sync][ERROR] AWS credentials missing." >&2
        exit 1
    }

    # shellcheck disable=SC2086
    aws s3 sync "${SRC}" "${DST}" \
        --region "${REGION}" \
        --only-show-errors \
        ${EXTRA_OPTS}

# --- 本地同步 (FSx / NVMe 之间) ---
else
    echo "[sync] transport: rsync"

    if [[ ! -e "${SRC}" ]]; then
        echo "[sync][ERROR] source not found: ${SRC}" >&2
        exit 1
    fi

    mkdir -p "${DST}"

    # shellcheck disable=SC2086
    rsync -a ${EXTRA_OPTS} "${SRC}/" "${DST}/"
fi

# --- 完成 ---
touch "${MARKER}"
echo "[sync] size: $(du -sh "${DST}" 2>/dev/null || true)"
df -h /local 2>/dev/null | tail -1 || true
echo "[sync] ${HOSTNAME} DONE at $(date -Iseconds)"
