#!/usr/bin/env bash
# =============================================================================
# training-ops / ktrain-job-ops / sync_artifact.sh
# =============================================================================
# 把「一个同步任务」的全部工件对同步到本节点 NVMe（hostPath）。
# 由 derive_sync_jobs.py 派生出的 sync Job 在每个白名单节点上各跑一个 Pod
# （completions=N/parallelism=N + podAntiAffinity(hostname)，一节点一个）。
#
# 入参（全部经 env 传入，脚本不接收位置参数）：
#   KINDS      与 SRCS/DSTS 同序、同长度、空格分隔的源类型：
#                s3     源是 S3 直链 s3://<bucket>/<key>，走 aws s3 sync --size-only
#                s3mnt  源是 S3 CSI 挂载点下的路径（对应 yaml 里的 s3mnt:// 工件），
#                       走本地文件系统增量同步（rsync/cp/tar，见 sync_one_local）
#                mnt    源是其他本地挂载 / 文件系统路径，同 s3mnt 的本地同步实现
#              （fsx:// 工件不产生同步 Job，不会出现在这里）
#   SRCS       空格分隔的源列表，语义由同位置的 KINDS 决定
#   DSTS       与 SRCS 一一对应的目标（<NVME_MOUNT>/<key>）
#   SYNC_PAR   并发度（由 derive_sync_jobs.py 按 Job CPU limit 显式注入）
#   NVME_MOUNT NVMe 容器挂载点，仅用于日志
#   NODE_NAME  节点名（fieldRef spec.nodeName），仅用于日志
#
# 每个工件对执行一次同步，返回码即成败，任一失败即退出 1。
# =============================================================================
set -uo pipefail

SRCS="${SRCS:?SRCS required (space-separated sources)}"
DSTS="${DSTS:?DSTS required (space-separated <NVME_MOUNT>/<key> targets)}"
KINDS="${KINDS:?KINDS required (space-separated: s3 / s3mnt / mnt)}"
NVME_MOUNT="${NVME_MOUNT:-/nvme}"

PAR_HARD_CAP=32

cpu_quota_cores() {
  local quota period
  if [[ -r /sys/fs/cgroup/cpu.max ]]; then
    read -r quota period < /sys/fs/cgroup/cpu.max 2>/dev/null || return 1
    [[ "${quota}" == "max" ]] && return 1
  elif [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then
    quota="$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null)" || return 1
    period="$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null)" || return 1
    [[ "${quota}" -le 0 ]] && return 1
  else
    return 1
  fi
  [[ -z "${period:-}" || "${period}" -le 0 ]] && return 1
  echo $(( quota / period > 0 ? quota / period : 1 ))
}

PAR_SRC="SYNC_PAR"
PAR="${SYNC_PAR:-0}"
if [[ -z "${PAR}" || "${PAR}" == "0" ]]; then
  if PAR="$(cpu_quota_cores)"; then
    PAR_SRC="cgroup-quota"
  else
    PAR=8
    PAR_SRC="default"
  fi
fi
if [[ "${PAR}" -gt "${PAR_HARD_CAP}" ]]; then
  echo "[sync-artifact] PAR=${PAR} exceeds hard cap; clamping to ${PAR_HARD_CAP}"
  PAR="${PAR_HARD_CAP}"
  PAR_SRC="${PAR_SRC}+capped"
fi

aws configure set default.s3.max_concurrent_requests "${PAR}" 2>/dev/null \
  || aws configure set s3.max_concurrent_requests "${PAR}" 2>/dev/null || true

read -r -a SRC_ARR <<< "${SRCS}"
read -r -a DST_ARR <<< "${DSTS}"
read -r -a KIND_ARR <<< "${KINDS}"
if [[ ${#SRC_ARR[@]} -ne ${#DST_ARR[@]} ]]; then
  echo "[sync-artifact][FATAL] SRCS/DSTS length mismatch: ${#SRC_ARR[@]} vs ${#DST_ARR[@]}" >&2
  exit 1
fi
if [[ ${#KIND_ARR[@]} -ne ${#SRC_ARR[@]} ]]; then
  echo "[sync-artifact][FATAL] KINDS/SRCS length mismatch: ${#KIND_ARR[@]} vs ${#SRC_ARR[@]}" >&2
  exit 1
fi
if [[ ${#SRC_ARR[@]} -eq 0 ]]; then
  echo "[sync-artifact][FATAL] empty SRCS/DSTS" >&2
  exit 1
fi

echo "[sync-artifact] $(date -u +%FT%TZ) node=${NODE_NAME:-unknown} nvme=${NVME_MOUNT}"
echo "[sync-artifact] pairs=${#SRC_ARR[@]} PAR=${PAR} (source=${PAR_SRC})"

# 探测桶真实 region，跨区域桶不带正确 region 会被 400 拒绝。
detect_region() {
  local bucket="$1" loc
  if [[ "${bucket}" =~ -([a-z]{2}-[a-z]+-[0-9])- ]]; then
    echo "${BASH_REMATCH[1]}"; return 0
  fi
  loc="$(aws s3api get-bucket-location --bucket "${bucket}" \
           --query LocationConstraint --output text 2>/dev/null || true)"
  if [[ -n "${loc}" && "${loc}" != "null" ]]; then
    echo "${loc}"; return 0
  fi
  echo "${AWS_DEFAULT_REGION:-us-east-1}"; return 0
}

sync_one_s3() {
  local SRC="$1" DST="$2"
  if [[ "${SRC}" != s3://* ]]; then
    echo "[sync-artifact][FATAL] kind=s3 but SRC is not an s3:// URL: ${SRC}" >&2
    return 1
  fi
  if ! command -v aws >/dev/null 2>&1; then
    echo "[sync-artifact][FATAL] aws CLI not found in image; kind=s3 requires it" >&2
    return 1
  fi
  local bucket region
  bucket="${SRC#s3://}"; bucket="${bucket%%/*}"
  region="$(detect_region "${bucket}")"
  echo "[sync-artifact] mode=s3-direct bucket=${bucket} region=${region}"
  if ! aws s3 sync "${SRC%/}/" "${DST}/" \
          --region "${region}" --size-only --no-progress --only-show-errors; then
    echo "[sync-artifact][FATAL] aws s3 sync failed for ${SRC}" >&2
    return 1
  fi
  return 0
}

# s3mnt / mnt 共用的本地文件系统增量同步：rsync -a --size-only -> cp -a -u -> tar。
sync_one_local() {
  local SRC="$1" DST="$2"
  if [[ ! -d "${SRC}" ]]; then
    echo "[sync-artifact][FATAL] local source directory not found: ${SRC}" >&2
    echo "[sync-artifact][FATAL]   check the source volume is mounted in the sync pod" >&2
    return 1
  fi
  if command -v rsync >/dev/null 2>&1; then
    echo "[sync-artifact] mode=local impl=rsync (rsync -a --size-only)"
    if ! rsync -a --size-only "${SRC%/}/" "${DST}/"; then
      echo "[sync-artifact][FATAL] rsync failed for ${SRC}" >&2
      return 1
    fi
    return 0
  fi
  if cp --help 2>&1 | grep -q -- '-u'; then
    echo "[sync-artifact] mode=local impl=cp (cp -a -u; rsync not found in image)"
    if ! cp -a -u "${SRC%/}/." "${DST}/"; then
      echo "[sync-artifact][FATAL] cp -au failed for ${SRC}" >&2
      return 1
    fi
    return 0
  fi
  echo "[sync-artifact] mode=local impl=tar (full re-copy)"
  if ! tar -C "${SRC%/}" -cf - . | tar -C "${DST}" -xf -; then
    echo "[sync-artifact][FATAL] tar pipe copy failed for ${SRC}" >&2
    return 1
  fi
  return 0
}

FAIL=0
for i in "${!SRC_ARR[@]}"; do
  KIND="${KIND_ARR[i]}"; SRC="${SRC_ARR[i]}"; DST="${DST_ARR[i]}"
  echo "[sync-artifact] ---- KIND=${KIND} SRC=${SRC}"
  echo "[sync-artifact]      DST=${DST}"
  mkdir -p "${DST}"
  rc=0
  case "${KIND}" in
    s3)         sync_one_s3    "${SRC}" "${DST}" || rc=1 ;;
    s3mnt|mnt)  sync_one_local "${SRC}" "${DST}" || rc=1 ;;
    *)
      echo "[sync-artifact][FATAL] unknown KIND='${KIND}' for SRC=${SRC}; expected s3 / s3mnt / mnt" >&2
      rc=1
      ;;
  esac
  if [[ "${rc}" -eq 0 ]]; then
    echo "[sync-artifact] OK ${SRC}"
  else
    FAIL=1
  fi
done

if [[ "${FAIL}" -ne 0 ]]; then
  echo "[sync-artifact][FATAL] one or more pairs failed on node ${NODE_NAME:-unknown}" >&2
  exit 1
fi
echo "[sync-artifact] ALL DONE on node ${NODE_NAME:-unknown} (${#SRC_ARR[@]} pair(s))"
