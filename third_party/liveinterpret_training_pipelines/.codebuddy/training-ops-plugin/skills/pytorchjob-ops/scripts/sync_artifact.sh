#!/usr/bin/env bash
# =============================================================================
# 把「一个同步任务」的全部工件对同步到本节点 NVMe（hostPath）。
# 由 derive_sync_jobs.py 派生出的 sync Job 在每个白名单节点上各跑一个 Pod
# （completions=N/parallelism=N + podAntiAffinity(hostname)，一节点一个）。
#
# 入参（全部经 env 传入，脚本本身不接收位置参数）：
#   KINDS      与 SRCS/DSTS 同序、同长度、空格分隔的源类型：
#                s3   源是 S3 直链 s3://<bucket>/<key>，走 `aws s3 sync --size-only`
#                mnt  源是容器内的 CSI 挂载点路径（对应 yaml 里的 s3mnt:// 工件），
#                     走本地文件系统增量同步（rsync/cp/tar，见 sync_one_mnt）
#              为兼容旧调用，KINDS 缺省时全部按 s3 处理。
#              （fsx:// 工件不会出现在这里 —— derive 侧根本不给它们生成 Job。）
#   SRCS       空格分隔的源列表，语义由同位置的 KINDS 决定。
#   DSTS       与 SRCS 一一对应的目标（NVMe hostPath + key，如 /nvme/...）。
#   SYNC_PAR   并发度：由 derive_sync_jobs.py 按 Job 的 CPU limit 显式传入。
#              0/空时回退到 cgroup 配额探测（见下），绝不使用 nproc。
#   NVME_MOUNT NVMe 容器挂载点（默认 /nvme），仅用于日志。
#   NODE_NAME  真实节点名（fieldRef spec.nodeName），仅用于日志。
#
# 为什么一个任务可能有多对（SRCS/DSTS 是列表）：
#   audio 任务的 ARTIFACT_AUDIO_SHARDS 是空格分隔的多个前缀，同属一个任务，
#   在同一个 Pod 内依次同步（规格 1.1）。其余任务通常只有一对。
#   任务内多对**串行**处理：任一对失败即整体失败（exit 1），Job 不 Complete，
#   submit.sh 侧的 completions=N 门控会挡住后续任务与训练。
#
# =============================================================================
# 就绪判定：完全交给 `aws s3 sync` 自身的比对机制
# =============================================================================
# `aws s3 sync` 原生按「相对 key 名 + 文件大小」做增量比对。官方 help 原文：
#   "The default behavior is to ignore same-sized items unless the local
#    version is newer than the S3 version."
# 因此：
#   * 本地缺失、或本地大小与源对象不一致 -> 自动重新下载并覆盖本地文件；
#   * 本地已一致 -> 跳过。天然断点续传、幂等（已一致时重跑输出 0 行）。
#
# 必须 --size-only：NVMe 本地文件 mtime 是下载时刻，与 S3 对象的
# last-modified 不同，若按默认 size+mtime 比较每次都会判"不一致"而全量重传。
#
# 同步动作就是「一次 `aws s3 sync --size-only`」，其返回码直接代表成败，
# 失败即 FATAL。不写任何额外的就绪标记文件，不遍历本地磁盘计数，
# 不做 dry-run 预检/后检 —— 这些机制相对 aws s3 sync 自带的比对都是多余的。
# =============================================================================
# 为什么不用 nproc 决定并发：
#   容器 CPU 限额由 cgroup cfs quota 实现，不会改 /proc/cpuinfo，
#   nproc 读到宿主机物理核数（P5E = 192），远超 Job 配额，按它并发会
#   CPU throttling 与进程堆积。并发来源依次为：
#     1) SYNC_PAR（由 Job 的 resources.limits.cpu 显式注入）—— 首选
#     2) cgroup v2 /sys/fs/cgroup/cpu.max
#     3) cgroup v1 cpu.cfs_quota_us / cpu.cfs_period_us
#     4) 保守默认 8
# 该 PAR 用作 aws s3 sync 的网络 IO 并发。注意：AWS CLI v1/v2 都没有
# `aws s3 sync --max-concurrent-requests` 子命令参数（放成参数会报
# Unknown options），也没有 AWS_MAX_CONCURRENT_REQUESTS 环境变量（已查源码
# 与 aws configure 实测确认：源码无 MAX_CONCURRENT 字符串）。并发只能经
# ~/.aws/config 的 [default].s3 段设置（max_concurrent_requests），本脚本用
# `aws configure set default.s3.max_concurrent_requests <N>` 写入，sync 会读取它。
# =============================================================================
# 不使用 `set -e`：本脚本对每一对工件显式检查返回码并累积到 rc，
# 需要在单对失败后继续（记录后再统一 exit 1），`set -e` 会在第一处非零就退出，
# 破坏"任务内多对都尝试、最终汇总失败"的语义。仍保留 -u/pipefail。
set -uo pipefail

SRCS="${SRCS:?SRCS required (space-separated s3:// URLs or CSI mount paths)}"
DSTS="${DSTS:?DSTS required (space-separated /nvme/<key> targets)}"
# KINDS 可缺省（旧调用方不传）：此时全部按 s3 处理，行为与改造前完全一致。
KINDS="${KINDS:-}"
NVME_MOUNT="${NVME_MOUNT:-/nvme}"

# 并发硬上限：AWS_MAX_CONCURRENT_REQUESTS 是网络 IO 并发，
# 32 已足够榨干 S3 吞吐，再高只增连接开销。
PAR_HARD_CAP=32

# 从 cgroup 读取本容器实际可用的 CPU 配额（核数，向下取整，至少 1）。
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

# ---- 配置 aws s3 并发：经 ~/.aws/config（无命令行参数/环境变量可用）----
# 已核实（aws configure set/get 实测）：AWS CLI 不读 AWS_MAX_CONCURRENT_REQUESTS
# 环境变量，只能经配置文件的 [default].s3.max_concurrent_requests 设置。
configure_concurrency() {
  local par="$1"
  # 优先写 default profile 的 s3 段；任一形式失败均不影响正确性（回退默认 10）。
  aws configure set default.s3.max_concurrent_requests "${par}" 2>/dev/null \
    || aws configure set s3.max_concurrent_requests "${par}" 2>/dev/null \
    || true
}

# ---- SRCS/DSTS 拆成数组并断言等长 ----
# derive 侧已保证 key 不含空白（否则 FATAL），这里可安全按空白 split。
read -r -a SRC_ARR <<< "${SRCS}"
read -r -a DST_ARR <<< "${DSTS}"
read -r -a KIND_ARR <<< "${KINDS}"
if [[ ${#SRC_ARR[@]} -ne ${#DST_ARR[@]} ]]; then
  echo "[sync-artifact][FATAL] SRCS/DSTS length mismatch: ${#SRC_ARR[@]} vs ${#DST_ARR[@]}" >&2
  exit 1
fi
if [[ ${#SRC_ARR[@]} -eq 0 ]]; then
  echo "[sync-artifact][FATAL] empty SRCS/DSTS" >&2
  exit 1
fi
# KINDS 缺省 -> 全部补 s3（向后兼容）；给了就必须等长，长度不齐说明派生侧
# 出了 bug，会把 kind 与 src 错位配对（例如拿 s3 的方式去处理本地路径），
# 必须当场炸掉而不是猜。
if [[ ${#KIND_ARR[@]} -eq 0 ]]; then
  for _i in "${!SRC_ARR[@]}"; do KIND_ARR[_i]="s3"; done
elif [[ ${#KIND_ARR[@]} -ne ${#SRC_ARR[@]} ]]; then
  echo "[sync-artifact][FATAL] KINDS/SRCS length mismatch: ${#KIND_ARR[@]} vs ${#SRC_ARR[@]}" >&2
  exit 1
fi

echo "[sync-artifact] $(date -u +%FT%TZ) node=${NODE_NAME:-unknown} nvme=${NVME_MOUNT}"
echo "[sync-artifact] pairs=${#SRC_ARR[@]} PAR=${PAR} (source=${PAR_SRC}, s3.max_concurrent_requests=${PAR} via ~/.aws/config; cgroup_quota=$(cpu_quota_cores || echo none), host_nproc=$(nproc 2>/dev/null || echo '?') [not used])"
configure_concurrency "${PAR}"

# ---- kind=s3：一次 `aws s3 sync --size-only`（增量、幂等、无 dry-run）----
sync_one_s3() {
  local SRC="$1" DST="$2"

  if [[ "${SRC}" != s3://* ]]; then
    echo "[sync-artifact][FATAL] kind=s3 but SRC is not an s3:// URL: ${SRC}" >&2
    return 1
  fi

  echo "[sync-artifact] mode=s3-direct (incremental aws s3 sync --size-only)"
  if ! command -v aws >/dev/null 2>&1; then
    echo "[sync-artifact][FATAL] aws CLI not found in image; s3-direct mode requires it" >&2
    return 1
  fi
  # --size-only：仅按文件大小比较，忽略 mtime（NVMe 本地 mtime 是下载时刻，
  #   与 S3 对象 last-modified 不同，按默认 size+mtime 会比较成"不一致"导致全量重传）。
  # --no-progress：避免刷屏；--only-show-errors：只报错误，正常传输静默。
  # 并发经 ~/.aws/config 的 [default].s3.max_concurrent_requests 设置
  # （见文件头说明；AWS CLI 无对应命令行参数/环境变量），已在本脚本启动时配置。
  # 不吞 stderr：凭证缺失 / 403 / 桶不存在都必须暴露。
  # 返回码即成败：非零 = 有对象传输失败，必须 FATAL（不静默）。
  if ! aws s3 sync "${SRC%/}/" "${DST}/" \
          --size-only --no-progress --only-show-errors; then
    echo "[sync-artifact][FATAL] aws s3 sync failed for ${SRC} (see errors above)" >&2
    return 1
  fi
  return 0
}

# ---- kind=mnt：源已由 CSI 挂进容器，做本地文件系统增量同步 ----
# 三档实现，先探测能力再降级，并在日志里写明用了哪一档（不能让人事后猜）：
#   1) rsync -a --size-only   最优：原生增量，语义与 aws s3 sync --size-only 一致
#                             （只比大小，不比 mtime —— 本地副本的 mtime 是拷贝
#                             时刻，比 mtime 会每次全量重传）
#   2) cp -au                 次选：coreutils 自带，-u 按 mtime+size 判新旧。
#                             注意它与 --size-only 语义不同（见下方 WARN），
#                             但对「只读挂载的静态工件」足够：源 mtime 稳定，
#                             首次拷完后 dst 不会更旧，不会反复重传。
#   3) tar 管道               兜底：无 rsync 也无 cp -u 支持时全量重拷。
#                             不增量，但保证正确性（宁可慢，不能少文件）。
sync_one_mnt() {
  local SRC="$1" DST="$2"

  if [[ ! -d "${SRC}" ]]; then
    # 源目录不存在 = CSI 卷没挂上 / 路径写错。绝不能当成"没东西要同步"跳过，
    # 那样训练会读到空目录才失败，排查成本高得多。
    echo "[sync-artifact][FATAL] kind=mnt source directory not found: ${SRC}" >&2
    echo "[sync-artifact][FATAL]   check that the s3all/CSI volume is mounted in the sync pod" >&2
    return 1
  fi

  if command -v rsync >/dev/null 2>&1; then
    echo "[sync-artifact] mode=mnt-local impl=rsync (rsync -a --size-only)"
    if ! rsync -a --size-only "${SRC%/}/" "${DST}/"; then
      echo "[sync-artifact][FATAL] rsync failed for ${SRC}" >&2
      return 1
    fi
    return 0
  fi

  # rsync 不在镜像里 -> 降级。先验证 cp 是否支持 -u（busybox cp 不支持）。
  if cp --help 2>&1 | grep -q -- '-u'; then
    echo "[sync-artifact] mode=mnt-local impl=cp (cp -a -u; rsync not found in image)"
    echo "[sync-artifact] NOTE: cp -u compares mtime+size (not size-only); artifacts are"
    echo "[sync-artifact]       read-only static files so their mtime is stable — safe here."
    # -a 保留属性并递归；-u 只在源更新或目标缺失时拷贝；SRC/. 保证拷的是目录内容
    if ! cp -a -u "${SRC%/}/." "${DST}/"; then
      echo "[sync-artifact][FATAL] cp -au failed for ${SRC}" >&2
      return 1
    fi
    return 0
  fi

  echo "[sync-artifact] mode=mnt-local impl=tar (full re-copy; neither rsync nor cp -u available)"
  echo "[sync-artifact] WARN: this impl is NOT incremental — every run re-copies everything."
  # pipefail 已开启：tar 读端或写端任一失败都会让整条管道返回非零。
  if ! tar -C "${SRC%/}" -cf - . | tar -C "${DST}" -xf -; then
    echo "[sync-artifact][FATAL] tar pipe copy failed for ${SRC}" >&2
    return 1
  fi
  return 0
}

# ---- 单对同步：按 kind 分派 ----
# 返回 0 表示该对同步成功；非 0 表示失败。
sync_one() {
  local KIND="$1" SRC="$2" DST="$3"

  echo "[sync-artifact] ---- KIND=${KIND} SRC=${SRC}"
  echo "[sync-artifact]      DST=${DST}"

  mkdir -p "${DST}"

  case "${KIND}" in
    s3)  sync_one_s3  "${SRC}" "${DST}" || return 1 ;;
    mnt) sync_one_mnt "${SRC}" "${DST}" || return 1 ;;
    *)
      # 未知 kind 绝不静默按某种方式猜着处理：错猜会把数据同步到错位置且返回 0。
      echo "[sync-artifact][FATAL] unknown KIND='${KIND}' for SRC=${SRC}; expected 's3' or 'mnt'" >&2
      return 1
      ;;
  esac

  echo "[sync-artifact] OK ${SRC}"
  return 0
}

# ---- 任务内多对：依次同步，任一失败即整体失败 ----
# 不使用 `|| true`：单对失败必须冒泡成 exit 1，让 Job 不 Complete，
# 从而被 submit.sh 的 completions=N 门控挡住（B2：失败绝不静默当成功）。
FAIL=0
for i in "${!SRC_ARR[@]}"; do
  if ! sync_one "${KIND_ARR[i]}" "${SRC_ARR[i]}" "${DST_ARR[i]}"; then
    FAIL=1
  fi
done

if [[ "${FAIL}" -ne 0 ]]; then
  echo "[sync-artifact][FATAL] one or more pairs failed on node ${NODE_NAME:-unknown}" >&2
  exit 1
fi
echo "[sync-artifact] ALL DONE on node ${NODE_NAME:-unknown} (${#SRC_ARR[@]} pair(s))"
