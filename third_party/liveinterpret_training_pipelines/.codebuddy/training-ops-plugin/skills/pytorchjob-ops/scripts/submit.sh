#!/usr/bin/env bash
# =============================================================================
# 提交 bridge-alignment PyTorchJob（纯 kubectl，无 anyscale/ray/ktrain）。
#
# 与训练 yaml（pytorch_job/pytorchjob.yaml）的关系：
#   * 该 yaml 是任务配置的唯一真源（镜像/节点/模型/tokenizer/manifest/音频路径）。
#   * 本脚本读取它，由 derive_sync_jobs.py 推导出「数据就绪门控」所需的
#     4 个串行同步 Job（model/tokenizer/manifest/audio）。
#
# 数据就绪门控的新架构（本次重构，替代旧的逐节点 check/分批 sync）：
#   * 每个同步任务 = 一个 batch/v1 Job，completions=N/parallelism=N/Indexed +
#     podAntiAffinity(hostname)，N=白名单节点数 -> N 个 Pod 落 N 个不同节点，
#     "全部成功才 Complete"是硬条件（PyTorchJob 会在 Master 成功时假成功，
#     清掉仍在同步的 Worker，正是要根除的 bug）。
#   * 四任务严格串行：前一个 Job 的 N 个 Pod 全部 Complete，才提交下一个。
#   * 每个 sync Pod 跑一次 `aws s3 sync --size-only`（按 key 名 + 大小增量、
#     幂等），其返回码即代表成败。sync 自带比对已足够，不做任何额外复核。
#   * 超时不硬编码：source plan.env 里 derive 已算好的 WAIT_TIMEOUT_<task>
#     （已严格 > 各自 activeDeadlineSeconds + 缓冲）。
#
# 流程：
#   1. 刷 kubecontext；建 ns + 准入 label；确保 SA、模型桶 PVC（缺失则创建）。
#   2. 把 repo 代码同步到 data bucket（容器经 S3 CSI 直接读，入口即 S3 路径）。
#   3. 建同步脚本 ConfigMap；推导任务与调度计划（plan.env）。
#   4. 预检白名单节点健康 -> 各任务串行同步 -> 提交训练。
#
# 用法：
#   submit.sh [--yaml <训练 yaml>] [--job-name-prefix <前缀>]
# 环境变量等价物（命令行参数优先）：
#   PYTORCHJOB_YAML   训练 yaml 路径，默认 <repo>/pytorch_job/pytorchjob.yaml
#   JOB_NAME_PREFIX   PyTorchJob 名前缀，默认 qwen3-omnispeak-bridge-align-h200
# 最终 job 名 = <prefix>-<RUN_ID>。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}"; while [ ! -d pytorch_job ] && [ "${PWD}" != / ]; do cd ..; done; pwd)"

# ---- 参数解析：--yaml / --job-name-prefix（可用同名环境变量给默认值）----
YAML="${PYTORCHJOB_YAML:-${REPO_ROOT}/pytorch_job/pytorchjob.yaml}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-qwen3-omnispeak-bridge-align-h200}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --yaml)             YAML="$2"; shift 2 ;;
    --job-name-prefix)  JOB_NAME_PREFIX="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)
      # 未知参数不静默忽略：拼错 flag（如 --yml）会让脚本用默认 yaml 提交
      # 一个完全不是用户想要的任务，且没人会发现。
      echo "[submit][FATAL] unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ -f "${YAML}" ]] || { echo "[submit][FATAL] yaml not found: ${YAML}" >&2; exit 1; }

LAUNCH="${REPO_ROOT}/pytorch_job/launch_train.sh"

# ---- 项目基础设施常量（模型桶 PVC / SA，仅首次需创建）----
CLUSTER=speech-h200-eks
REGION=ap-southeast-3
NAMESPACE=speech-h200
SERVICE_ACCOUNT=speech-h200-sa
DATA_PVC=speech-h200-s3-pvc

RUN_ID="$(tr -dc 'a-z0-9' < /dev/urandom | head -c4 2>/dev/null || true)"
RUN_ID="${RUN_ID:-r$(date +%s | tail -c4)}"   # 兜底：urandom 不可用时退化为时间戳
TMPD="$(mktemp -d /tmp/pytorchjob.${RUN_ID}.XXXXXX)"
trap 'rm -rf "${TMPD}"' EXIT

# ---- 本地落地日志：每一步的 stdout/stderr 同时写盘，便于集群侧出问题时排查 ----
# 用户明确要求：submit 每个步骤都要落地本地日志供排查。LOG 在本地（提交机），
# 与集群 Job 日志解耦——即使 kubectl 输出被截断，本地这份完整留痕还在。
LOG="/tmp/submit-${RUN_ID}.log"
: > "${LOG}"
echo "[submit] local log -> ${LOG}"
exec > >(tee -a "${LOG}") 2>&1

echo "[submit] kubecontext -> ${CLUSTER}"
aws eks update-kubeconfig --region "${REGION}" --name "${CLUSTER}"

echo "[submit] ensure namespace ${NAMESPACE} + admission label"
kubectl create namespace "${NAMESPACE}" 2>/dev/null || true
kubectl label namespace "${NAMESPACE}" ktrain.io/managed=true --overwrite

echo "[submit] ensure SA"
kubectl -n "${NAMESPACE}" get sa "${SERVICE_ACCOUNT}" >/dev/null 2>&1 || \
  kubectl -n "${NAMESPACE}" create serviceaccount "${SERVICE_ACCOUNT}"

echo "[submit] ensure data PVC (${DATA_PVC})"
if ! kubectl -n "${NAMESPACE}" get pvc "${DATA_PVC}" >/dev/null 2>&1; then
  echo "[submit][FATAL] data PVC ${DATA_PVC} not found; it must be pre-provisioned in the cluster." >&2
  exit 1
fi

echo "[submit] sync repo code -> s3"
REFRESH_KUBECONFIG=0 bash "${SCRIPT_DIR}/sync_code.sh" "${YAML}"

# ---- 同步脚本 ConfigMap ----
# derive_sync_jobs.py 派生的 sync Job 挂卷 syncbin -> ConfigMap
# `artifact-sync-script`，执行 /opt/sync/sync_artifact.sh。
echo "[submit] build artifact-sync-script-${RUN_ID} ConfigMap"
# CM 名带 RUN_ID（与 derive 约定的 artifact-sync-script-<run-id> 对齐，避免并发
# submit 互相覆盖脚本；D4）。衍生 yaml 里 configMap.name 由 derive 用同一 run-id 生成。
kubectl create configmap "artifact-sync-script-${RUN_ID}" \
  --from-file=sync_artifact.sh="${SCRIPT_DIR}/sync_artifact.sh" \
  -n "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

echo "[submit] derive sync jobs from ${YAML}"
# derive 把产物写到 ${TMPD}：sync_<task>.yaml / plan.env。
# 它的 stdout 是人读的摘要（N/nodes/cpu/par + 任务列表），不参与解析。
python3 "${SCRIPT_DIR}/derive_sync_jobs.py" --yaml "${YAML}" --out "${TMPD}" --run-id "${RUN_ID}"

# plan.env 由 derive 写出。字段来源见 derive_sync_jobs.py 的 main()：
#   NAMESPACE / NODES / NODE_COUNT / SYNC_TASKS / SKIPPED_TASKS /
#   FSX_TASKS / FSX_ARTIFACTS / FSX_MOUNT / ARTIFACT_COUNT / SYNC_PAR /
#   SYNC_CPU_REQUEST / SYNC_CPU_LIMIT /
#   每个 active 任务的 WAIT_TIMEOUT_<task> 与 JOB_NAME_<task>。
# shellcheck source=/dev/null
source "${TMPD}/plan.env"
read -r -a NODES_ARR <<< "${NODES}"
read -r -a SYNC_TASKS_ARR <<< "${SYNC_TASKS:-}"
read -r -a SKIPPED_TASKS_ARR <<< "${SKIPPED_TASKS:-}"
read -r -a FSX_ARTIFACTS_ARR <<< "${FSX_ARTIFACTS:-}"

echo "[submit] namespace=${NAMESPACE} nodes(${NODE_COUNT}): ${NODES}"
echo "[submit] sync tasks (serial): ${SYNC_TASKS:-<none>}"
if [[ ${#SKIPPED_TASKS_ARR[@]} -gt 0 ]]; then
  # env 缺失的任务显式打印，不能静默丢掉（会让人误以为"同步过了"）。
  echo "[submit] skipped tasks (env not set): ${SKIPPED_TASKS}"
fi
# FSX 直读工件：不产生同步 Job，但必须逐条打印 —— "跳过同步"绝不能是静默的，
# 否则出问题时无法区分「本来就不用同步」和「同步任务被漏掉了」。
if [[ -n "${FSX_TASKS:-}" ]]; then
  echo "[submit] fsx-direct tasks (read straight from FSX, no sync job): ${FSX_TASKS}"
fi
if [[ ${#FSX_ARTIFACTS_ARR[@]} -gt 0 ]]; then
  echo "[submit] fsx mount in container: ${FSX_MOUNT:-<unset>}"
  for _a in "${FSX_ARTIFACTS_ARR[@]}"; do
    echo "[submit]   fsx-direct artifact: ${_a}"
  done
fi
# 「一个同步任务都没有」不再是错误：工件可以全部走 FSX 直读。
# 真正的错误是「一个工件都没解析出来」，那由 derive 侧的 ARTIFACT_COUNT 判定。
if [[ "${ARTIFACT_COUNT:-0}" -eq 0 ]]; then
  echo "[submit][FATAL] no artifacts derived from yaml env at all" >&2
  exit 1
fi
if [[ ${#SYNC_TASKS_ARR[@]} -eq 0 ]]; then
  echo "[submit] no sync jobs needed (all artifacts are fsx-direct); proceeding to training"
fi

# =============================================================================
# B5：提交任何同步 Job 前，预检白名单每个节点 Ready 且未 cordon。
# 不做这一步的话，异常节点上的 Pod 会一直 Pending 到 activeDeadlineSeconds
# 才失败（audio 是 6h），白等一场。这里一次性用 kubectl get node -o json
# 结构化判断，异常立即 FATAL 列出。
# =============================================================================
precheck_nodes() {
  echo "[submit] pre-flight: checking ${NODE_COUNT} whitelisted node(s) Ready & schedulable..."
  local bad=() node json ready unsched
  for node in "${NODES_ARR[@]}"; do
    if ! json="$(kubectl get node "${node}" -o json 2>/dev/null)"; then
      bad+=("${node}(not-found)")
      continue
    fi
    # status.conditions 里 type=Ready 的 status 必须为 "True"
    ready="$(echo "${json}" | python3 -c '
import json,sys
d=json.load(sys.stdin)
c=[x for x in d.get("status",{}).get("conditions",[]) if x.get("type")=="Ready"]
print(c[0].get("status") if c else "Unknown")
')"
    # spec.unschedulable=true 即被 cordon
    unsched="$(echo "${json}" | python3 -c '
import json,sys
d=json.load(sys.stdin)
print(str(d.get("spec",{}).get("unschedulable", False)).lower())
')"
    if [[ "${ready}" != "True" ]]; then
      bad+=("${node}(NotReady:${ready})")
    elif [[ "${unsched}" == "true" ]]; then
      bad+=("${node}(cordoned)")
    fi
  done
  if [[ ${#bad[@]} -gt 0 ]]; then
    echo "[submit][FATAL] ${#bad[@]} whitelisted node(s) unhealthy/cordoned: ${bad[*]}" >&2
    echo "[submit][FATAL] fix node health before syncing; aborting to avoid Pending-until-deadline." >&2
    exit 1
  fi
  echo "[submit] pre-flight OK: all ${NODE_COUNT} node(s) Ready & schedulable"
}
precheck_nodes

# =============================================================================
# B4：采集一个 Job 全部 Pod 的日志。
# 禁止 `kubectl logs job/<name>`：多 Pod 时它只取其中一个（取哪个不确定），
# 会漏掉失败节点的日志。这里先列出该 Job 的全部 Pod（按 label sync-job=<name>，
# 是我们在 template.metadata.labels 自打的自定义 label，不依赖 controller 注入的
# job-name label —— 后者在不同 k8s 版本命名不一致），再逐 Pod 取日志，
# 前缀标注节点名，便于精确定位到哪个节点失败。
# =============================================================================
dump_job_logs() {
  local job="$1" tail="${2:-40}" pods pod node
  pods="$(kubectl -n "${NAMESPACE}" get pods -l "sync-job=${job}" \
            -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.spec.nodeName}{"\n"}{end}' 2>/dev/null || true)"
  if [[ -z "${pods}" ]]; then
    echo "      (no pods found for job ${job})" >&2
    return 0
  fi
  while read -r pod node; do
    [[ -z "${pod}" ]] && continue
    echo "      ---- pod=${pod} node=${node:-<unscheduled>} ----" >&2
    kubectl -n "${NAMESPACE}" logs "${pod}" --tail="${tail}" 2>&1 \
      | sed "s/^/      [${node:-?}] /" >&2 || true
  done <<< "${pods}"
}

# =============================================================================
# 校验一个 Job 是否「N 个 Pod 全部 succeeded」。
# 不能只信 kubectl wait 的返回码：这里再显式核对 status.succeeded == N，
# 任何一处不满足都 FATAL（B2：失败绝不静默当成功；不足 N 视为失败）。
# =============================================================================
assert_job_complete() {
  local job="$1" succeeded
  succeeded="$(kubectl -n "${NAMESPACE}" get job "${job}" \
                 -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
  succeeded="${succeeded:-0}"
  if [[ "${succeeded}" != "${NODE_COUNT}" ]]; then
    echo "[submit][FATAL] job ${job}: succeeded=${succeeded} != N=${NODE_COUNT}; not all nodes synced" >&2
    dump_job_logs "${job}"
    return 1
  fi
  return 0
}

# =============================================================================
# 轮询等待 Job 进入终态（Complete / Failed）。
# 不能用 `kubectl wait --for=condition=complete --for=condition=failed`：
# kubectl wait 的多个 --for 条件是 **AND** 语义，而 complete 与 failed 不可能
# 同时为真，于是永远等不到、傻等满 timeout 才超时（已实测：dual 条件 15s 超时、
# 单条件 0.9s 返回）。这里每 ${POLL_STEP}s 查一次 status.conditions：
#   返回 0 = Complete=true（成功）；返回 2 = Failed=true（失败）；返回 1 = 超时。
# 成功与失败都能在 ~POLL_STEP 秒内被感知，不再傻等。
# =============================================================================
POLL_STEP=10
await_job() {
  local job="$1" timeout="$2" elapsed=0 cond
  while (( elapsed < timeout )); do
    cond="$(kubectl -n "${NAMESPACE}" get job "${job}" \
              -o jsonpath='{range .status.conditions[*]}{.type}={.status};{end}' \
              2>/dev/null || true)"
    # K8s condition 状态值是大写 True/False（已实测条件串形如
    # "Complete=True;"/"Failed=True;"），必须严格匹配大小写。
    [[ "${cond}" == *"Complete=True"* ]] && return 0
    [[ "${cond}" == *"Failed=True"* ]]   && return 2
    sleep "${POLL_STEP}"; elapsed=$(( elapsed + POLL_STEP ))
  done
  return 1
}

# =============================================================================
# 提交并等待单个 sync Job 完成。
#   $1 job 名，$2 yaml 路径，$3 wait 超时秒数（来自 plan.env，已 > deadline）。
# 严格串行的单元：apply -> 轮询终态 -> 显式核对 succeeded==N -> 采日志。
# =============================================================================
run_job() {
  local job="$1" yaml="$2" timeout="$3" rc=0 failed succeeded
  echo "[submit] >>> applying job ${job} (wait timeout=${timeout}s)"
  # 幂等：先删同名残留（仅本 run 的对象，run-id 随机；--ignore-not-found 安全）。
  # --wait=true：必须等删除真正完成再 apply。默认的异步删除会让紧接着的 apply
  # 撞上仍在 Terminating 的同名 Job（AlreadyExists / 竞态复活旧 Pod）。
  kubectl -n "${NAMESPACE}" delete job "${job}" --ignore-not-found \
    --wait=true --timeout=120s >/dev/null 2>&1 || true
  kubectl -n "${NAMESPACE}" apply -f "${yaml}"

  await_job "${job}" "${timeout}"; local aw=$?
  succeeded="$(kubectl -n "${NAMESPACE}" get job "${job}" \
                 -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
  failed="$(kubectl -n "${NAMESPACE}" get job "${job}" \
              -o jsonpath='{.status.failed}' 2>/dev/null || true)"
  succeeded="${succeeded:-0}"; failed="${failed:-0}"

  if [[ "${aw}" -eq 0 && "${succeeded}" == "${NODE_COUNT}" ]]; then
    :   # 真完成：全部 N 个节点 Pod 成功
  elif [[ "${aw}" -eq 2 || "${failed}" -gt 0 ]]; then
    echo "[submit][ERR] job ${job} FAILED: ${failed} pod(s) errored on some node(s)" >&2
    rc=1
  elif [[ "${aw}" -eq 1 ]]; then
    echo "[submit][ERR] job ${job} not complete within ${timeout}s; " \
         "succeeded=${succeeded} failed=${failed}" >&2
    kubectl -n "${NAMESPACE}" describe "job/${job}" 2>&1 | tail -20 | sed 's/^/      /' >&2 || true
    dump_job_logs "${job}"
    rc=1
  else
    echo "[submit][ERR] job ${job} reached terminal but not complete (succeeded=${succeeded} failed=${failed})" >&2
    rc=1
  fi

  if [[ "${rc}" -eq 0 ]]; then
    echo "[submit] <<< job ${job} complete on all ${NODE_COUNT} node(s)"
    dump_job_logs "${job}" 20
  else
    # 失败也必须抓日志（B4）：超时分支已抓过，这里覆盖 FAILED / 终态非 Complete 两类。
    echo "[submit][ERR] job ${job} did NOT complete; dumping pod logs for diagnosis:" >&2
    dump_job_logs "${job}" 40
  fi

  # 无论成功失败都删除 Job，释放节点 CPU/内存配额。失败 Pod（32核/32Gi）若不删
  # 会占住该节点资源，阻碍后续重跑的同节点调度。日志已在上方采集完毕。
  # 删除放在采日志之后：Job 没设 ttlSecondsAfterFinished，避免 TTL 提前删掉 Pod。
  kubectl -n "${NAMESPACE}" delete job "${job}" --ignore-not-found >/dev/null 2>&1 || true
  return "${rc}"
}

# =============================================================================
# 四任务严格串行：前一个 Job 全部 Complete 才提交下一个。
# 任一任务失败立即 FATAL 退出，绝不继续（B2）。
# =============================================================================
for task in "${SYNC_TASKS_ARR[@]}"; do
  yaml="${TMPD}/sync_${task}.yaml"
  # JOB_NAME_<task> 与 WAIT_TIMEOUT_<task> 由 plan.env 提供（间接展开变量名）。
  job_var="JOB_NAME_${task}"
  to_var="WAIT_TIMEOUT_${task}"
  job="${!job_var}"
  timeout="${!to_var}"
  if [[ ! -f "${yaml}" ]]; then
    echo "[submit][FATAL] derived yaml missing for task ${task}: ${yaml}" >&2
    exit 1
  fi
  echo "[submit] === sync task [${task}] ==="
  if ! run_job "${job}" "${yaml}" "${timeout}"; then
    echo "[submit][FATAL] sync task ${task} failed; aborting (no training submitted)." >&2
    exit 1
  fi
done

# =============================================================================
# 4) 渲染并提交训练 PyTorchJob
# =============================================================================
JOB_NAME="${JOB_NAME_PREFIX}-${RUN_ID}"
echo "[submit] rendering ${YAML} -> ${JOB_NAME}"
YAML_RUN="${TMPD}/pytorchjob.yaml"

# =============================================================================
# launcher ConfigMap 必须带 RUN_ID（与 artifact-sync-script-<RUN_ID> 同理）。
# 为什么：CM 名若固定为 `launch-train-script`，并发提交第二个任务时
# `kubectl apply` 会覆盖同一个 CM；ConfigMap 挂载是投影式的，kubelet 会在数十秒
# 内把**正在运行**的 Pod 里的 /opt/launcher/launch_train.sh 内容也换掉。已 exec
# 的进程不受影响，但 restartPolicy: OnFailure 下任何一次重启都会用**另一个实验
# 的启动脚本**起来 —— 静默的实验污染，且几乎无法从日志看出来。
#
# 替换方式（明确选定）：yaml 里 volumes 的 configMap.name 写成占位符
#   name: launch-train-script-__RUN_ID__
# 复用已有的 `s/__RUN_ID__/${RUN_ID}/g` 渲染，无需额外 sed 规则。
# 为兼容尚未加占位符的旧 yaml，下面再补一条兜底 sed：把裸的
#   name: launch-train-script
# 也替换成带 RUN_ID 的形式（顺序上放在 __RUN_ID__ 替换之后，避免二次替换）。
# =============================================================================
LAUNCHER_CM="launch-train-script-${RUN_ID}"
sed -e "s/__RUN_ID__/${RUN_ID}/g" -e "s/__JOB_NAME__/${JOB_NAME}/g" "${YAML}" \
  | sed -E "s/(name:[[:space:]]*)launch-train-script[[:space:]]*$/\1${LAUNCHER_CM}/" \
  > "${YAML_RUN}"

# 渲染后硬校验：yaml 里必须确实引用了带 RUN_ID 的 CM 名。若两条 sed 都没命中
# （例如 yaml 用了别的 CM 名），说明 Pod 会去挂一个我们没创建的 CM，Pod 会卡在
# ContainerCreating —— 与其等集群报错，不如在这里 FATAL。
if ! grep -q "${LAUNCHER_CM}" "${YAML_RUN}"; then
  echo "[submit][FATAL] rendered yaml does not reference launcher ConfigMap ${LAUNCHER_CM};" >&2
  echo "[submit][FATAL]   expected 'name: launch-train-script-__RUN_ID__' (or bare" >&2
  echo "[submit][FATAL]   'name: launch-train-script') under the launcher volume in ${YAML}" >&2
  exit 1
fi

echo "[submit] build launcher ConfigMap ${LAUNCHER_CM} from ${LAUNCH}"
kubectl create configmap "${LAUNCHER_CM}" \
  --from-file=launch_train.sh="${LAUNCH}" \
  -n "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

kubectl -n "${NAMESPACE}" apply -f "${YAML_RUN}"

echo
echo "[submit] done. Monitor with:"
echo "  kubectl -n ${NAMESPACE} get pods -l training.kubeflow.org/job-name=${JOB_NAME} -o wide -w"
echo "  kubectl -n ${NAMESPACE} logs -f ${JOB_NAME}-master-0 -c pytorch"
echo
echo "  # tear down:"
echo "  kubectl -n ${NAMESPACE} delete pytorchjob ${JOB_NAME}"
