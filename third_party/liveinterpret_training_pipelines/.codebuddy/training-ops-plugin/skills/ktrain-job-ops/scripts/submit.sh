#!/usr/bin/env bash
# =============================================================================
# training-ops / ktrain-job-ops / submit.sh
# =============================================================================
# KtrainJob 提交编排。全程：KtrainJob 用 ktrain CLI（validate/submit），
# 派生的同步 Job（batch/v1 Job，不属 KtrainJob 管理）用 kubectl apply/delete。
#
# 步骤：
#   1. 解析任务 yaml（命名空间等）；可选刷 kubecontext（--cluster + --region）
#   1b. __OWNER__ 预渲染：yaml 中所有 __OWNER__ 占位符替换为
#       metadata.annotations["ktrain.oit-svc.com/owner"]（小写、[a-z0-9-]）。
#       CODE_PREFIX / OUTPUT_ROOT 等按人隔离的路径写 __OWNER__ 即可，换人提交不用改。
#   2. sync_code.sh：代码库覆盖式同步到代码 bucket 的 CODE_PREFIX
#   3. 建同步脚本 ConfigMap（artifact-sync-script-<RUN_ID>）
#   4. derive_sync_jobs.py：从任务 yaml 派生同步 Job + plan.env
#   5. 预检白名单节点（Ready 且未 cordon）
#   6. 同步任务严格串行：apply -> 等终态 -> 核对 succeeded==N -> 采日志 -> 删 Job
#   7. 渲染任务 yaml（__RUN_ID__/__JOB_NAME__）+ 注入 launcher ConfigMap 卷
#   8. ktrain job validate + ktrain job submit
#
# 用法：
#   submit.sh --yaml <任务yaml> --repo <本地代码库根> --launcher <入口脚本本地路径> \
#     [--job-name-prefix <前缀>] [--cluster <集群> --region <区域>] \
#     [--pvc <代码PVC>] [--aws-creds-secret <secret名>] \
#     [--deadline-sec <秒>] [--task-deadline <任务名|env名>=<秒>]...
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 参数解析 ----
YAML=""
REPO=""
LAUNCH=""
JOB_NAME_PREFIX=""
CLUSTER=""
REGION=""
PVC_NAME=""
AWS_CREDS_SECRET=""
DERIVE_EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --yaml)              YAML="$2"; shift 2 ;;
    --repo)              REPO="$2"; shift 2 ;;
    --launcher)          LAUNCH="$2"; shift 2 ;;
    --job-name-prefix)   JOB_NAME_PREFIX="$2"; shift 2 ;;
    --cluster)           CLUSTER="$2"; shift 2 ;;
    --region)            REGION="$2"; shift 2 ;;
    --pvc)               PVC_NAME="$2"; shift 2 ;;
    --aws-creds-secret)  AWS_CREDS_SECRET="$2"; shift 2 ;;
    --deadline-sec)      DERIVE_EXTRA+=(--deadline-sec "$2"); shift 2 ;;
    --task-deadline)     DERIVE_EXTRA+=(--task-deadline "$2"); shift 2 ;;
    -h|--help)           sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "[submit][FATAL] unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ -n "${YAML}" && -f "${YAML}" ]]     || { echo "[submit][FATAL] --yaml not found: ${YAML}" >&2; exit 1; }
[[ -n "${REPO}" && -d "${REPO}" ]]     || { echo "[submit][FATAL] --repo not found: ${REPO}" >&2; exit 1; }
[[ -n "${LAUNCH}" && -f "${LAUNCH}" ]] || { echo "[submit][FATAL] --launcher not found: ${LAUNCH}" >&2; exit 1; }
if [[ -n "${CLUSTER}" || -n "${REGION}" ]]; then
  [[ -n "${CLUSTER}" && -n "${REGION}" ]] || { echo "[submit][FATAL] --cluster and --region must be given together" >&2; exit 1; }
fi

# ---- 命名空间：从任务 yaml 解析 ----
NAMESPACE="$(python3 -c '
import sys, yaml
ns = (yaml.safe_load(open(sys.argv[1])).get("metadata") or {}).get("namespace")
if not ns:
    sys.exit("metadata.namespace missing in task yaml")
print(ns)
' "${YAML}")" || { echo "[submit][FATAL] cannot parse namespace from ${YAML}" >&2; exit 1; }

# ---- 任务名前缀：缺省取 yaml 文件名，sanitize 为 DNS-1123 ----
if [[ -z "${JOB_NAME_PREFIX}" ]]; then
  JOB_NAME_PREFIX="$(basename "${YAML}")"
  JOB_NAME_PREFIX="${JOB_NAME_PREFIX%.*}"
fi
JOB_NAME_PREFIX="$(echo "${JOB_NAME_PREFIX}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g')"
[[ -n "${JOB_NAME_PREFIX}" ]] || { echo "[submit][FATAL] empty job name prefix after sanitize" >&2; exit 1; }

RUN_ID="$(LC_ALL=C tr -dc 'a-z0-9' < /dev/urandom | head -c4 2>/dev/null || true)"
RUN_ID="${RUN_ID:-r$(date +%s | tail -c4)}"
TMPD="$(mktemp -d /tmp/ktrainjob.${RUN_ID}.XXXXXX)"
trap 'rm -rf "${TMPD}"' EXIT

LOG="/tmp/submit-ktrain-${RUN_ID}.log"
: > "${LOG}"
echo "[submit] local log -> ${LOG}"
exec > >(tee -a "${LOG}") 2>&1

# ---- __OWNER__ 预渲染：占位符 -> owner 注解（代码同步/派生/渲染/提交全用替换后的文件） ----
if grep -q '__OWNER__' "${YAML}"; then
  OWNER="$(python3 -c '
import sys, yaml
ann = (yaml.safe_load(open(sys.argv[1])).get("metadata") or {}).get("annotations") or {}
print(ann.get("ktrain.oit-svc.com/owner") or "")
' "${YAML}")" || { echo "[submit][FATAL] cannot parse owner annotation from ${YAML}" >&2; exit 1; }
  [[ -n "${OWNER}" ]] || { echo "[submit][FATAL] yaml 含 __OWNER__ 占位符，但未设置 metadata.annotations[ktrain.oit-svc.com/owner]" >&2; exit 1; }
  OWNER="$(echo "${OWNER}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9-]+/-/g; s/^-+|-+$//g')"
  [[ -n "${OWNER}" ]] || { echo "[submit][FATAL] owner 注解清洗后为空" >&2; exit 1; }
  YAML_SUB="${TMPD}/ktrainjob.owner.yaml"
  sed "s/__OWNER__/${OWNER}/g" "${YAML}" > "${YAML_SUB}"
  echo "[submit] owner=${OWNER} (from ktrain.oit-svc.com/owner); __OWNER__ pre-rendered -> ${YAML_SUB}"
  YAML="${YAML_SUB}"
fi

echo "[submit] yaml=${YAML} repo=${REPO} launcher=${LAUNCH}"
echo "[submit] namespace=${NAMESPACE} prefix=${JOB_NAME_PREFIX} run_id=${RUN_ID}"

# ---- 1. 同步代码到代码 bucket ----
SYNC_CODE_ARGS=(--yaml "${YAML}" --repo "${REPO}")
[[ -n "${PVC_NAME}" ]] && SYNC_CODE_ARGS+=(--pvc "${PVC_NAME}")
[[ -n "${CLUSTER}" ]] && SYNC_CODE_ARGS+=(--cluster "${CLUSTER}" --region "${REGION}")
echo "[submit] step 1: sync repo code -> code bucket"
bash "${SCRIPT_DIR}/sync_code.sh" "${SYNC_CODE_ARGS[@]}"

# ---- 2. 建同步脚本 ConfigMap + 派生同步任务 ----
echo "[submit] step 2: build artifact-sync-script-${RUN_ID} ConfigMap + derive sync jobs"
kubectl create configmap "artifact-sync-script-${RUN_ID}" \
  --from-file=sync_artifact.sh="${SCRIPT_DIR}/sync_artifact.sh" \
  -n "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

DERIVE_ARGS=(--yaml "${YAML}" --out "${TMPD}" --run-id "${RUN_ID}")
[[ -n "${AWS_CREDS_SECRET}" ]] && DERIVE_ARGS+=(--aws-creds-secret "${AWS_CREDS_SECRET}")
python3 "${SCRIPT_DIR}/derive_sync_jobs.py" "${DERIVE_ARGS[@]}" ${DERIVE_EXTRA[@]+"${DERIVE_EXTRA[@]}"}

# shellcheck source=/dev/null
source "${TMPD}/plan.env"
read -r -a NODES_ARR <<< "${NODES}"
read -r -a SYNC_TASKS_ARR <<< "${SYNC_TASKS:-}"
read -r -a FSX_ARTIFACTS_ARR <<< "${FSX_ARTIFACTS:-}"

echo "[submit] nodes(${NODE_COUNT}): ${NODES}"
echo "[submit] artifacts derived: ${ARTIFACT_COUNT:-0}"
echo "[submit] sync tasks (serial): ${SYNC_TASKS:-<none>}"
if [[ -n "${FSX_TASKS:-}" ]]; then
  echo "[submit] fsx-direct tasks (read straight from FSX, no sync job): ${FSX_TASKS}"
fi
if [[ ${#FSX_ARTIFACTS_ARR[@]} -gt 0 ]]; then
  echo "[submit] fsx mount in container: ${FSX_MOUNT:-<unset>}"
  for _a in "${FSX_ARTIFACTS_ARR[@]}"; do
    echo "[submit]   fsx-direct artifact: ${_a}"
  done
fi
if [[ "${ARTIFACT_COUNT:-0}" -eq 0 ]]; then
  echo "[submit][WARN] no artifacts derived from yaml env; training must find data by itself" >&2
fi

# ---- 3. 预检白名单节点健康 ----
precheck_nodes() {
  echo "[submit] step 3: checking ${NODE_COUNT} whitelisted node(s) Ready & schedulable..."
  local bad=() node json ready unsched
  for node in "${NODES_ARR[@]}"; do
    if ! json="$(kubectl get node "${node}" -o json 2>/dev/null)"; then
      bad+=("${node}(not-found)")
      continue
    fi
    ready="$(echo "${json}" | python3 -c '
import json,sys
d=json.load(sys.stdin)
c=[x for x in d.get("status",{}).get("conditions",[]) if x.get("type")=="Ready"]
print(c[0].get("status") if c else "Unknown")
')"
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
    echo "[submit][FATAL] aborting; no training submitted." >&2
    exit 1
  fi
  echo "[submit] pre-flight OK: all ${NODE_COUNT} node(s) Ready & schedulable"
}
if [[ ${#SYNC_TASKS_ARR[@]} -gt 0 ]]; then
  precheck_nodes
fi

# ---- 采集 sync Job 全部 Pod 日志 ----
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

# ---- 轮询等待 sync Job 终态 ----
POLL_STEP=10
await_job() {
  local job="$1" timeout="$2" elapsed=0 cond
  while (( elapsed < timeout )); do
    cond="$(kubectl -n "${NAMESPACE}" get job "${job}" \
              -o jsonpath='{range .status.conditions[*]}{.type}={.status};{end}' \
              2>/dev/null || true)"
    [[ "${cond}" == *"Complete=True"* ]] && return 0
    [[ "${cond}" == *"Failed=True"* ]]   && return 2
    sleep "${POLL_STEP}"; elapsed=$(( elapsed + POLL_STEP ))
  done
  return 1
}

# ---- 提交并等待单个 sync Job ----
run_job() {
  local job="$1" yaml="$2" timeout="$3" rc=0 failed succeeded
  echo "[submit] >>> applying sync job ${job} (wait timeout=${timeout}s)"
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
    :
  elif [[ "${aw}" -eq 2 || "${failed}" -gt 0 ]]; then
    echo "[submit][ERR] sync job ${job} FAILED: ${failed} pod(s) errored" >&2
    rc=1
  elif [[ "${aw}" -eq 1 ]]; then
    echo "[submit][ERR] sync job ${job} not complete within ${timeout}s; " \
         "succeeded=${succeeded} failed=${failed}" >&2
    kubectl -n "${NAMESPACE}" describe "job/${job}" 2>&1 | tail -20 | sed 's/^/      /' >&2 || true
    rc=1
  else
    echo "[submit][ERR] sync job ${job} terminal but not complete (succeeded=${succeeded} failed=${failed})" >&2
    rc=1
  fi

  if [[ "${rc}" -eq 0 ]]; then
    echo "[submit] <<< sync job ${job} complete on all ${NODE_COUNT} node(s)"
  else
    echo "[submit][ERR] sync job ${job} did NOT complete; dumping pod logs:" >&2
    dump_job_logs "${job}" 40
  fi

  kubectl -n "${NAMESPACE}" delete job "${job}" --ignore-not-found >/dev/null 2>&1 || true
  return "${rc}"
}

# ---- 4. 同步任务严格串行 ----
if [[ ${#SYNC_TASKS_ARR[@]} -eq 0 ]]; then
  echo "[submit] no sync tasks; skip artifact sync"
fi
for task in ${SYNC_TASKS:-}; do
  yaml="${TMPD}/sync_${task}.yaml"
  # 任务名含连字符时不能直接做 shell 变量名后缀；与 derive_sync_jobs.py 生成侧一致转下划线
  task_var="${task//-/_}"
  job_var="JOB_NAME_${task_var}"
  to_var="WAIT_TIMEOUT_${task_var}"
  job="${!job_var}"
  timeout="${!to_var}"
  if [[ ! -f "${yaml}" ]]; then
    echo "[submit][FATAL] derived yaml missing for task ${task}: ${yaml}" >&2
    exit 1
  fi
  echo "[submit] step 4: === sync task [${task}] ==="
  if ! run_job "${job}" "${yaml}" "${timeout}"; then
    echo "[submit][FATAL] sync task ${task} failed; aborting (no training submitted)." >&2
    exit 1
  fi
done

# ---- 5. 渲染任务 yaml + launcher ConfigMap ----
JOB_NAME="${JOB_NAME_PREFIX}-${RUN_ID}"
echo "[submit] step 5: render ${YAML} -> ${JOB_NAME}"
YAML_RUN="${TMPD}/ktrainjob.yaml"
sed -e "s/__RUN_ID__/${RUN_ID}/g" -e "s/__JOB_NAME__/${JOB_NAME}/g" "${YAML}" > "${YAML_RUN}"

LAUNCHER_CM="launch-train-script-${RUN_ID}"
echo "[submit] build launcher ConfigMap ${LAUNCHER_CM} from ${LAUNCH}"
kubectl create configmap "${LAUNCHER_CM}" \
  --from-file="$(basename "${LAUNCH}")=${LAUNCH}" \
  -n "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

python3 - "${YAML_RUN}" "${LAUNCHER_CM}" "$(basename "${LAUNCH}")" <<'PY'
import json, sys, yaml
path, cm, launcher_file = sys.argv[1], sys.argv[2], sys.argv[3]
doc = yaml.safe_load(open(path))
ann = (doc.get('metadata') or {}).get('annotations') or {}
raw = ann.get('ktrain.oit-svc.com/render-extra')
if not raw:
    sys.exit('[submit][FATAL] render-extra not found in rendered yaml')
data = json.loads(raw)
vols = data.setdefault('extra_volumes', [])
if not any(v.get('name') == 'launcher' for v in vols):
    vols.append({'name': 'launcher', 'configMap': {'name': cm, 'defaultMode': 0o755}})
mounts = data.setdefault('extra_volume_mounts', [])
if not any(m.get('name') == 'launcher' for m in mounts):
    mounts.append({'name': 'launcher', 'mountPath': '/opt/launcher'})
ann['ktrain.oit-svc.com/render-extra'] = json.dumps(data, ensure_ascii=False)
with open(path, 'w') as f:
    yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
print(f'[submit] injected launcher ConfigMap volume ({cm}, file {launcher_file}) into render-extra')
PY

# ---- 6. ktrain job validate + submit ----
echo "[submit] step 6: validating KtrainJob..."
ktrain job validate -f "${YAML_RUN}" 2>&1 | tail -3

echo "[submit] submitting KtrainJob via ktrain job submit..."
ktrain job submit -f "${YAML_RUN}" --yes 2>&1 | tail -10

echo
echo "[submit] done. Monitor with:"
echo "  ktrain job status ${JOB_NAME} --namespace ${NAMESPACE}"
echo "  ktrain job logs ${JOB_NAME} --namespace ${NAMESPACE} --follow"
echo
echo "  # stop (release resources, keep job object):"
echo "  kubectl -n ${NAMESPACE} patch pytorchjob ${JOB_NAME} --type merge -p '{\"spec\":{\"runPolicy\":{\"stop\":true}}}'"
echo "  # fully delete (non-default, only when the job object is no longer needed):"
echo "  ktrain job delete ${JOB_NAME} --namespace ${NAMESPACE} --force"
