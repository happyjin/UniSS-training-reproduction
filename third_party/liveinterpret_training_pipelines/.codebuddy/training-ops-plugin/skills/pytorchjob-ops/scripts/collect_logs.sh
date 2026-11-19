#!/usr/bin/env bash
# =============================================================================
# training-ops / pytorchjob-ops / collect_logs.sh
#
# 通用日志收集脚本：把 PyTorchJob 各 pod 的全量日志下载到本地，遵守
# “全量日志本地分析”纪律（禁止 cherry-pick）。不依赖特定集群/项目。
# =============================================================================
# 用法：
#   bash collect_logs.sh <JOB_NAME> [NAMESPACE]
#     JOB_NAME    必填，PyTorchJob 名称
#     NAMESPACE   可选，k8s 命名空间；也可用环境变量 K8S_NS 传入
#
# 输出（写入 /tmp/pytorchjob-collect/<JOB_NAME>/）：
#   pods.txt / <role>-<idx>.log / summary.txt
# 输出（stdout，KEY=VALUE）：
#   COLLECT_OK / JOB_NAME / LOG_LINES / OUT_BUNDLE / FILES / FAIL_REASON
# =============================================================================
set -euo pipefail

JOB_NAME="${1:-}"
NS="${2:-${K8S_NS:-}}"
if [[ -z "${JOB_NAME}" || -z "${NS}" ]]; then
  echo "usage: collect_logs.sh <JOB_NAME> [NAMESPACE]" >&2
  echo "  (NAMESPACE 也可通过环境变量 K8S_NS 传入)" >&2
  exit 2
fi

OUT="/tmp/pytorchjob-collect/${JOB_NAME}"
mkdir -p "${OUT}"

echo "[collect] job=${JOB_NAME} ns=${NS}"
echo "[collect] out=${OUT}"
echo ""

echo "=== pod status ==="
kubectl -n "${NS}" get pods -l "training.kubeflow.org/job-name=${JOB_NAME}" -o wide > "${OUT}/pods.txt" 2>&1 || true
cat "${OUT}/pods.txt"
echo ""

TOTAL_LOG_LINES=0
while read -r pod; do
    [[ -z "${pod}" ]] && continue
    f="${OUT}/${pod}.log"
    echo "=== downloading logs: ${pod} ==="
    kubectl -n "${NS}" logs "${pod}" -c pytorch > "${f}" 2>&1 || true
    n=$(wc -l < "${f}")
    TOTAL_LOG_LINES=$((TOTAL_LOG_LINES + n))
    echo "    -> ${f} (${n} lines)"
done < <(kubectl -n "${NS}" get pods -l "training.kubeflow.org/job-name=${JOB_NAME}" -o name 2>/dev/null | sed 's|pod/||')

echo ""

COLLECT_OK="yes"
if [[ "${TOTAL_LOG_LINES}" -lt 3 ]]; then
    COLLECT_OK="no"
fi

{
    echo "================================================================="
    echo "  STRUCTURED OUTPUT"
    echo "================================================================="
    echo "COLLECT_OK=${COLLECT_OK}"
    echo "JOB_NAME=${JOB_NAME}"
    echo "LOG_LINES=${TOTAL_LOG_LINES}"
    echo "OUT_BUNDLE=${OUT}"
    echo "FILES=$(ls "${OUT}" | tr '\n' ',')"
    if [[ "${COLLECT_OK}" == "no" ]]; then
        echo "FAIL_REASON=logs-fetch-failed"
    fi
    echo "================================================================="
} | tee "${OUT}/summary.txt"

[[ "${COLLECT_OK}" == "no" ]] && exit 2
exit 0
