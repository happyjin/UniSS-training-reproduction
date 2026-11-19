#!/usr/bin/env bash
# =============================================================================
# training-ops / ktrain-job-ops / collect_logs.sh
# =============================================================================
# 收集 ktrain 任务诊断所需的全部远程信息：任务状态 + 全量日志。
# 全程用 ktrain CLI（ktrain job status / ktrain job logs），不直接用 kubectl。
#
# 用法：
#   bash collect_logs.sh <JOB_NAME> <NAMESPACE>
#     JOB_NAME    必填，KtrainJob 名称
#     NAMESPACE   必填，k8s 命名空间（也可用环境变量 K8S_NS 传入）；
#                 取自任务 yaml metadata.namespace，无默认值
#
# 输出（写入 /tmp/ktrain-collect/<JOB_NAME>/）：status.txt / logs.txt / summary.txt
# 输出（stdout，KEY=VALUE）：
#   COLLECT_OK / JOB_NAME / JOB_STATE / LOG_LINES / STATUS_LINES / OUT_BUNDLE / FAIL_REASON
# =============================================================================
set -euo pipefail

JOB_NAME="${1:?用法: $0 <JOB_NAME> <NAMESPACE>}"
NS="${2:-${K8S_NS:?命名空间必填：传第二个参数或设置 K8S_NS（取自任务 yaml metadata.namespace）}}"
OUT="/tmp/ktrain-collect/${JOB_NAME}"
mkdir -p "${OUT}"

echo "[collect] job=${JOB_NAME} ns=${NS}"
echo "[collect] out=${OUT}"
echo ""

echo "=== ktrain job status ==="
ktrain job status "${JOB_NAME}" --namespace "${NS}" > "${OUT}/status.txt" 2>&1 || true
STATUS_LINES=$(wc -l < "${OUT}/status.txt")
JOB_STATE=$(grep -oiE '\b(Pending|Creating|Running|Succeeded|Failed|Error|Unknown)\b' "${OUT}/status.txt" | head -1 || echo "UNKNOWN")
echo "  JOB_STATE=${JOB_STATE} (${STATUS_LINES} lines)"
echo ""

echo "=== ktrain job logs ==="
ktrain job logs "${JOB_NAME}" --namespace "${NS}" > "${OUT}/logs.txt" 2>&1 || true
LOG_LINES=$(wc -l < "${OUT}/logs.txt")
echo "  ${LOG_LINES} lines"
if [[ "${LOG_LINES}" -lt 10 ]]; then
    echo "[collect] retry with --tail 5000"
    ktrain job logs "${JOB_NAME}" --namespace "${NS}" --tail 5000 > "${OUT}/logs.txt" 2>&1 || true
    LOG_LINES=$(wc -l < "${OUT}/logs.txt")
fi
echo ""

COLLECT_OK="yes"
if [[ "${STATUS_LINES}" -lt 3 && "${LOG_LINES}" -lt 3 ]]; then
    COLLECT_OK="no"
fi

{
    echo "================================================================="
    echo "  STRUCTURED OUTPUT"
    echo "================================================================="
    echo "COLLECT_OK=${COLLECT_OK}"
    echo "JOB_NAME=${JOB_NAME}"
    echo "JOB_STATE=${JOB_STATE}"
    echo "LOG_LINES=${LOG_LINES}"
    echo "STATUS_LINES=${STATUS_LINES}"
    echo "OUT_BUNDLE=${OUT}"
    echo "FILES=status.txt,logs.txt,summary.txt"
    if [[ "${COLLECT_OK}" == "no" ]]; then
        echo "FAIL_REASON=status-or-logs-fetch-failed"
    fi
    echo "================================================================="
} | tee "${OUT}/summary.txt"

[[ "${COLLECT_OK}" == "no" ]] && exit 2
exit 0
