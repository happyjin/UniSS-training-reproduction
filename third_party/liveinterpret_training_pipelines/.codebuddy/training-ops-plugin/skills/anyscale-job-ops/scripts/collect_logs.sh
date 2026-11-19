#!/usr/bin/env bash
# =============================================================================
# training-ops / anyscale-job-ops / collect_logs.sh
# =============================================================================
# 收集诊断 prodjob 所需的全部远程信息：任务状态 + 全量日志。
# 不读取任何本地 repo 文件，所有信息通过 Anyscale CLI 远程获取。
#
# 用法：
#   bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/collect_logs.sh <PRODJOB_ID>
#
# 输入参数：
#   <PRODJOB_ID>  — 如 prodjob_abc123def456
#
# 输出（写入 /tmp/anyscale-collect/<PRODJOB_ID>/）：
#   job_status.txt / job_logs.txt / job_config.txt / context.txt / summary.txt
#
# 输出（stdout，KEY=VALUE 格式）：
#   COLLECT_OK / PRODJOB_ID / JOB_STATE / LOG_LINES / STATUS_LINES /
#   OUT_BUNDLE / FAIL_REASON
#
# 退出码：0 = 收集成功；2 = 参数错误或无法获取任务信息。
# =============================================================================
set -euo pipefail

PRODJOB_ID="${1:?用法: $0 <PRODJOB_ID>}"
OUT="/tmp/anyscale-collect/${PRODJOB_ID}"
mkdir -p "${OUT}"

echo "[collect] prodjob=${PRODJOB_ID}"
echo "[collect] out=${OUT}"
echo ""

echo "=== anyscale job status ==="
STATUS_FILE="${OUT}/job_status.txt"
{
    echo "=== anyscale job status --id ${PRODJOB_ID} ==="
    echo ""
    anyscale job status --id "${PRODJOB_ID}" 2>&1 || echo "STATUS_FETCH_FAILED"
} > "${STATUS_FILE}"

STATUS_LINES=$(wc -l < "${STATUS_FILE}")
echo "    → ${STATUS_FILE} (${STATUS_LINES} lines)"

JOB_STATE=$(grep -oiE '\b(STARTING|RUNNING|SUCCEEDED|FAILED|TERMINATED|OUT_OF_RETRIES|QUEUED|UNKNOWN)\b' "${STATUS_FILE}" | head -1 || echo "UNKNOWN")
echo "    → JOB_STATE=${JOB_STATE}"
echo ""

echo "=== anyscale job logs ==="
LOGS_FILE="${OUT}/job_logs.txt"

echo "[collect] pulling full logs (may take 30-90s)…"
{
    echo "=== anyscale job logs --id ${PRODJOB_ID} ==="
    echo ""
    anyscale job logs --id "${PRODJOB_ID}" 2>&1 || echo "LOGS_FETCH_FAILED"
} > "${LOGS_FILE}"

LOG_LINES=$(wc -l < "${LOGS_FILE}")
echo "    → ${LOGS_FILE} (${LOG_LINES} lines)"

if [[ "${LOG_LINES}" -lt 10 ]]; then
    echo "[collect] log is very short, retrying with --max-lines 100000..."
    anyscale job logs --id "${PRODJOB_ID}" --max-lines 100000 > "${LOGS_FILE}" 2>&1 || true
    LOG_LINES=$(wc -l < "${LOGS_FILE}")
    echo "    → retry got ${LOG_LINES} lines"
fi
echo ""

echo "=== anyscale job config (best-effort) ==="
CONFIG_FILE="${OUT}/job_config.txt"

if anyscale job get --id "${PRODJOB_ID}" > "${CONFIG_FILE}" 2>&1; then
    echo "    → ${CONFIG_FILE} ($(wc -l < "${CONFIG_FILE}") lines)"
elif anyscale job describe --id "${PRODJOB_ID}" > "${CONFIG_FILE}" 2>&1; then
    echo "    → ${CONFIG_FILE} ($(wc -l < "${CONFIG_FILE}") lines)"
else
    echo "    → (anyscale job get/describe not available, skipping)"
    echo "JOB_CONFIG_AVAILABLE=no" > "${CONFIG_FILE}"
fi
echo ""

echo "=== extracting context from remote data ==="
CONTEXT_FILE="${OUT}/context.txt"

{
    echo "=== job state ==="
    echo "JOB_STATE=${JOB_STATE}"
    echo ""
    echo "=== compute config (from status output) ==="
    grep -iE 'compute.config|config.name' "${STATUS_FILE}" | head -5 || echo "(not found in status output)"
    echo ""
    echo "=== image URI (from status output) ==="
    grep -iE 'image|uri' "${STATUS_FILE}" | grep -v '^$' | head -5 || echo "(not found in status output)"
    echo ""
    echo "=== first 20 lines of logs (for entrypoint chain inspection) ==="
    head -20 "${LOGS_FILE}" 2>/dev/null || echo "(no logs)"
    echo ""
    echo "=== entrypoint commands (grep from logs) ==="
    grep -E '^[#$>] |entrypoint|command:' "${LOGS_FILE}" 2>/dev/null | head -20 || echo "(not found)"
    echo ""
} > "${CONTEXT_FILE}"

echo "    → ${CONTEXT_FILE}"
echo ""

COLLECT_OK="yes"
if [[ "${STATUS_LINES}" -lt 3 ]]; then
    COLLECT_OK="no"
fi
if [[ "${LOG_LINES}" -lt 3 && "${JOB_STATE}" == "UNKNOWN" ]]; then
    COLLECT_OK="no"
fi

echo "================================================================="
echo "  STRUCTURED OUTPUT"
echo "================================================================="
echo "COLLECT_OK=${COLLECT_OK}"
echo "PRODJOB_ID=${PRODJOB_ID}"
echo "JOB_STATE=${JOB_STATE}"
echo "LOG_LINES=${LOG_LINES}"
echo "STATUS_LINES=${STATUS_LINES}"
echo "OUT_BUNDLE=${OUT}"
echo "FILES=job_status.txt,job_logs.txt,job_config.txt,context.txt,summary.txt"

if [[ "${COLLECT_OK}" == "no" ]]; then
    echo "FAIL_REASON=status-or-logs-fetch-failed"
    echo "================================================================="
    {
        echo "COLLECT_OK=no"
        echo "FAIL_REASON=status-or-logs-fetch-failed"
        echo "OUT_BUNDLE=${OUT}"
    } > "${OUT}/summary.txt"
    exit 2
fi

echo "================================================================="

{
    echo "COLLECT_OK=yes"
    echo "PRODJOB_ID=${PRODJOB_ID}"
    echo "JOB_STATE=${JOB_STATE}"
    echo "LOG_LINES=${LOG_LINES}"
    echo "STATUS_LINES=${STATUS_LINES}"
    echo "OUT_BUNDLE=${OUT}"
} > "${OUT}/summary.txt"

exit 0
