#!/usr/bin/env bash
# =============================================================================
# training-ops / ktrain-job-ops / query_job.sh
# =============================================================================
# 查询 ktrain 训练任务的进度。通过 ktrain CLI 拉取 status + logs，解析
# step / loss / token_acc / checkpoint / 异常信号。
#
# 全程用 ktrain CLI（ktrain job status / ktrain job logs），不直接用 kubectl。
#
# 用法：
#   bash query_job.sh <JOB_NAME> <NAMESPACE>
#     JOB_NAME    必填，KtrainJob 名称
#     NAMESPACE   必填，k8s 命名空间（也可用环境变量 K8S_NS 传入）；
#                 取自任务 yaml metadata.namespace，无默认值
#
# 输出（写入 /tmp/ktrain-progress/<JOB_NAME>/）：status.txt / logs.txt / summary.txt
# 输出格式（KEY=VALUE，stdout）：
#   JOB_STATE / STEP_PROGRESS / STEP_TIME / LATEST_LOSS / TOKEN_ACC /
#   GRAD_NORM / LR_LATEST / SAVE_CHECKPOINTS / HEALTH / HEALTH_NOTES /
#   LOG_LINES / OUT_BUNDLE
#
# 全程只读。
# =============================================================================
set -euo pipefail

JOB_NAME="${1:?用法: $0 <JOB_NAME> <NAMESPACE>}"
NS="${2:-${K8S_NS:?命名空间必填：传第二个参数或设置 K8S_NS（取自任务 yaml metadata.namespace）}}"
OUT="/tmp/ktrain-progress/${JOB_NAME}"
mkdir -p "${OUT}"

echo "[query] job=${JOB_NAME} ns=${NS}"
echo "[query] out=${OUT}"
echo ""

echo "=== ktrain job status ==="
ktrain job status "${JOB_NAME}" --namespace "${NS}" > "${OUT}/status.txt" 2>&1 || {
    echo "[query] WARN: ktrain job status failed" >&2
}
cat "${OUT}/status.txt" | head -40
echo ""

JOB_STATE=$(grep -oiE '\b(Pending|Creating|Running|Succeeded|Failed|Error|Unknown)\b' "${OUT}/status.txt" | head -1 || echo "UNKNOWN")

echo "[query] pulling logs..."
ktrain job logs "${JOB_NAME}" --namespace "${NS}" > "${OUT}/logs.txt" 2>&1 || true
LOG_LINES=$(wc -l < "${OUT}/logs.txt")
echo "    → ${OUT}/logs.txt (${LOG_LINES} lines)"
echo ""

LOG="${OUT}/logs.txt"

if [[ "${LOG_LINES}" -lt 5 ]]; then
    echo "[query] log very short, try --tail 2000"
    ktrain job logs "${JOB_NAME}" --namespace "${NS}" --tail 2000 > "${OUT}/logs.txt" 2>&1 || true
    LOG_LINES=$(wc -l < "${OUT}/logs.txt")
fi

# 进度解析（swift trainer 日志格式）
LATEST_LOSS_LINE=$(grep -oE "'loss': [0-9.]+, 'grad_norm': [0-9.]+, 'learning_rate': [0-9.e-]+,.*'global_step/max_steps': '[0-9]+/[0-9]+'" "${LOG}" | tail -1 || true)
CURRENT_STEP=""
MAX_STEPS=""
if [[ -n "${LATEST_LOSS_LINE}" ]]; then
    CURRENT_STEP=$(echo "${LATEST_LOSS_LINE}" | grep -oE "'global_step/max_steps': '[0-9]+/[0-9]+'" | grep -oE '[0-9]+/[0-9]+' | head -1 | cut -d/ -f1)
    MAX_STEPS=$(echo "${LATEST_LOSS_LINE}" | grep -oE "'global_step/max_steps': '[0-9]+/[0-9]+'" | grep -oE '[0-9]+/[0-9]+' | head -1 | cut -d/ -f2)
fi
PROGRESS_PCT=""
if [[ -n "${CURRENT_STEP}" && -n "${MAX_STEPS}" && "${MAX_STEPS}" -gt 0 ]] 2>/dev/null; then
    PROGRESS_PCT=$(awk -v c="${CURRENT_STEP}" -v m="${MAX_STEPS}" 'BEGIN{printf "%.1f", c*100/m}')
fi

TQDM_LATEST=$(grep -oE "Train: +[0-9]+%[|][^|]*[|] +[0-9]+/[0-9]+ \[[^]]+\]" "${LOG}" | tail -1 || true)
STEP_TIME_RAW=$(echo "${TQDM_LATEST}" | grep -oE '[0-9]+\.[0-9]+s/it' | tail -1 || true)
[[ -z "${STEP_TIME_RAW}" ]] && STEP_TIME_RAW=$(echo "${TQDM_LATEST}" | grep -oE '[0-9]+\.[0-9]+it/s' | tail -1 || true)

RECENT_LOSSES=""
if grep -q "'loss'" "${LOG}" 2>/dev/null; then
    RECENT_LOSSES=$(grep -E "'loss'" "${LOG}" | grep "'global_step/max_steps'" | tail -3 | \
        grep -oE "'loss': [0-9.]+.*'global_step/max_steps': '[0-9]+" | \
        sed "s/.*'loss': \([0-9.]*\).*global_step\/max_steps': '\([0-9]*\)/\2:\1/" | \
        awk -F: '{printf "step %s: %.3f | ", $1, $2}' | sed 's/| $//' || true)
fi
TOKEN_ACC=$(grep -oE "'token_acc': [0-9.]+" "${LOG}" | tail -1 | grep -oE '[0-9.]+' || true)
GRAD_NORM=$(grep -oE "'grad_norm': [0-9.]+" "${LOG}" | tail -1 | grep -oE '[0-9.]+' || true)
LR_LATEST=$(grep -oE "'learning_rate': [0-9.e-]+" "${LOG}" | tail -1 | grep -oE '[0-9.e-]+' || true)
CKPT_LIST=""
if grep -q "checkpoint-[0-9]" "${LOG}" 2>/dev/null; then
    CKPT_LIST=$(grep -oE 'checkpoint-[0-9]+' "${LOG}" | sort -u -t- -k2 -n | tr '\n' ',' | sed 's/,$//')
fi

HEALTH="OK"
HEALTH_NOTES=""
if grep -qE 'RuntimeError|Cannot re-initialize CUDA|CUDA out of memory|NCCL.*ERROR|Xid [0-9]+' "${LOG}" 2>/dev/null; then
    HEALTH="FAIL"
    HEALTH_NOTES=$(grep -nE 'RuntimeError|Cannot re-initialize CUDA|CUDA out of memory|NCCL.*ERROR|Xid [0-9]+' "${LOG}" 2>/dev/null | head -3 | cut -d: -f2- | tr '\n' ';' | sed 's/;$//' || true)
fi
if [[ "${HEALTH}" == "OK" ]] && grep -qE 'UserWarning|FutureWarning' "${LOG}" 2>/dev/null; then
    HEALTH="WARN"
    HEALTH_NOTES="benign deprecation warnings present"
fi
if [[ "${JOB_STATE}" =~ ^(Failed|Error)$ ]]; then
    HEALTH="FAIL"
    HEALTH_NOTES="${HEALTH_NOTES} job_state=${JOB_STATE}"
fi

{
    echo "================================================================="
    echo "  STRUCTURED OUTPUT"
    echo "================================================================="
    echo "JOB_STATE=${JOB_STATE}"
    if [[ -n "${CURRENT_STEP}" && -n "${MAX_STEPS}" ]]; then
        if [[ -n "${PROGRESS_PCT}" ]]; then
            echo "STEP_PROGRESS=${CURRENT_STEP}/${MAX_STEPS} (${PROGRESS_PCT}%)"
        else
            echo "STEP_PROGRESS=${CURRENT_STEP}/${MAX_STEPS}"
        fi
    else
        echo "STEP_PROGRESS=N/A"
    fi
    [[ -n "${STEP_TIME_RAW}" ]] && echo "STEP_TIME=${STEP_TIME_RAW}"
    [[ -n "${RECENT_LOSSES}" ]] && echo "LATEST_LOSS=${RECENT_LOSSES}"
    [[ -n "${TOKEN_ACC}" ]]     && echo "TOKEN_ACC=${TOKEN_ACC}"
    [[ -n "${GRAD_NORM}" ]]     && echo "GRAD_NORM=${GRAD_NORM}"
    [[ -n "${LR_LATEST}" ]]     && echo "LR_LATEST=${LR_LATEST}"
    [[ -n "${CKPT_LIST}" ]]     && echo "SAVE_CHECKPOINTS=${CKPT_LIST}"
    echo "HEALTH=${HEALTH}"
    [[ -n "${HEALTH_NOTES}" ]]  && echo "HEALTH_NOTES=${HEALTH_NOTES}"
    echo "OUT_BUNDLE=${OUT}"
    echo "LOG_LINES=${LOG_LINES}"
    echo "================================================================="
} | tee "${OUT}/summary.txt"

exit 0
