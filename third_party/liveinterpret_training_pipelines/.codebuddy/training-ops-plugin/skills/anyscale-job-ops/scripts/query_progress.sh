#!/usr/bin/env bash
# =============================================================================
# training-ops / anyscale-job-ops / query_progress.sh
# =============================================================================
# 查询训练 prodjob 的进度。通过 Anyscale CLI 拉取 status + logs，用 grep/awk/sed
# 提取 step / loss / token_acc / checkpoint / 异常信号。
# 不读取任何本地 repo 文件。
#
# 用法：
#   bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/query_progress.sh <PRODJOB_ID>
#
# 输入参数：
#   <PRODJOB_ID>  — 如 prodjob_abc123def456
#
# 输出（写入 /tmp/anyscale-progress/<PRODJOB_ID>/）：
#   status.txt / logs.txt / summary.txt
#
# 输出格式（KEY=VALUE，stdout 同时输出）：
#   JOB_STATE / STEP_PROGRESS / STEP_TIME / LATEST_LOSS / TOKEN_ACC / GRAD_NORM /
#   LR_LATEST / SAVE_CHECKPOINTS / HEALTH / HEALTH_NOTES / FINAL_CHECKPOINT_S3 /
#   LOG_LINES / OUT_BUNDLE
#
# 全程只读，不 follow 日志，不 cancel。
# =============================================================================
set -euo pipefail

PRODJOB_ID="${1:?用法: $0 <PRODJOB_ID>}"
OUT="/tmp/anyscale-progress/${PRODJOB_ID}"
mkdir -p "${OUT}"

echo "[query] prodjob=${PRODJOB_ID}"
echo "[query] out=${OUT}"
echo ""

# =============================================================================
# (1) 获取任务状态
# =============================================================================
echo "=== anyscale job status ==="
anyscale job status --id "${PRODJOB_ID}" > "${OUT}/status.txt" 2>&1 || {
    echo "[query] WARN: anyscale job status failed" >&2
}
cat "${OUT}/status.txt" | head -40
echo ""

JOB_STATE=$(grep -oiE '\b(STARTING|RUNNING|SUCCEEDED|FAILED|TERMINATED|OUT_OF_RETRIES|QUEUED|UNKNOWN)\b' "${OUT}/status.txt" | head -1 || echo "UNKNOWN")

CREATED_AT=$(grep -oiE '[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2}' "${OUT}/status.txt" | head -1 || true)
STARTED_AT=$(grep -iE 'start' "${OUT}/status.txt" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1 || true)
if [[ "${CREATED_AT}" == "${STARTED_AT}" ]]; then
    STARTED_AT=""
fi

# =============================================================================
# (2) 获取日志
# =============================================================================
echo "[query] pulling full logs (may take 30-90s)…"
anyscale job logs --id "${PRODJOB_ID}" > "${OUT}/logs.txt" 2>&1 || true
LOG_LINES=$(wc -l < "${OUT}/logs.txt")
echo "    → ${OUT}/logs.txt (${LOG_LINES} lines)"
echo ""

LOG="${OUT}/logs.txt"

if [[ "${LOG_LINES}" -lt 5 ]]; then
    echo "[query] log is very short (${LOG_LINES} lines), trying with --max-lines..."
    anyscale job logs --id "${PRODJOB_ID}" --max-lines 100000 > "${OUT}/logs.txt" 2>&1 || true
    LOG_LINES=$(wc -l < "${OUT}/logs.txt")
    echo "    → retry got ${LOG_LINES} lines"
    echo ""
fi

# =============================================================================
# (3) 进度信息提取
# =============================================================================
TQDM_LATEST=$(grep -oE "Train: +[0-9]+%[|][^|]*[|] +[0-9]+/[0-9]+ \[[^]]+\]" "${LOG}" | tail -1 || true)
if [[ -z "${TQDM_LATEST}" ]]; then
    TQDM_LATEST=$(grep -E "Train:.*%\|.*\|" "${LOG}" | tail -1 || true)
fi

LATEST_LOSS_LINE=$(grep -oE "'loss': [0-9.]+, 'grad_norm': [0-9.]+, 'learning_rate': [0-9.e-]+,.*'global_step/max_steps': '[0-9]+/[0-9]+'" "${LOG}" | tail -1 || true)
if [[ -z "${LATEST_LOSS_LINE}" ]]; then
    LATEST_LOSS_LINE=$(grep -E "'loss'.*'global_step/max_steps'" "${LOG}" | tail -1 || true)
fi

CURRENT_STEP=""
MAX_STEPS=""
if [[ -n "${LATEST_LOSS_LINE}" ]]; then
    CURRENT_STEP=$(echo "${LATEST_LOSS_LINE}" | grep -oE "'global_step/max_steps': '[0-9]+/[0-9]+'" | grep -oE '[0-9]+/[0-9]+' | head -1 | cut -d/ -f1)
    MAX_STEPS=$(echo "${LATEST_LOSS_LINE}" | grep -oE "'global_step/max_steps': '[0-9]+/[0-9]+'" | grep -oE '[0-9]+/[0-9]+' | head -1 | cut -d/ -f2)
elif [[ -n "${TQDM_LATEST}" ]]; then
    CURRENT_STEP=$(echo "${TQDM_LATEST}" | grep -oE '[0-9]+/[0-9]+' | head -1 | cut -d/ -f1 || true)
    MAX_STEPS=$(echo "${TQDM_LATEST}" | grep -oE '[0-9]+/[0-9]+' | head -1 | cut -d/ -f2 || true)
fi

PROGRESS_PCT=""
if [[ -n "${CURRENT_STEP}" && -n "${MAX_STEPS}" && "${MAX_STEPS}" -gt 0 ]] 2>/dev/null; then
    PROGRESS_PCT=$(awk -v c="${CURRENT_STEP}" -v m="${MAX_STEPS}" 'BEGIN{printf "%.1f", c*100/m}')
fi

STEP_TIME_RAW=$(echo "${TQDM_LATEST}" | grep -oE '[0-9]+\.[0-9]+s/it' | tail -1 || true)
if [[ -z "${STEP_TIME_RAW}" ]]; then
    STEP_TIME_RAW=$(echo "${TQDM_LATEST}" | grep -oE '[0-9]+\.[0-9]+it/s' | tail -1 || true)
fi

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

# =============================================================================
# (4) 健康检查
# =============================================================================
HEALTH="OK"
HEALTH_NOTES=""

if grep -qE 'RuntimeError|Cannot re-initialize CUDA|CUDA out of memory|NCCL.*ERROR|Xid [0-9]+' "${LOG}" 2>/dev/null; then
    HEALTH="FAIL"
    HEALTH_NOTES=$(grep -nE 'RuntimeError|Cannot re-initialize CUDA|CUDA out of memory|NCCL.*ERROR|Xid [0-9]+' "${LOG}" 2>/dev/null | head -3 | cut -d: -f2- | tr '\n' ';' | sed 's/;$//' || true)
fi

if [[ "${HEALTH}" == "OK" ]] && grep -qE 'NCCL_ASYNC_ERROR_HANDLING.*deprecated|UserWarning|FutureWarning' "${LOG}" 2>/dev/null; then
    HEALTH="WARN"
    HEALTH_NOTES="benign deprecation warnings present"
fi

if [[ "${JOB_STATE}" =~ ^(FAILED|OUT_OF_RETRIES|TERMINATED)$ ]]; then
    HEALTH="FAIL"
    HEALTH_NOTES="${HEALTH_NOTES} job_state=${JOB_STATE}"
fi

# =============================================================================
# (5) Final checkpoint S3 path（仅 SUCCEEDED 时）
# =============================================================================
FINAL_S3=""
if [[ "${JOB_STATE}" == "SUCCEEDED" ]]; then
    FINAL_S3=$(grep -oE 's3://[^[:space:]]+checkpoints/[^[:space:]]+' "${LOG}" | tail -1 || true)
fi

# =============================================================================
# (6) 输出
# =============================================================================
{
    echo "================================================================="
    echo "  STRUCTURED OUTPUT"
    echo "================================================================="
    echo "JOB_STATE=${JOB_STATE}"
    [[ -n "${CREATED_AT}" ]] && echo "CREATED_AT=${CREATED_AT}"
    [[ -n "${STARTED_AT}" ]] && echo "STARTED_AT=${STARTED_AT}"
    if [[ -n "${CURRENT_STEP}" && -n "${MAX_STEPS}" ]]; then
        if [[ -n "${PROGRESS_PCT}" ]]; then
            echo "STEP_PROGRESS=${CURRENT_STEP}/${MAX_STEPS} (${PROGRESS_PCT}%)"
        else
            echo "STEP_PROGRESS=${CURRENT_STEP}/${MAX_STEPS}"
        fi
    else
        echo "STEP_PROGRESS=N/A (training not started or no step log yet)"
    fi
    [[ -n "${STEP_TIME_RAW}" ]] && echo "STEP_TIME=${STEP_TIME_RAW}"
    [[ -n "${RECENT_LOSSES}" ]] && echo "LATEST_LOSS=${RECENT_LOSSES}"
    [[ -n "${TOKEN_ACC}" ]]     && echo "TOKEN_ACC=${TOKEN_ACC}"
    [[ -n "${GRAD_NORM}" ]]     && echo "GRAD_NORM=${GRAD_NORM}"
    [[ -n "${LR_LATEST}" ]]     && echo "LR_LATEST=${LR_LATEST}"
    [[ -n "${CKPT_LIST}" ]]     && echo "SAVE_CHECKPOINTS=${CKPT_LIST}"
    echo "HEALTH=${HEALTH}"
    [[ -n "${HEALTH_NOTES}" ]]  && echo "HEALTH_NOTES=${HEALTH_NOTES}"
    [[ -n "${FINAL_S3}" ]]      && echo "FINAL_CHECKPOINT_S3=${FINAL_S3}"
    echo "OUT_BUNDLE=${OUT}"
    echo "LOG_LINES=${LOG_LINES}"
    [[ -n "${TQDM_LATEST}" ]]   && echo "TQDM_LATEST=${TQDM_LATEST}"
    echo "================================================================="
} | tee "${OUT}/summary.txt"

exit 0
