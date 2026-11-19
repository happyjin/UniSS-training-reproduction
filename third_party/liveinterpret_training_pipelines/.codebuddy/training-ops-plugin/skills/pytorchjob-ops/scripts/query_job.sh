#!/usr/bin/env bash
# =============================================================================
# training-ops / pytorchjob-ops / query_job.sh
#
# 通用 PyTorchJob 训练进度查询脚本（纯 kubectl，不依赖特定集群/项目）。
# 集群、命名空间、job 名等均为参数，不在此硬编码。
# =============================================================================
# 用法：
#   bash query_job.sh <JOB_NAME> [NAMESPACE]
#     JOB_NAME    必填，PyTorchJob 名称（如项目 pytorchjob.yaml 中的 metadata.name）
#     NAMESPACE   可选，k8s 命名空间；也可用环境变量 K8S_NS 传入
#
# 输出（写入 /tmp/pytorchjob-progress/<JOB_NAME>/）：pods.txt / logs.txt / summary.txt
# 输出（stdout，KEY=VALUE）：
#   JOB_STATE / POD_STATUS / MASTER_NODE / RESTARTS / STEP_PROGRESS / LATEST_LOSS /
#   TOKEN_ACC / HEALTH / HEALTH_NOTES / LOG_LINES / OUT_BUNDLE
# =============================================================================
set -euo pipefail

JOB_NAME="${1:-}"
NS="${2:-${K8S_NS:-}}"

if [[ -z "${JOB_NAME}" || -z "${NS}" ]]; then
  echo "usage: query_job.sh <JOB_NAME> [NAMESPACE]" >&2
  echo "  (NAMESPACE 也可通过环境变量 K8S_NS 传入)" >&2
  exit 2
fi

OUT="/tmp/pytorchjob-progress/${JOB_NAME}"
mkdir -p "${OUT}"

echo "[query] job=${JOB_NAME} ns=${NS}"
echo "[query] out=${OUT}"
echo ""

echo "=== pod status ==="
kubectl -n "${NS}" get pods -l "training.kubeflow.org/job-name=${JOB_NAME}" -o wide > "${OUT}/pods.txt" 2>&1 || true
cat "${OUT}/pods.txt"
echo ""

POD_STATE=$(awk 'NR>1{print $3; exit}' "${OUT}/pods.txt" 2>/dev/null || echo "Unknown")
MASTER_NODE=$(grep -E 'master-0' "${OUT}/pods.txt" 2>/dev/null | awk '{print $7}' || true)
RESTARTS=$(grep -E 'master-0' "${OUT}/pods.txt" 2>/dev/null | awk '{print $4}' || true)

JOB_STATE="Running"
if [[ "${POD_STATE}" == "Pending" ]]; then JOB_STATE="Pending"; fi
if [[ "${POD_STATE}" == "Error" || "${POD_STATE}" == "OOMKilled" ]]; then JOB_STATE="Failed"; fi

echo "=== master-0 logs ==="
MASTER_POD=$(kubectl -n "${NS}" get pods -l "training.kubeflow.org/job-name=${JOB_NAME}" -o name 2>/dev/null | grep master-0 | head -1 || echo "")
if [[ -n "${MASTER_POD}" ]]; then
    kubectl -n "${NS}" logs "${MASTER_POD}" -c pytorch > "${OUT}/logs.txt" 2>&1 || true
else
    kubectl -n "${NS}" logs -l "training.kubeflow.org/job-name=${JOB_NAME}" -c pytorch --tail=2000 > "${OUT}/logs.txt" 2>&1 || true
fi
LOG_LINES=$(wc -l < "${OUT}/logs.txt")
echo "    -> ${OUT}/logs.txt (${LOG_LINES} lines)"
echo ""

LOG="${OUT}/logs.txt"

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

RECENT_LOSSES=""
if grep -q "'loss'" "${LOG}" 2>/dev/null; then
    RECENT_LOSSES=$(grep -E "'loss'" "${LOG}" | grep "'global_step/max_steps'" | tail -3 | \
        grep -oE "'loss': [0-9.]+.*'global_step/max_steps': '[0-9]+" | \
        sed "s/.*'loss': \([0-9.]*\).*global_step\/max_steps': '\([0-9]*\)/\2:\1/" | \
        awk -F: '{printf "step %s: %.3f | ", $1, $2}' | sed 's/| $//' || true)
fi
TOKEN_ACC=$(grep -oE "'token_acc': [0-9.]+" "${LOG}" | tail -1 | grep -oE '[0-9.]+' || true)

HEALTH="OK"
HEALTH_NOTES=""
if grep -qE 'RuntimeError|CUDA out of memory|NCCL.*ERROR|Xid [0-9]+' "${LOG}" 2>/dev/null; then
    HEALTH="FAIL"
    HEALTH_NOTES=$(grep -nE 'RuntimeError|CUDA out of memory|NCCL.*ERROR|Xid [0-9]+' "${LOG}" 2>/dev/null | head -3 | cut -d: -f2- | tr '\n' ';' | sed 's/;$//' || true)
fi
if [[ -n "${RESTARTS}" && "${RESTARTS}" != "0" && -n "${RESTARTS}" ]]; then
    HEALTH="WARN"
    HEALTH_NOTES="pod restarts=${RESTARTS} (possible OOM cascade)"
fi
if [[ "${JOB_STATE}" == "Failed" ]]; then
    HEALTH="FAIL"
    HEALTH_NOTES="${HEALTH_NOTES} pod_state=${POD_STATE}"
fi

{
    echo "================================================================="
    echo "  STRUCTURED OUTPUT"
    echo "================================================================="
    echo "JOB_STATE=${JOB_STATE}"
    echo "POD_STATUS=${POD_STATE}"
    echo "MASTER_NODE=${MASTER_NODE:-unknown}"
    echo "RESTARTS=${RESTARTS:-0}"
    if [[ -n "${CURRENT_STEP}" && -n "${MAX_STEPS}" ]]; then
        if [[ -n "${PROGRESS_PCT}" ]]; then
            echo "STEP_PROGRESS=${CURRENT_STEP}/${MAX_STEPS} (${PROGRESS_PCT}%)"
        else
            echo "STEP_PROGRESS=${CURRENT_STEP}/${MAX_STEPS}"
        fi
    else
        echo "STEP_PROGRESS=N/A"
    fi
    [[ -n "${RECENT_LOSSES}" ]] && echo "LATEST_LOSS=${RECENT_LOSSES}"
    [[ -n "${TOKEN_ACC}" ]]     && echo "TOKEN_ACC=${TOKEN_ACC}"
    echo "HEALTH=${HEALTH}"
    [[ -n "${HEALTH_NOTES}" ]]  && echo "HEALTH_NOTES=${HEALTH_NOTES}"
    echo "OUT_BUNDLE=${OUT}"
    echo "LOG_LINES=${LOG_LINES}"
    echo "================================================================="
} | tee "${OUT}/summary.txt"

exit 0
