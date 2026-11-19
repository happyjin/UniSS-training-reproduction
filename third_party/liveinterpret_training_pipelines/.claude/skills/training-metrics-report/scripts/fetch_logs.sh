#!/usr/bin/env bash
# =============================================================================
# training-metrics-report / fetch_logs.sh
# =============================================================================
# 拉取一个 ktrain 训练任务的原始日志 + mfu_log.jsonl 到本地，供
# parse_metrics.py 解析。全程只读（kubectl logs / kubectl exec cat），
# 不修改远程任何状态、不停止任务。
#
# 用法：
#   bash fetch_logs.sh <JOB_NAME> <NAMESPACE> <POD_NAME> [<OUTPUT_DIR> [<EXEC_POD_NAME>]]
#     JOB_NAME       任务名（仅用于本地输出目录命名）
#     NAMESPACE      k8s 命名空间
#     POD_NAME       拉取训练日志的 pod：
#                       pytorch 直连（单机/多机）-> "<JOB_NAME>-master-0"
#                       ray（KubeRay）           -> 任务提交器 pod（不带 -head-/-worker-
#                                                    后缀，用 kubectl get pods | grep 定位）
#     OUTPUT_DIR     可选，任务 yaml spec.env.OUTPUT_DIR 的值。传了才会尝试拉
#                    mfu_log.jsonl（路径 = ${OUTPUT_DIR}/mfu_log.jsonl）
#     EXEC_POD_NAME  可选，拉 mfu_log.jsonl 时 exec 进哪个还在运行的训练容器；
#                    不传则默认复用 POD_NAME（仅当它是普通训练容器而非 ray
#                    提交器时才有效；ray 模式必须显式传一个 -worker- pod）
#
# 输出（/tmp/training-metrics-report/<JOB_NAME>/）：
#   raw_log.txt      完整 kubectl logs 输出
#   mfu_log.jsonl    拉取到的 mfu_log.jsonl（若指定 OUTPUT_DIR 且拉取成功）
#
# stdout（KEY=VALUE）：FETCH_OK / RAW_LOG_LINES / MFU_LOG_LINES / OUT_DIR
# =============================================================================
set -uo pipefail

JOB_NAME="${1:?用法: $0 <JOB_NAME> <NAMESPACE> <POD_NAME> [<OUTPUT_DIR> [<EXEC_POD_NAME>]]}"
NS="${2:?命名空间必填}"
POD_NAME="${3:?日志 pod 名必填（pytorch: <JOB_NAME>-master-0；ray: 提交器 pod}"
OUTPUT_DIR_REMOTE="${4:-}"
EXEC_POD_NAME="${5:-${POD_NAME}}"

OUT="/tmp/training-metrics-report/${JOB_NAME}"
mkdir -p "${OUT}"

echo "[fetch] job=${JOB_NAME} ns=${NS} pod=${POD_NAME}"
echo "[fetch] out=${OUT}"

echo "=== kubectl logs ${POD_NAME} ==="
kubectl logs -n "${NS}" "${POD_NAME}" > "${OUT}/raw_log.txt" 2>&1
RAW_LOG_LINES=$(wc -l < "${OUT}/raw_log.txt" 2>/dev/null || echo 0)
echo "  ${RAW_LOG_LINES} lines -> ${OUT}/raw_log.txt"

MFU_LOG_LINES=0
if [[ -n "${OUTPUT_DIR_REMOTE}" ]]; then
    MFU_PATH="${OUTPUT_DIR_REMOTE%/}/mfu_log.jsonl"
    echo "=== kubectl exec ${EXEC_POD_NAME} -- cat ${MFU_PATH} ==="
    if kubectl exec -n "${NS}" "${EXEC_POD_NAME}" -- cat "${MFU_PATH}" > "${OUT}/mfu_log.jsonl" 2>/dev/null; then
        MFU_LOG_LINES=$(wc -l < "${OUT}/mfu_log.jsonl" 2>/dev/null || echo 0)
        echo "  ${MFU_LOG_LINES} lines -> ${OUT}/mfu_log.jsonl"
    else
        echo "  [fetch] WARN: mfu_log.jsonl 拉取失败（路径不存在，或 EXEC_POD_NAME 已不在运行）" >&2
        rm -f "${OUT}/mfu_log.jsonl"
    fi
else
    echo "[fetch] 未传 OUTPUT_DIR，跳过 mfu_log.jsonl 拉取"
fi

FETCH_OK="yes"
[[ "${RAW_LOG_LINES}" -lt 3 ]] && FETCH_OK="no"

echo "================================================================="
echo "FETCH_OK=${FETCH_OK}"
echo "RAW_LOG_LINES=${RAW_LOG_LINES}"
echo "MFU_LOG_LINES=${MFU_LOG_LINES}"
echo "OUT_DIR=${OUT}"
echo "================================================================="

[[ "${FETCH_OK}" == "no" ]] && exit 2
exit 0
