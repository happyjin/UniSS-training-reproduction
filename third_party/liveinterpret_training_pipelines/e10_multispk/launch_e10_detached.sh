#!/usr/bin/env bash
# E10 formal training, detached (setsid + nohup; survives the session). Refuses if the run's output exists.
# Do NOT test-run this script: check it with `bash -n` or `DRY_RUN=1` (train_e10.sh then only prints the command).
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash e10_multispk/launch_e10_detached.sh
#   DRY_RUN=1 bash e10_multispk/launch_e10_detached.sh
set -euo pipefail
E10_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${RUN_LABEL:=e10_thinker_v1}"
: "${OUTPUT_ROOT:=${E10_DIR}/output}"
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${MASTER_PORT:=29641}"
NPROC_PER_NODE=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
export RUN_LABEL OUTPUT_ROOT CUDA_VISIBLE_DEVICES MASTER_PORT NPROC_PER_NODE
LOG="${OUTPUT_ROOT}/${RUN_LABEL}.log"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    bash "${E10_DIR}/train_e10.sh"
    echo "[launch] DRY_RUN: would log to ${LOG}"
    exit 0
fi
if [[ -e "${OUTPUT_ROOT}/${RUN_LABEL}" || -e "${LOG}" ]]; then
    echo "[launch] ${OUTPUT_ROOT}/${RUN_LABEL} or its log exists; refusing to overwrite" >&2
    exit 1
fi
mkdir -p "${OUTPUT_ROOT}"
# close inherited lock fds (flock fd inheritance stalled other jobs before)
setsid nohup bash "${E10_DIR}/train_e10.sh" > "${LOG}" 2>&1 < /dev/null 9>&- &
PID=$!
disown
echo "${PID}" > "${OUTPUT_ROOT}/${RUN_LABEL}.pid"
echo "[launch] E10 training pid ${PID} (session leader) on GPUs ${CUDA_VISIBLE_DEVICES}; log ${LOG}"
