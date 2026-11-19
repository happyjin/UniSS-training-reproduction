#!/usr/bin/env bash
# =============================================================================
# training-ops / anyscale-job-ops / locate_failure.sh
# =============================================================================
# 定位 prodjob 失败的根本原因。通过 Anyscale CLI 获取日志，用 grep 分类
# 失败阶段（提交失败/集群启动失败/初始化失败/训练失败）。
# 不读取任何本地 repo 文件。
#
# 用法：
#   bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/locate_failure.sh <PRODJOB_ID>
#   bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/locate_failure.sh <PRODJOB_ID> <LOGS_FILE>
#
# 输入参数：
#   <PRODJOB_ID>  — 如 prodjob_abc123def456
#   <LOGS_FILE>   — （可选）已有日志文件路径，跳过 anyscale job logs 拉取
#
# 输出（stdout，KEY=VALUE 格式）：
#   PRODJOB_ID / JOB_STATE / FAIL_CATEGORY(A/B/C/UNKNOWN) / FAIL_STAGE /
#   FAIL_LINE / FAIL_REASON / LOG_LINES / OUT_BUNDLE / CONTEXT_FILE
#
# 退出码：0 = 诊断完成；2 = 无法获取日志。
# =============================================================================
set -euo pipefail

PRODJOB_ID="${1:?用法: $0 <PRODJOB_ID> [LOGS_FILE]}"
LOGS_FILE="${2:-}"

TS="$(date +%s)"
OUT="/tmp/anyscale-locate-${TS}"
mkdir -p "${OUT}"

echo "[locate] prodjob=${PRODJOB_ID}"
echo "[locate] out=${OUT}"
echo ""

LOG="${OUT}/job_logs.txt"

if [[ -n "${LOGS_FILE}" && -f "${LOGS_FILE}" ]]; then
    echo "[locate] using provided logs file: ${LOGS_FILE}"
    cp "${LOGS_FILE}" "${LOG}"
else
    echo "[locate] pulling logs from Anyscale..."
    anyscale job logs --id "${PRODJOB_ID}" > "${LOG}" 2>&1 || {
        echo "[locate] FATAL: cannot fetch logs for ${PRODJOB_ID}" >&2
        echo "LOCATE_OK=no"
        echo "FAIL_REASON=logs-fetch-failed"
        echo "OUT_BUNDLE=${OUT}"
        exit 2
    }
fi

LOG_LINES=$(wc -l < "${LOG}")
echo "[locate] log has ${LOG_LINES} lines"
echo ""

STATUS_FILE="${OUT}/job_status.txt"
anyscale job status --id "${PRODJOB_ID}" > "${STATUS_FILE}" 2>&1 || true
JOB_STATE=$(grep -oiE '\b(STARTING|RUNNING|SUCCEEDED|FAILED|TERMINATED|OUT_OF_RETRIES|QUEUED|UNKNOWN)\b' "${STATUS_FILE}" | head -1 || echo "UNKNOWN")

# =============================================================================
# (2) 阶段检查
# =============================================================================
echo "================================================================="
echo "  STAGE PROGRESS (which entrypoint command died)"
echo "================================================================="

stage_check() {
    local label="$1" pattern="$2"
    local hits
    hits=$(grep -cE "${pattern}" "${LOG}" 2>/dev/null || true)
    if [[ "${hits}" -gt 0 ]]; then
        local first
        first=$(grep -nE "${pattern}" "${LOG}" | head -1)
        printf "  %-35s ✅ (%d hits, first: line %s)\n" "${label}" "${hits}" "${first%%:*}"
    else
        printf "  %-35s ❌ (no hits)\n" "${label}"
    fi
}

stage_check "head_setup started"      '\[head-setup\]|head_setup\.sh'
stage_check "head_setup phase 6"      'patch_rope_scaling|6/6|\(6/'
stage_check "head_setup all-done"     'all phases complete|\.head_setup_done|head_setup_done'
stage_check "dispatch_setup started"  '\[dispatch\][^a-z]|dispatch_setup'
stage_check "worker_setup all-done"   'all worker_setup done|worker_setup\.sh.*OK|\.worker_setup_done'
stage_check "ray_sanity alive nodes"  '\[sanity\] alive nodes:'
stage_check "ray_sanity creating actors" '\[sanity\] creating .* GPUactors'
stage_check "ray_sanity PASS"         '\[sanity\] PASS'
stage_check "swift sft started"       'swift sft|run_train|Trainer.train|TRAINING'
stage_check "NCCL EFA selected"       'Selected provider is efa|fabric is efa-direct'
stage_check "first step / loss"       '("loss":|loss=[0-9]|step.*loss|epoch.*loss)'

echo ""

# =============================================================================
# (3) 分类诊断
# =============================================================================
echo "================================================================="
echo "  CATEGORY A — 路径 / 配置 / REPO_ROOT 相关"
echo "================================================================="

CAT_A_NOTES=""
CAT_A_HIT="no"

A2=$(grep -nE 'FileNotFoundError|No such file|cannot find|cannot stat|File does not exist' "${LOG}" | head -10 || true)
if [[ -n "${A2}" ]]; then
    CAT_A_HIT="yes"
    CAT_A_NOTES+="FileNotFoundError; "
    echo "--- FileNotFoundError / No such file ---"
    echo "${A2}"
    echo ""
fi

A3=$(grep -nE '//.*/|REPO_ROOT.*//' "${LOG}" | head -10 || true)
if [[ -n "${A3}" ]]; then
    CAT_A_HIT="yes"
    CAT_A_NOTES+="double-path; "
    echo "--- 路径拼接异常（双斜杠或重复目录）---"
    echo "${A3}"
    echo ""
fi

A4=$(grep -nE 'ModuleNotFoundError|ImportError' "${LOG}" | head -10 || true)
if [[ -n "${A4}" ]]; then
    CAT_A_HIT="yes"
    CAT_A_NOTES+="ModuleNotFoundError; "
    echo "--- ModuleNotFoundError / ImportError ---"
    echo "${A4}"
    echo ""
fi

if [[ "${CAT_A_HIT}" == "no" ]]; then
    echo "  (clean — no Category A issues found)"
    echo ""
fi

echo "================================================================="
echo "  CATEGORY B — 集群/K8s 相关"
echo "================================================================="

CAT_B_NOTES=""
CAT_B_HIT="no"

B1=$(grep -nE 'FailedScheduling|Insufficient|did not match|0/[0-9]+ nodes are available' "${LOG}" | head -10 || true)
if [[ -n "${B1}" ]]; then
    CAT_B_HIT="yes"
    CAT_B_NOTES+="FailedScheduling; "
    echo "--- FailedScheduling / Insufficient ---"
    echo "${B1}"
    echo ""
fi

B2=$(grep -nE 'ActorUnschedulableError|RayTaskError|Failed to startup worker|placement group' "${LOG}" | head -10 || true)
if [[ -n "${B2}" ]]; then
    CAT_B_HIT="yes"
    CAT_B_NOTES+="ActorUnschedulable; "
    echo "--- ray actor 起不来 ---"
    echo "${B2}"
    echo ""
fi

B3=$(grep -nE 'NCCL.*Socket|fi_getinfo.*-61|No eligible providers|libcudart.so.*cannot|EFA.*not.*available' "${LOG}" | head -10 || true)
if [[ -n "${B3}" ]]; then
    CAT_B_HIT="yes"
    CAT_B_NOTES+="NCCL-EFA-issue; "
    echo "--- NCCL fallback / EFA missing ---"
    echo "${B3}"
    echo ""
fi

B4=$(grep -nE 'Pending|SCHEDULING|unschedulable' "${LOG}" | head -10 || true)
if [[ -n "${B4}" ]]; then
    CAT_B_HIT="yes"
    CAT_B_NOTES+="Pending/Scheduling; "
    echo "--- Pending / Scheduling ---"
    echo "${B4}"
    echo ""
fi

if [[ "${CAT_B_HIT}" == "no" ]]; then
    echo "  (clean — no Category B issues found)"
    echo ""
fi

echo "================================================================="
echo "  CATEGORY C — 训练相关"
echo "================================================================="

CAT_C_NOTES=""
CAT_C_HIT="no"

C1=$(grep -nE 'OOMKilled|out of memory|CUDA out|Xid.*[0-9]|model.*load.*fail|FSDPModule|deepspeed.*nvcc' "${LOG}" | head -10 || true)
if [[ -n "${C1}" ]]; then
    CAT_C_HIT="yes"
    CAT_C_NOTES+="OOM/Xid; "
    echo "--- OOM / Xid / model load ---"
    echo "${C1}"
    echo ""
fi

C2=$(grep -nE 'TORCHELASTIC_USE_AGENT_STORE|Socket::connect|TCPStore|c10d::detail|timeout.*rank' "${LOG}" | head -10 || true)
if [[ -n "${C2}" ]]; then
    CAT_C_HIT="yes"
    CAT_C_NOTES+="TCPStore-hang; "
    echo "--- TCPStore hang / torchelastic ---"
    echo "${C2}"
    echo ""
fi

C3=$(grep -nE 'Traceback|Exception:|Error:|raises' "${LOG}" | head -10 || true)
if [[ -n "${C3}" ]]; then
    CAT_C_HIT="yes"
    CAT_C_NOTES+="Python-exception; "
    echo "--- Python Traceback / Exception ---"
    echo "${C3}"
    echo ""
fi

if [[ "${CAT_C_HIT}" == "no" ]]; then
    echo "  (clean — no Category C issues found)"
    echo ""
fi

# =============================================================================
# (4) 确定 FAIL_CATEGORY 和 FAIL_STAGE
# =============================================================================
FAIL_CATEGORY="UNKNOWN"
FAIL_STAGE="unknown"
FAIL_REASON=""

if [[ "${CAT_A_HIT}" == "yes" ]]; then
    FAIL_CATEGORY="A"
    FAIL_REASON="${CAT_A_NOTES%;}"
elif [[ "${CAT_B_HIT}" == "yes" ]]; then
    FAIL_CATEGORY="B"
    FAIL_REASON="${CAT_B_NOTES%;}"
elif [[ "${CAT_C_HIT}" == "yes" ]]; then
    FAIL_CATEGORY="C"
    FAIL_REASON="${CAT_C_NOTES%;}"
fi

if ! grep -qE 'head_setup.*✅' <<< "$(grep 'head_setup' <<< "$(cat "${LOG}" || true)")"; then
    FAIL_STAGE="head_setup"
elif ! grep -q 'ray_sanity PASS' "${LOG}" 2>/dev/null; then
    FAIL_STAGE="ray_sanity"
elif ! grep -qE 'swift sft|Trainer.train' "${LOG}" 2>/dev/null; then
    FAIL_STAGE="dispatch_train"
else
    FAIL_STAGE="training"
fi

# =============================================================================
# (5) 第一个 Traceback / Error 的上下文（60 行）
# =============================================================================
echo "================================================================="
echo "  FIRST TRACEBACK / ERROR (with 60-line context)"
echo "================================================================="
FIRST_ERR_LINE=$(grep -nE 'Traceback|^ERROR |No such file|cannot open file|exit code [1-9]|Exception:' "${LOG}" | head -1 | cut -d: -f1 || true)

CONTEXT_OUT="${OUT}/first_error_context.txt"
if [[ -n "${FIRST_ERR_LINE:-}" ]]; then
    START=$(( FIRST_ERR_LINE > 10 ? FIRST_ERR_LINE - 10 : 1 ))
    END=$(( FIRST_ERR_LINE + 50 ))
    echo "[locate] first error at line ${FIRST_ERR_LINE}, showing lines ${START}..${END}"
    echo "---" | tee -a "${CONTEXT_OUT}"
    sed -n "${START},${END}p" "${LOG}" | tee -a "${CONTEXT_OUT}"
    FAIL_LINE="${FIRST_ERR_LINE}"
else
    echo "  (no Traceback / ERROR / Exception found)" | tee -a "${CONTEXT_OUT}"
    FAIL_LINE=""
fi
echo ""

# =============================================================================
# (6) 末尾 60 行
# =============================================================================
echo "================================================================="
echo "  TAIL (last 60 lines — how the job actually died)"
echo "================================================================="
tail -60 "${LOG}" | tee "${OUT}/tail_60.txt"
echo ""

# =============================================================================
# (7) 输出结构化结果
# =============================================================================
echo "================================================================="
echo "  STRUCTURED OUTPUT"
echo "================================================================="
echo "PRODJOB_ID=${PRODJOB_ID}"
echo "JOB_STATE=${JOB_STATE}"
echo "FAIL_CATEGORY=${FAIL_CATEGORY}"
echo "FAIL_STAGE=${FAIL_STAGE}"
echo "FAIL_LINE=${FAIL_LINE:-}"
echo "FAIL_REASON=${FAIL_REASON:-unknown}"
echo "LOG_LINES=${LOG_LINES}"
echo "OUT_BUNDLE=${OUT}"
echo "CONTEXT_FILE=${CONTEXT_OUT}"
echo "================================================================="

{
    echo "PRODJOB_ID=${PRODJOB_ID}"
    echo "JOB_STATE=${JOB_STATE}"
    echo "FAIL_CATEGORY=${FAIL_CATEGORY}"
    echo "FAIL_STAGE=${FAIL_STAGE}"
    echo "FAIL_LINE=${FAIL_LINE:-}"
    echo "FAIL_REASON=${FAIL_REASON:-unknown}"
    echo "LOG_LINES=${LOG_LINES}"
    echo "OUT_BUNDLE=${OUT}"
} > "${OUT}/summary.txt"

exit 0
