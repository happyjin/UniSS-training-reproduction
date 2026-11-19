#!/usr/bin/env bash
# =============================================================================
# training-ops / anyscale-job-ops / submit_helper.sh
# =============================================================================
# 预校验 Job YAML 配置合法性，可选执行提交。
#
# 用法：
#   bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/submit_helper.sh <JOB_YAML> [options]
#   DRY_RUN=1 bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/submit_helper.sh <JOB_YAML>
#
# 选项（在 JOB_YAML 之后）：
#   --name <name>   覆盖自动生成的任务名
#   --               之后的参数原样传给 anyscale job submit
#
# 输出：stdout KEY=VALUE 格式
# 退出码：0=成功  2=预检失败  3=提交失败
# =============================================================================
set -euo pipefail

# ---- 参数解析 -------------------------------------------------------------
JOB_YAML=""
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)      JOB_NAME_OVERRIDE="$2"; shift 2 ;;
        --name=*)    JOB_NAME_OVERRIDE="${1#*=}"; shift ;;
        --)          shift; PASSTHROUGH+=("$@"); break ;;
        -*)          echo "[submit-helper] ERROR: unknown flag: $1" >&2; exit 2 ;;
        *)
            if [[ -z "${JOB_YAML}" ]]; then
                JOB_YAML="$1"
            else
                PASSTHROUGH+=("$1")
            fi
            shift
            ;;
    esac
done

if [[ -z "${JOB_YAML}" ]]; then
    echo "[submit-helper] ERROR: JOB_YAML is required" >&2
    echo "Usage: $0 <JOB_YAML> [--name <name>] [-- ...]" >&2
    exit 2
fi

if [[ ! -f "${JOB_YAML}" ]]; then
    echo "[submit-helper] FATAL: ${JOB_YAML} not found (cwd: $(pwd))" >&2
    exit 2
fi

OUT="/tmp/anyscale-submit/$(basename "${JOB_YAML}" .yaml)-$(date +%s)"
mkdir -p "${OUT}"

echo "[submit-helper] job_yaml=${JOB_YAML}"
echo "[submit-helper] out=${OUT}"
[[ ${#PASSTHROUGH[@]} -gt 0 ]] && echo "[submit-helper] passthrough: ${PASSTHROUGH[*]}"
echo ""

# ---- (1) AWS 凭证检查 -------------------------------------------------------
echo "=== (1/3) AWS credentials ==="
AWS_ID_JSON="${OUT}/aws_identity.json"
if aws sts get-caller-identity --output json > "${AWS_ID_JSON}" 2>"${OUT}/aws_err.txt"; then
    AWS_ACCOUNT_ID=$(grep -oE '"Account"[[:space:]]*:[[:space:]]*"[^"]+' "${AWS_ID_JSON}" | head -1 | grep -oE '"[^"]+"' | tr -d '"' || echo "")
    AWS_ARN=$(grep -oE '"Arn"[[:space:]]*:[[:space:]]*"[^"]+' "${AWS_ID_JSON}" | head -1 | grep -oE '"[^"]+"' | tr -d '"' || echo "")
    echo "  AWS account = ${AWS_ACCOUNT_ID:-<unknown>}"
else
    echo "[submit-helper] FATAL: aws sts get-caller-identity failed:" >&2
    cat "${OUT}/aws_err.txt" >&2
    echo "SUBMIT_OK=no"
    echo "FAIL_REASON=aws-credentials-missing"
    exit 2
fi

echo ""

# ---- (2) Anyscale cloud 可见性 ---------------------------------------------
echo "=== (2/3) Anyscale cloud list ==="
if anyscale cloud list --output json > "${OUT}/cloud_list.json" 2>"${OUT}/anyscale_err.txt"; then
    echo "  cloud list fetched (json) OK"
else
    if ! anyscale cloud list > "${OUT}/cloud_list.txt" 2>"${OUT}/anyscale_err.txt"; then
        echo "[submit-helper] FATAL: anyscale cloud list failed:" >&2
        cat "${OUT}/anyscale_err.txt" >&2
        echo "SUBMIT_OK=no"
        echo "FAIL_REASON=anyscale-not-logged-in"
        exit 2
    fi
    echo "  cloud list fetched (table) OK"
fi
echo ""

# ---- (3) 解析 + 校验 job.yaml ----------------------------------------------
echo "=== (3/3) job.yaml pins ==="
IMAGE_URI=$(grep -E '^image_uri:' "${JOB_YAML}" | head -1 | awk '{print $2}')
COMPUTE_CFG=$(grep -E '^compute_config:' "${JOB_YAML}" | head -1 | awk '{print $2}')
echo "  image_uri      = ${IMAGE_URI}"
echo "  compute_config = ${COMPUTE_CFG}"

if [[ -z "${IMAGE_URI}" || -z "${COMPUTE_CFG}" ]]; then
    echo "[submit-helper] FATAL: cannot parse image_uri/compute_config from ${JOB_YAML}" >&2
    echo "SUBMIT_OK=no"
    echo "FAIL_REASON=yaml-parse-failed"
    exit 2
fi

if ! anyscale compute-config get -n "${COMPUTE_CFG}" >"${OUT}/compute_config.txt" 2>&1; then
    echo "[submit-helper] FATAL: compute_config ${COMPUTE_CFG} not found:" >&2
    cat "${OUT}/compute_config.txt" >&2
    echo "SUBMIT_OK=no"
    echo "FAIL_REASON=compute-config-not-found"
    exit 2
fi
echo "  compute config exists OK"
echo ""

# ---- (4) 生成任务名 -------------------------------------------------------
# 三段式：<模型名>-<user>-<4位随机字符>
rand="$(LC_ALL=C tr -dc 'a-z0-9' </dev/urandom | head -c 4 || true)"
user="$(git config user.name 2>/dev/null || whoami)"
user="$(printf '%s' "$user" | tr '[:upper:]_ ' '[:lower:]--' | sed 's/[^a-z0-9-]//g')"

# 模型名：优先 MODEL_NAME 环境变量，否则从 YAML 文件名推断
model_name="${MODEL_NAME:-}"
if [[ -z "${model_name}" ]]; then
    yaml_base="$(basename "${JOB_YAML}" .yaml)"
    if [[ "${yaml_base}" == *etl* ]]; then
        model_name="etl"
    else
        model_name="train"
    fi
fi

default_name="${model_name}-${user}-${rand}"
JOB_NAME="${JOB_NAME_OVERRIDE:-${default_name}}"
echo "[submit-helper] job name = ${JOB_NAME}"
echo ""

# ---- (5) DRY_RUN：只预检，不提交 -------------------------------------------
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo ""
    echo "=== DRY_RUN=1 -> skipping anyscale job submit ==="
    echo ""
    echo "================================================================="
    echo "  STRUCTURED OUTPUT (DRY-RUN, no job submitted)"
    echo "================================================================="
    echo "DRY_RUN=yes"
    echo "PRECHECK_OK=yes"
    echo "JOB_NAME=${JOB_NAME}"
    echo "AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-}"
    echo "AWS_USER_ARN=${AWS_ARN:-}"
    echo "IMAGE_URI=${IMAGE_URI}"
    echo "COMPUTE_CONFIG=${COMPUTE_CFG}"
    echo "OUT_BUNDLE=${OUT}"
    echo "================================================================="
    exit 0
fi

# ---- (6) 正式提交 -----------------------------------------------------------
echo "=== submitting ==="
echo "[submit-helper] anyscale job submit -f ${JOB_YAML} --name ${JOB_NAME}"
echo ""

SUBMIT_LOG="${OUT}/submit.log"
SUBMIT_RC=0

submit_cmd=(anyscale job submit -f "${JOB_YAML}" --name "${JOB_NAME}")
[[ ${#PASSTHROUGH[@]} -gt 0 ]] && submit_cmd+=("${PASSTHROUGH[@]}")

# 打印命令（调试用）
{
    echo "anyscale job submit \\"
    echo "    -f ${JOB_YAML} \\"
    echo "    --name ${JOB_NAME} \\"
    for pf in "${PASSTHROUGH[@]}"; do echo "    ${pf} \\"; done
    echo ""
} | tee "${OUT}/submit_cmd.txt" || true

# 执行提交
"${submit_cmd[@]}" 2>&1 | tee "${SUBMIT_LOG}" || SUBMIT_RC=$?

if [[ "${SUBMIT_RC}" -ne 0 ]]; then
    echo "[submit-helper] FAIL: anyscale job submit exited rc=${SUBMIT_RC}" >&2
    echo "SUBMIT_OK=no"
    echo "FAIL_REASON=anyscale-job-submit-failed rc=${SUBMIT_RC}"
    exit 3
fi

# ---- (7) 抓取 prodjob_id ----------------------------------------------------
PRODJOB_ID=$(grep -oE 'prodjob_[a-z0-9]{20,}' "${SUBMIT_LOG}" | head -1)
if [[ -z "${PRODJOB_ID}" ]]; then
    PRODJOB_ID=$(anyscale job list --output json 2>/dev/null \
        | grep -oE 'prodjob_[a-z0-9]{20,}' | head -1 || true)
fi
if [[ -z "${PRODJOB_ID}" ]]; then
    echo "[submit-helper] FAIL: cannot find prodjob_id in submit output" >&2
    echo "SUBMIT_OK=no"
    echo "FAIL_REASON=prodjob-id-not-captured"
    exit 3
fi

JOB_URL="https://console.anyscale.com/jobs/${PRODJOB_ID}"

# ---- (8) 一次性 status ------------------------------------------------------
echo ""
echo "=== one-shot status ==="
STATUS_TXT=$(anyscale job status --id "${PRODJOB_ID}" 2>&1 || true)
echo "${STATUS_TXT}" | tee "${OUT}/status.txt"
JOB_STATE=$(echo "${STATUS_TXT}" | grep -oiE '\b(STARTING|RUNNING|SUCCEEDED|FAILED|TERMINATED|OUT_OF_RETRIES)\b' | head -1 || echo "UNKNOWN")

# ---- 输出结构化字段 ---------------------------------------------------------
echo ""
echo "================================================================="
echo "  STRUCTURED OUTPUT"
echo "================================================================="
echo "PRODJOB_ID=${PRODJOB_ID}"
echo "JOB_URL=${JOB_URL}"
echo "JOB_NAME=${JOB_NAME}"
echo "SUBMIT_OK=yes"
echo "STATUS=${JOB_STATE}"
echo "IMAGE_URI=${IMAGE_URI}"
echo "COMPUTE_CONFIG=${COMPUTE_CFG}"
echo "OUT_BUNDLE=${OUT}"
echo "================================================================="
