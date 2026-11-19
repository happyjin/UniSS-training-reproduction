#!/usr/bin/env bash
# =============================================================================
# training-ops / anyscale-job-ops / baseline_check.sh
# =============================================================================
# 远程基线检查：通过 Anyscale CLI 获取任务状态、compute config、image 等信息，
# 不读取本地 repo 文件。所有信息均通过 Anyscale / AWS CLI 远程获取。
#
# 用法：
#   bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/baseline_check.sh <PRODJOB_ID>
#   bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/baseline_check.sh <JOB_YAML_NAME>
#
# 输入参数（二选一）：
#   <PRODJOB_ID>      — 如 prodjob_abc123def456（直接查该任务）
#   <JOB_YAML_NAME>   — 如 train/job.yaml（anyscale job list | grep 取其最近提交）
#
# 输出（stdout，全 KEY=VALUE，单行）：
#   BASELINE_OK / PRODJOB_ID / JOB_STATE / COMPUTE_CONFIG / COMPUTE_CONFIG_EXISTS /
#   IMAGE_URI / IMAGE_EXISTS / AWS_CREDENTIALS_OK / AWS_ACCOUNT_ID / AWS_USER_ARN /
#   CLOUD_VISIBLE / OUT_BUNDLE
#
# 退出码：0 = 检查完成（结果见 BASELINE_OK）；2 = 参数错误或无法获取任务信息。
# =============================================================================
set -euo pipefail

INPUT="${1:?用法: $0 <PRODJOB_ID>|<JOB_YAML_NAME>}"

TS="$(date +%s)"
OUT="/tmp/anyscale-baseline-${TS}"
mkdir -p "${OUT}"

echo "[baseline] input=${INPUT}"
echo "[baseline] out=${OUT}"
echo ""

PRODJOB_ID=""
JOB_STATUS_FILE="${OUT}/job_status.txt"

if [[ "${INPUT}" =~ ^prodjob_ ]]; then
    PRODJOB_ID="${INPUT}"
    echo "=== anyscale job status --id ${PRODJOB_ID} ==="
    anyscale job status --id "${PRODJOB_ID}" > "${JOB_STATUS_FILE}" 2>&1 || {
        echo "[baseline] WARN: anyscale job status failed, saving output" >&2
    }
else
    echo "=== anyscale job list | grep ${INPUT} ==="
    anyscale job list > "${OUT}/job_list.txt" 2>&1 || {
        echo "BASELINE_OK=no"
        echo "FAIL_REASON=anyscale-job-list-failed"
        exit 2
    }
    PRODJOB_ID=$(awk -v name="${INPUT}" '$0 ~ name {print $1; exit}' "${OUT}/job_list.txt" || true)
    if [[ -z "${PRODJOB_ID}" ]]; then
        echo "BASELINE_OK=no"
        echo "FAIL_REASON=no-job-found-for-name:${INPUT}"
        exit 2
    fi
    echo "  found prodjob_id=${PRODJOB_ID}"
    echo ""
    echo "=== anyscale job status --id ${PRODJOB_ID} ==="
    anyscale job status --id "${PRODJOB_ID}" > "${JOB_STATUS_FILE}" 2>&1 || true
fi

echo ""

JOB_STATE=$(grep -oiE '\b(STARTING|RUNNING|SUCCEEDED|FAILED|TERMINATED|OUT_OF_RETRIES|QUEUED|UNKNOWN)\b' "${JOB_STATUS_FILE}" | head -1 || echo "UNKNOWN")

COMPUTE_CONFIG=$(grep -iE 'compute.config' "${JOB_STATUS_FILE}" | head -1 | grep -oE '[a-zA-Z0-9_-]+(:[a-zA-Z0-9_-]+)?' | tail -1 || true)
if [[ -z "${COMPUTE_CONFIG}" ]]; then
    COMPUTE_CONFIG=$(grep -iE 'config' "${JOB_STATUS_FILE}" | grep -v '^#' | head -1 | awk '{for(i=1;i<=NF;i++) if($i ~ /[a-zA-Z0-9_-]+:[a-zA-Z0-9_-]+/) {print $i; exit}}' || true)
fi

IMAGE_URI=$(grep -E 'https?://[0-9]+\.dkr\.ecr\.[^ ]+|:[a-zA-Z0-9_.-]+' "${JOB_STATUS_FILE}" | head -1 | awk '{print $NF}' || true)
if [[ -z "${IMAGE_URI}" ]]; then
    IMAGE_URI=$(grep -iE 'image' "${JOB_STATUS_FILE}" | grep -v '^#' | head -1 | grep -oE '[0-9]+\.dkr\.ecr\.[^ ]+|:[a-zA-Z0-9_.-]+' | head -1 || true)
fi

echo "[baseline] extracted: JOB_STATE=${JOB_STATE}"
echo "[baseline] extracted: COMPUTE_CONFIG=${COMPUTE_CONFIG:-<empty>}"
echo "[baseline] extracted: IMAGE_URI=${IMAGE_URI:-<empty>}"
echo ""

echo "=== compute config verification ==="
COMPUTE_CONFIG_EXISTS="no"
if [[ -n "${COMPUTE_CONFIG}" ]]; then
    if anyscale compute-config get -n "${COMPUTE_CONFIG}" > "${OUT}/compute_config.txt" 2>&1; then
        COMPUTE_CONFIG_EXISTS="yes"
        echo "  compute config '${COMPUTE_CONFIG}' exists ✅"
    else
        echo "  compute config '${COMPUTE_CONFIG}' NOT FOUND ❌"
        anyscale compute-config list > "${OUT}/compute_config_list.txt" 2>&1 || true
    fi
else
    echo "  (no compute config extracted from job status, skipping verify)"
    anyscale compute-config list > "${OUT}/compute_config_list.txt" 2>&1 || true
fi
echo ""

echo "=== image verification ==="
IMAGE_EXISTS="no"
if [[ -n "${IMAGE_URI}" ]]; then
    IMAGE_NAME=$(echo "${IMAGE_URI}" | sed 's/:.*$//' | sed 's|.*/||')
    if anyscale image list > "${OUT}/image_list.txt" 2>&1; then
        if grep -q "${IMAGE_NAME}" "${OUT}/image_list.txt"; then
            IMAGE_EXISTS="yes"
            echo "  image '${IMAGE_NAME}' exists ✅"
        else
            echo "  image '${IMAGE_NAME}' NOT FOUND in anyscale image list ❌"
        fi
    else
        echo "  anyscale image list failed, trying ECR direct check..."
        if aws ecr describe-images --repository-name "${IMAGE_NAME}" --region "$(aws configure get region 2>/dev/null || echo ap-northeast-1)" >/dev/null 2>&1; then
            IMAGE_EXISTS="yes"
            echo "  image found in ECR ✅"
        fi
    fi
else
    echo "  (no image URI extracted from job status, skipping verify)"
    anyscale image list > "${OUT}/image_list.txt" 2>&1 || true
fi
echo ""

echo "=== AWS credentials verification ==="
AWS_CREDENTIALS_OK="no"
AWS_ACCOUNT_ID=""
AWS_USER_ARN=""
if aws sts get-caller-identity > "${OUT}/aws_identity.txt" 2>&1; then
    AWS_CREDENTIALS_OK="yes"
    AWS_ACCOUNT_ID=$(grep -oE '"Account":[[:space:]]*"[^"]+' "${OUT}/aws_identity.txt" | cut -d'"' -f4 || true)
    AWS_USER_ARN=$(grep -oE '"Arn":[[:space:]]*"[^"]+' "${OUT}/aws_identity.txt" | cut -d'"' -f4 || true)
    echo "  AWS credentials OK ✅"
    echo "  Account: ${AWS_ACCOUNT_ID:-<unknown>}"
    echo "  ARN:     ${AWS_USER_ARN:-<unknown>}"
else
    echo "  AWS credentials FAILED ❌"
    aws sts get-caller-identity 2>&1 | head -5 >&2 || true
fi
echo ""

echo "=== Anyscale cloud verification ==="
CLOUD_VISIBLE="no"
if anyscale cloud list > "${OUT}/cloud_list.txt" 2>&1; then
    if grep -qE 'anyscale-poc-tokyo-cloud|cld_z9sb92gg9vewg9l5181hlat7wv' "${OUT}/cloud_list.txt"; then
        CLOUD_VISIBLE="yes"
        echo "  anyscale-poc-tokyo-cloud visible ✅"
    else
        echo "  anyscale-poc-tokyo-cloud NOT FOUND in cloud list"
        cat "${OUT}/cloud_list.txt" | head -10
    fi
else
    echo "  anyscale cloud list failed ❌"
fi
echo ""

BASELINE_OK="yes"
if [[ "${AWS_CREDENTIALS_OK}" == "no" ]]; then
    BASELINE_OK="no"
fi
if [[ "${CLOUD_VISIBLE}" == "no" ]]; then
    BASELINE_OK="no"
fi

echo "================================================================="
echo "  STRUCTURED OUTPUT"
echo "================================================================="
echo "BASELINE_OK=${BASELINE_OK}"
echo "PRODJOB_ID=${PRODJOB_ID}"
echo "JOB_STATE=${JOB_STATE}"
echo "COMPUTE_CONFIG=${COMPUTE_CONFIG:-<unknown>}"
echo "COMPUTE_CONFIG_EXISTS=${COMPUTE_CONFIG_EXISTS}"
echo "IMAGE_URI=${IMAGE_URI:-<unknown>}"
echo "IMAGE_EXISTS=${IMAGE_EXISTS}"
echo "AWS_CREDENTIALS_OK=${AWS_CREDENTIALS_OK}"
echo "AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-<unknown>}"
echo "AWS_USER_ARN=${AWS_USER_ARN:-<unknown>}"
echo "CLOUD_VISIBLE=${CLOUD_VISIBLE}"
echo "OUT_BUNDLE=${OUT}"
echo "================================================================="

if [[ "${BASELINE_OK}" == "no" ]]; then
    exit 2
fi
exit 0
