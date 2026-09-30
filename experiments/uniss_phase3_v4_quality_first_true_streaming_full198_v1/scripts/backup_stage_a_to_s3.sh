#!/usr/bin/env bash
set -euo pipefail

# Backup full198 Stage A artifacts to the shared UniSS S3 prefix.
# Destinations were probed empty before this run.

export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-ap-southeast-3}"
S3="${S3:?set S3 to the shared UniSS S3 project prefix}"
REPO="${REPO:-/opt/dlami/nvme/neuhao/UniSS}"
NAME=uniss_phase3_v4_quality_first_true_streaming_full198_v1
HF_DIR="${NAME}_stage_a_formal8_20260903T054603Z_iter_0004797_hf"

sync_one() {
  local src=$1
  local dst=$2
  shift 2
  echo
  echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) sync ${src} -> ${dst} ====="
  aws s3 sync "${src}" "${dst}" --region "${AWS_DEFAULT_REGION}" "$@"
  echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) done ${dst} ====="
}

echo "start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "identity=$(aws sts get-caller-identity --query Arn --output text)"
echo "s3=${S3}"

sync_one \
  "${REPO}/reports/${NAME}/" \
  "${S3}/reports/${NAME}/"

sync_one \
  "${REPO}/logs/${NAME}/stage_a_formal/" \
  "${S3}/logs/${NAME}/stage_a_formal/"

sync_one \
  "${REPO}/logs/${NAME}/stage_a_checkpoint_diagnosis/" \
  "${S3}/logs/${NAME}/stage_a_checkpoint_diagnosis/"

sync_one \
  "${REPO}/experiments/${NAME}/" \
  "${S3}/experiments/${NAME}/" \
  --no-follow-symlinks \
  --exclude "__pycache__/*" \
  --exclude "*/__pycache__/*"

sync_one \
  "${REPO}/checkpoints/exported_hf/${HF_DIR}/" \
  "${S3}/checkpoints/exported_hf/${HF_DIR}/"

sync_one \
  "${REPO}/checkpoints/${NAME}/" \
  "${S3}/checkpoints/${NAME}/"

echo
echo "===== verify prefixes ====="
for prefix in \
  "reports/${NAME}/" \
  "logs/${NAME}/stage_a_formal/" \
  "logs/${NAME}/stage_a_checkpoint_diagnosis/" \
  "experiments/${NAME}/" \
  "checkpoints/exported_hf/${HF_DIR}/" \
  "checkpoints/${NAME}/"
do
  count=$(aws s3 ls "${S3}/${prefix}" --recursive --summarize | awk '/Total Objects/ {print $3}')
  size=$(aws s3 ls "${S3}/${prefix}" --recursive --summarize | awk '/Total Size/ {print $3}')
  echo "${prefix} objects=${count:-0} bytes=${size:-0}"
done

echo "finish=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
