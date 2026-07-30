#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE=""
SHARD_INDEX=""
GPU=""
OUTPUT_ROOT=""
LIMIT_RECORDS=""
SIDE=""
SKIP_SHA256=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --shard-index) SHARD_INDEX="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --limit-records) LIMIT_RECORDS="$2"; shift 2 ;;
    --side) SIDE="$2"; shift 2 ;;
    --skip-sha256) SKIP_SHA256=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v1/stage_ab.env}"
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"

[[ -n "${SHARD_INDEX}" && -n "${GPU}" ]] || { echo "--shard-index and --gpu are required" >&2; exit 2; }
OUTPUT_ROOT="${OUTPUT_ROOT:-${STAGE_A_ROOT}}"
SIDE="${SIDE:-${STAGE_A_SIDE}}"
input="${UNIST_ROOT}/train-$(printf '%05d' "${SHARD_INDEX}").parquet"
part_dir="${OUTPUT_ROOT}/parts/train-$(printf '%05d' "${SHARD_INDEX}")"
mkdir -p "${part_dir}" "${LOG_ROOT}/stage_a"

cmd=(python -m training.simul_uniss.subsecond_v1.stage_a prepare-part
  --input "${input}"
  --output-dir "${part_dir}"
  --bicodec-checkpoint "${BICODEC_CHECKPOINT}"
  --device cuda:0
  --side "${SIDE}"
)
if [[ -n "${LIMIT_RECORDS}" ]]; then
  cmd+=(--limit-records "${LIMIT_RECORDS}")
fi
if [[ "${SKIP_SHA256}" == "1" ]]; then
  cmd+=(--skip-sha256)
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${cmd[@]}" \
  2>&1 | tee -a "${LOG_ROOT}/stage_a/train-$(printf '%05d' "${SHARD_INDEX}").log"
