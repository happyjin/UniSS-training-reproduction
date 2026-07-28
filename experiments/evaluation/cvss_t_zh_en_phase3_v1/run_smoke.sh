#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TOKENIZED_ROOT="${TOKENIZED_ROOT:-/opt/dlami/nvme/jasonleeeli/CVSS/tokenized/cvss_t_zh_en_v1}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/eval_outputs/cvss_t_zh_en_phase3_smoke10_v1}"
GPU_LIST_VALUE="${EVAL_GPU_LIST:-0}"
ZH_EN_MANIFEST="${TOKENIZED_ROOT}/manifests/zh_en/unist_test_smoke_10.jsonl"
EN_ZH_MANIFEST="${TOKENIZED_ROOT}/manifests/en_zh/unist_test_smoke_10.jsonl"
ZH_EN_OUTPUT="${OUTPUT_BASE}/cvss_t_phase3_smoke_cmn_to_eng"
EN_ZH_OUTPUT="${OUTPUT_BASE}/cvss_t_phase3_smoke_eng_to_cmn"

for manifest in "${ZH_EN_MANIFEST}" "${EN_ZH_MANIFEST}"; do
  [[ -f "${manifest}" ]] || { echo "Missing smoke manifest: ${manifest}" >&2; exit 1; }
done

run_direction() {
  local manifest="$1"
  local output="$2"
  local direction="$3"
  RESUME=1 EVAL_GPU_LIST="${GPU_LIST_VALUE}" REQUEST_BATCH_SIZE=32 MAX_NUM_SEQS=32 \
    "${REPO_ROOT}/experiments/evaluation/cvss_t_zh_en_phase3_v1/run_vllm_eval.sh" \
    "${manifest}" "${output}"
  EXPECTED_PAIRS=10 EVAL_GPU_LIST="${GPU_LIST_VALUE}" ASR_BATCH_SIZE=8 AUTOPCP_BATCH_SIZE=16 \
    "${REPO_ROOT}/experiments/evaluation/cvss_t_zh_en_phase3_v1/run_objective_metrics.sh" \
    "${output}" "${direction}"
}

run_direction "${ZH_EN_MANIFEST}" "${ZH_EN_OUTPUT}" "cmn->eng"
run_direction "${EN_ZH_MANIFEST}" "${EN_ZH_OUTPUT}" "eng->cmn"
EXPECTED_PAIRS=10 \
  "${REPO_ROOT}/experiments/evaluation/cvss_t_zh_en_phase3_v1/build_report.sh" \
  "${ZH_EN_OUTPUT}" "${EN_ZH_OUTPUT}" "${OUTPUT_BASE}/report"

echo "CVSS-T smoke evaluation completed under ${OUTPUT_BASE}"
