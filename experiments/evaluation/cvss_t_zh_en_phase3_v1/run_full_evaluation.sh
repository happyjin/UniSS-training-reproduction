#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TOKENIZED_ROOT="${TOKENIZED_ROOT:-/opt/dlami/nvme/jasonleeeli/CVSS/tokenized/cvss_t_zh_en_v1}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/eval_outputs/cvss_t_zh_en_phase3_full198_iter_0009075_v1}"
GPU_LIST_VALUE="${EVAL_GPU_LIST:-0,1,2,3,4,5,6,7}"
ZH_EN_MANIFEST="${ZH_EN_MANIFEST:-${TOKENIZED_ROOT}/manifests/zh_en/unist_test_all.jsonl}"
EN_ZH_MANIFEST="${EN_ZH_MANIFEST:-${TOKENIZED_ROOT}/manifests/en_zh/unist_test_all.jsonl}"
ZH_EN_OUTPUT="${OUTPUT_BASE}/cvss_t_phase3_full_cmn_to_eng"
EN_ZH_OUTPUT="${OUTPUT_BASE}/cvss_t_phase3_full_eng_to_cmn"
REPORT_DIR="${OUTPUT_BASE}/report"

for manifest in "${ZH_EN_MANIFEST}" "${EN_ZH_MANIFEST}"; do
  [[ -f "${manifest}" ]] || {
    echo "Missing tokenized CVSS-T manifest: ${manifest}" >&2
    echo "Run tokenize_8gpu.sh first." >&2
    exit 1
  }
done

run_direction() {
  local manifest="$1"
  local output="$2"
  local direction="$3"
  RESUME=1 EVAL_GPU_LIST="${GPU_LIST_VALUE}" \
    "${REPO_ROOT}/experiments/evaluation/cvss_t_zh_en_phase3_v1/run_vllm_eval.sh" \
    "${manifest}" "${output}"
  EXPECTED_PAIRS=4897 EVAL_GPU_LIST="${GPU_LIST_VALUE}" \
    "${REPO_ROOT}/experiments/evaluation/cvss_t_zh_en_phase3_v1/run_objective_metrics.sh" \
    "${output}" "${direction}"
}

# One direction uses all selected GPUs. Running directions sequentially avoids
# loading duplicate vLLM + ASR + metric stacks on the same device.
run_direction "${ZH_EN_MANIFEST}" "${ZH_EN_OUTPUT}" "cmn->eng"
run_direction "${EN_ZH_MANIFEST}" "${EN_ZH_OUTPUT}" "eng->cmn"

EXPECTED_PAIRS=4897 \
  "${REPO_ROOT}/experiments/evaluation/cvss_t_zh_en_phase3_v1/build_report.sh" \
  "${ZH_EN_OUTPUT}" "${EN_ZH_OUTPUT}" "${REPORT_DIR}"

echo "CVSS-T full evaluation completed under ${OUTPUT_BASE}"
