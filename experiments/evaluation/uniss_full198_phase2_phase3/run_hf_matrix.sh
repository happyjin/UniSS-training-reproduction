#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {smoke|listen}" >&2
  exit 2
fi

KIND="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MANIFEST_ROOT="${REPO_ROOT}/experiments/evaluation/uniss_full198_phase2_phase3/manifests"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

case "${KIND}" in
  smoke) MANIFEST="${MANIFEST_ROOT}/unist_dev_smoke_3.jsonl" ;;
  listen) MANIFEST="${MANIFEST_ROOT}/unist_dev_listen_50.jsonl" ;;
  *) echo "Unsupported matrix kind: ${KIND}" >&2; exit 2 ;;
esac

PHASE2_ITERATION="${PHASE2_ITERATION:-$(<"${REPO_ROOT}/checkpoints/uniss_qwen0p5b_phase2_unist198_from_phase1_fast_decay_v4/latest_checkpointed_iteration.txt")}"
PHASE3_ITERATION="${PHASE3_ITERATION:-$(<"${REPO_ROOT}/checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4/latest_checkpointed_iteration.txt")}"
printf -v PHASE2_TAG 'iter_%07d' "$((10#${PHASE2_ITERATION}))"
printf -v PHASE3_TAG 'iter_%07d' "$((10#${PHASE3_ITERATION}))"

PHASE2_HF="${PHASE2_HF:-${REPO_ROOT}/checkpoints/exported_hf/qwen0p5b_phase2_unist198_${PHASE2_TAG}_hf}"
PHASE3_HF="${PHASE3_HF:-${REPO_ROOT}/checkpoints/exported_hf/qwen0p5b_phase3_unist198_${PHASE3_TAG}_hf}"

for model in "${PHASE2_HF}" "${PHASE3_HF}"; do
  if [[ ! -d "${model}" ]]; then
    echo "Missing HF export: ${model}" >&2
    exit 1
  fi
done

"${REPO_ROOT}/experiments/evaluation/uniss_full198_phase2_phase3/run_hf_manifest.sh" \
  phase2 "${PHASE2_HF}" "${MANIFEST}" \
  "${REPO_ROOT}/eval_outputs/qwen0p5b_phase2_unist198_${PHASE2_TAG}_unist_dev_${KIND}_${RUN_ID}"

"${REPO_ROOT}/experiments/evaluation/uniss_full198_phase2_phase3/run_hf_manifest.sh" \
  phase3 "${PHASE3_HF}" "${MANIFEST}" \
  "${REPO_ROOT}/eval_outputs/qwen0p5b_phase3_unist198_${PHASE3_TAG}_unist_dev_${KIND}_${RUN_ID}"
