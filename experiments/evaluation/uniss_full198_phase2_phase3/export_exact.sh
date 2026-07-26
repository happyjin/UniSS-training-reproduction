#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {phase2|phase3}" >&2
  exit 2
fi

STAGE="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train}"
HF_REFERENCE="${HF_REFERENCE:-${REPO_ROOT}/checkpoints/qwen2_0p5b_uniss_vocab_hf}"

case "${STAGE}" in
  phase2)
    CHECKPOINT_ROOT="${PHASE2_CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/uniss_qwen0p5b_phase2_unist198_from_phase1_fast_decay_v4}"
    REQUESTED_ITERATION="${PHASE2_ITERATION:-}"
    ;;
  phase3)
    CHECKPOINT_ROOT="${PHASE3_CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4}"
    REQUESTED_ITERATION="${PHASE3_ITERATION:-}"
    ;;
  *)
    echo "Unsupported stage: ${STAGE}" >&2
    exit 2
    ;;
esac

TRACKER="${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"
if [[ ! -f "${TRACKER}" ]]; then
  echo "Missing checkpoint tracker: ${TRACKER}" >&2
  exit 1
fi
if [[ -z "${REQUESTED_ITERATION}" ]]; then
  REQUESTED_ITERATION="$(<"${TRACKER}")"
fi
if [[ ! "${REQUESTED_ITERATION}" =~ ^[0-9]+$ ]]; then
  echo "Invalid checkpoint iteration: ${REQUESTED_ITERATION}" >&2
  exit 1
fi

printf -v ITER_TAG 'iter_%07d' "$((10#${REQUESTED_ITERATION}))"
MEGATRON_PATH="${CHECKPOINT_ROOT}/${ITER_TAG}"
HF_OUTPUT="${HF_OUTPUT:-${REPO_ROOT}/checkpoints/exported_hf/qwen0p5b_${STAGE}_unist198_${ITER_TAG}_hf}"

if [[ ! -d "${MEGATRON_PATH}" ]]; then
  echo "Exact checkpoint directory does not exist: ${MEGATRON_PATH}" >&2
  exit 1
fi
if [[ -e "${HF_OUTPUT}" ]]; then
  echo "Refusing to overwrite existing HF export: ${HF_OUTPUT}" >&2
  exit 1
fi

PARTIAL_OUTPUT="${HF_OUTPUT}.partial.$$"
FINAL_CREATED=0
COMPLETED=0
cleanup() {
  if [[ -d "${PARTIAL_OUTPUT}" ]]; then
    rm -rf -- "${PARTIAL_OUTPUT}"
  fi
  if [[ "${FINAL_CREATED}" == "1" && "${COMPLETED}" != "1" && -d "${HF_OUTPUT}" ]]; then
    rm -rf -- "${HF_OUTPUT}"
  fi
}
trap cleanup EXIT

"${REPO_ROOT}/scripts/convert_uniss_checkpoint.sh" export \
  --hf-model "${HF_REFERENCE}" \
  --megatron-path "${MEGATRON_PATH}" \
  --hf-output "${PARTIAL_OUTPUT}" \
  --model-type gpt \
  --no-progress

mv "${PARTIAL_OUTPUT}" "${HF_OUTPUT}"
FINAL_CREATED=1

"${ENV_ROOT}/bin/python" "${REPO_ROOT}/experiments/evaluation/uniss_full198_phase2_phase3/verify_hf_export.py" \
  --model "${HF_OUTPUT}" \
  --source-checkpoint "${MEGATRON_PATH}" \
  --expected-model-vocab-size 180480 \
  --expected-tokenizer-size 180407

COMPLETED=1
echo "HF_OUTPUT=${HF_OUTPUT}"
