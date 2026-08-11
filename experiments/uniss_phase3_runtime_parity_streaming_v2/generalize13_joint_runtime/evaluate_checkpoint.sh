#!/usr/bin/env bash
set -euo pipefail

V13_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${V13_DIR}/../../.." && pwd)"
# shellcheck source=/dev/null
source "${V13_DIR}/config_canary.env"

ITERATION="${ITERATION:?Set ITERATION to the checkpoint step}"
TAG="${TAG:-held_out2_strict_v13_v1}"
CHECKPOINT="${CHECKPOINT:-${SAVE_DIR}/iter_$(printf '%07d' "${ITERATION}")}"
EVAL_ROOT="${REPO_ROOT}/reports/uniss_phase3_runtime_parity_streaming_v2/${EXPERIMENT_NAME}_${TAG}/iter_$(printf '%07d' "${ITERATION}")"
EXPORT_ROOT="${REPO_ROOT}/reports/uniss_phase3_runtime_parity_streaming_v2/runtime_exports/${EXPERIMENT_NAME}_iter_$(printf '%07d' "${ITERATION}")_${TAG}"
FORMAL_ROOT="${REPO_ROOT}/data/processed/simul_uniss_subsecond_v2/formal_15shard_v1/stage_a_formal"
FORMAL_MANIFEST="${FORMAL_MANIFEST:-${FORMAL_ROOT}/formal_valid_manifest.jsonl}"
SPEAKER_FORMAL_MANIFEST="${SPEAKER_FORMAL_MANIFEST:-${FORMAL_ROOT}/formal_train_manifest.jsonl}"
BASE_MODEL="${REPO_ROOT}/checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf"
SPEECH_TOKENIZER="${REPO_ROOT}/pretrained_models/UniSS"
WHISPERVQ_MODEL="${SPEECH_TOKENIZER}/glm4_tokenizer"
INFERENCE_PYTHON="${USER_ROOT}/conda_envs/uniss-offline-demo/bin/python"
SAMPLES="${SAMPLES:-2}"

for value in "${CHECKPOINT}/.metadata" "${BASE_MODEL}/config.json" \
  "${FORMAL_MANIFEST}" "${FORMAL_MANIFEST}.offsets.bin" \
  "${SPEAKER_FORMAL_MANIFEST}" "${SPEAKER_FORMAL_MANIFEST}.offsets.bin" \
  "${WHISPERVQ_MODEL}/model.safetensors" "${INFERENCE_PYTHON}"; do
  [[ -e "${value}" ]] || { echo "Missing generalize13 evaluation input: ${value}" >&2; exit 1; }
done
[[ ! -e "${EVAL_ROOT}" ]] || { echo "Refusing to overwrite ${EVAL_ROOT}" >&2; exit 1; }

export HF_HOME="${HF_HOME:-${USER_ROOT}/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PYTORCH_KERNEL_CACHE_PATH="${PYTORCH_KERNEL_CACHE_PATH:-${USER_ROOT}/cache/pytorch/kernels}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${USER_ROOT}/cache/cuda}"
export TMPDIR="${TMPDIR:-${USER_ROOT}/tmp}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "${PYTORCH_KERNEL_CACHE_PATH}" "${CUDA_CACHE_PATH}" "${TMPDIR}"

exec "${INFERENCE_PYTHON}" -m experiments.uniss_phase3_runtime_parity_streaming_v2.generalize13_joint_runtime.evaluate_checkpoint \
  --checkpoint "${CHECKPOINT}" --base-model "${BASE_MODEL}" --export "${EXPORT_ROOT}" \
  --formal-manifest "${FORMAL_MANIFEST}" \
  --speaker-formal-manifest "${SPEAKER_FORMAL_MANIFEST}" \
  --speaker-source-index 0 --whispervq-model "${WHISPERVQ_MODEL}" \
  --speech-tokenizer "${SPEECH_TOKENIZER}" --output "${EVAL_ROOT}" --device cuda:0 \
  --samples "${SAMPLES}" --maximum-drain-ticks 8 --minimum-text-similarity 0.98 \
  --maximum-rtf 1.0 --maximum-first-audio-wall-ms 1000 --fuse-ticks
