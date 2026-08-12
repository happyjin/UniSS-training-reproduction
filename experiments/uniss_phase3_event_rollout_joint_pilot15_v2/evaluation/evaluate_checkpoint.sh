#!/usr/bin/env bash
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EVAL_DIR}/../../../.." && pwd)"
USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"

ITERATION="${ITERATION:?Set ITERATION to a saved checkpoint iteration}"
SPLIT="${SPLIT:-valid}"
case "${SPLIT}" in
  train|valid) ;;
  *) echo "SPLIT must be train or valid" >&2; exit 2 ;;
esac

RUN_NAME="${RUN_NAME:-uniss_phase3_event_rollout_joint_pilot15_v2_formal_v1}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/checkpoints/${RUN_NAME}/iter_$(printf '%07d' "${ITERATION}")}"
FORMAL_ROOT="${REPO_ROOT}/data/processed/simul_uniss_subsecond_v2/formal_15shard_v1/stage_a_formal"
FORMAL_MANIFEST="${FORMAL_MANIFEST:-${FORMAL_ROOT}/formal_${SPLIT}_manifest.jsonl}"
SPEAKER_FORMAL_MANIFEST="${SPEAKER_FORMAL_MANIFEST:-${FORMAL_ROOT}/formal_train_manifest.jsonl}"
BASE_MODEL="${BASE_MODEL:-${REPO_ROOT}/checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf}"
SPEECH_TOKENIZER="${SPEECH_TOKENIZER:-${REPO_ROOT}/pretrained_models/UniSS}"
WHISPERVQ_MODEL="${WHISPERVQ_MODEL:-${SPEECH_TOKENIZER}/glm4_tokenizer}"
INFERENCE_PYTHON="${INFERENCE_PYTHON:-${USER_ROOT}/conda_envs/uniss-offline-demo/bin/python}"
TAG="${TAG:-strict_exact_runtime_v2}"
SAMPLES="${SAMPLES:-2}"
DEVICE="${DEVICE:-cuda:0}"
FUSE_TICKS="${FUSE_TICKS:-1}"
STATIC_CACHE="${STATIC_CACHE:-1}"
EXPORT_ROOT="${EXPORT_ROOT:-${REPO_ROOT}/reports/${RUN_NAME}/runtime_exports/iter_$(printf '%07d' "${ITERATION}")}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/reports/${RUN_NAME}/evaluation/${SPLIT}_${TAG}/iter_$(printf '%07d' "${ITERATION}")}"

case "${FUSE_TICKS}" in 0|1) ;; *) echo "FUSE_TICKS must be 0 or 1" >&2; exit 2 ;; esac
case "${STATIC_CACHE}" in 0|1) ;; *) echo "STATIC_CACHE must be 0 or 1" >&2; exit 2 ;; esac

for value in "${CHECKPOINT}/.metadata" "${BASE_MODEL}/config.json" \
  "${FORMAL_MANIFEST}" "${FORMAL_MANIFEST}.offsets.bin" \
  "${SPEAKER_FORMAL_MANIFEST}" "${SPEAKER_FORMAL_MANIFEST}.offsets.bin" \
  "${WHISPERVQ_MODEL}/model.safetensors" "${INFERENCE_PYTHON}"; do
  [[ -e "${value}" ]] || { echo "Missing fixed15 v2 evaluation input: ${value}" >&2; exit 1; }
done
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing to overwrite ${OUTPUT}" >&2; exit 1; }

export HF_HOME="${HF_HOME:-${USER_ROOT}/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PYTORCH_KERNEL_CACHE_PATH="${PYTORCH_KERNEL_CACHE_PATH:-${USER_ROOT}/cache/pytorch/kernels}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${USER_ROOT}/cache/cuda}"
export TMPDIR="${TMPDIR:-${USER_ROOT}/tmp}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "${PYTORCH_KERNEL_CACHE_PATH}" "${CUDA_CACHE_PATH}" "${TMPDIR}"

RUNTIME_FLAGS=()
[[ "${FUSE_TICKS}" == 1 ]] && RUNTIME_FLAGS+=(--fuse-ticks)
[[ "${STATIC_CACHE}" == 1 ]] && RUNTIME_FLAGS+=(--static-cache)

exec "${INFERENCE_PYTHON}" \
  -m experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.evaluate_checkpoint \
  --checkpoint "${CHECKPOINT}" --base-model "${BASE_MODEL}" --export "${EXPORT_ROOT}" \
  --formal-manifest "${FORMAL_MANIFEST}" \
  --speaker-formal-manifest "${SPEAKER_FORMAL_MANIFEST}" \
  --speaker-source-index 0 --whispervq-model "${WHISPERVQ_MODEL}" \
  --speech-tokenizer "${SPEECH_TOKENIZER}" --output "${OUTPUT}" --device "${DEVICE}" \
  --samples "${SAMPLES}" --maximum-drain-ticks 32 --minimum-text-similarity 0.50 \
  --maximum-rtf 1.0 --maximum-first-audio-wall-ms 1000 "${RUNTIME_FLAGS[@]}"
