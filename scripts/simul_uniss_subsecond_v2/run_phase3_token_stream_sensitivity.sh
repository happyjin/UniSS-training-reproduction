#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
# shellcheck source=/dev/null
source "${USER_ROOT}/env_recovery/uniss-train-20260721/activate_uniss.sh"

export TMPDIR="${USER_ROOT}/tmp"
export XDG_CACHE_HOME="${USER_ROOT}/cache/xdg"
export HF_HOME="${USER_ROOT}/cache/huggingface"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${REPO_ROOT}/reports/simul_uniss_subsecond_v2"

GPU="${GPU:-0}"
SAMPLES="${SAMPLES:-8}"
AUDIO_WORKERS="${AUDIO_WORKERS:-8}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/reports/simul_uniss_subsecond_v2/stage_b_phase3_token_stream_sensitivity_v1.json}"

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" python -m \
  training.simul_uniss.subsecond_v2.evaluate_phase3_token_streams \
  --manifest data/processed/simul_uniss_subsecond_v2/formal_15shard_v1/stage_a_formal/formal_valid_manifest.jsonl \
  --whispervq-model pretrained_models/UniSS/glm4_tokenizer \
  --student-checkpoint checkpoints/simul_uniss_subsecond_v2/stage_b_latent_formal_15shard_v1/best.pt \
  --phase3-model checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf \
  --output "${OUTPUT}" \
  --device cuda:0 \
  --samples "${SAMPLES}" \
  --audio-workers "${AUDIO_WORKERS}" \
  --chunk-ms 160 \
  --lookahead-ms 80 160 320 640 \
  --max-audio-seconds 8 \
  --max-new-tokens 192
