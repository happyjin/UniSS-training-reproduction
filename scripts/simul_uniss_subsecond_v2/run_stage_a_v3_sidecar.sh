#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
# shellcheck source=/dev/null
source "${USER_ROOT}/env_recovery/uniss-train-20260721/activate_uniss.sh"

export TMPDIR="${USER_ROOT}/tmp"
export XDG_CACHE_HOME="${USER_ROOT}/cache/xdg"
export HF_HOME="${USER_ROOT}/cache/huggingface"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}"

MODE="${MODE:-clone}"
WORLD_SIZE="${WORLD_SIZE:-8}"
LIMIT_RECORDS="${LIMIT_RECORDS:-0}"
AUDIO_WORKERS="${AUDIO_WORKERS:-4}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-16}"
RECORDS_PER_SHARD="${RECORDS_PER_SHARD:-512}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/data/processed/simul_uniss_subsecond_v2/stage_a_v3_${MODE}_15shard_v1}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/data/processed/simul_uniss_subsecond_v2/formal_15shard_v1/stage_a_formal/formal_train_manifest.jsonl}"

[[ "${MODE}" == "clone" || "${MODE}" == "prefix80" ]] || {
  echo "MODE must be clone or prefix80" >&2
  exit 2
}
mkdir -p "${OUTPUT_ROOT}/logs"

limit_args=()
if [[ "${LIMIT_RECORDS}" -gt 0 ]]; then
  limit_args+=(--limit-records "${LIMIT_RECORDS}")
fi

pids=()
for ((rank=0; rank<WORLD_SIZE; rank++)); do
  CUDA_VISIBLE_DEVICES="${rank}" python -m \
    training.simul_uniss.subsecond_v2.prepare_stage_a_v3_sidecar \
    --manifest "${MANIFEST}" \
    --whispervq-model "${REPO_ROOT}/pretrained_models/UniSS/glm4_tokenizer" \
    --output-dir "${OUTPUT_ROOT}" \
    --mode "${MODE}" \
    --device cuda:0 \
    --rank "${rank}" \
    --world-size "${WORLD_SIZE}" \
    --audio-workers "${AUDIO_WORKERS}" \
    --teacher-batch-size "${TEACHER_BATCH_SIZE}" \
    --records-per-shard "${RECORDS_PER_SHARD}" \
    --chunk-ms 160 \
    --lookahead-ms 80 \
    --codebook-topk 32 \
    "${limit_args[@]}" \
    > "${OUTPUT_ROOT}/logs/part-${rank}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
[[ "${failed}" -eq 0 ]] || exit 1

python -m training.simul_uniss.subsecond_v2.assemble_stage_a_v3_sidecar \
  --root "${OUTPUT_ROOT}" --world-size "${WORLD_SIZE}"
