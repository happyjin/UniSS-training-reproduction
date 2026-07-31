#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/opt/dlami/nvme/jasonleeeli/projects/UniSS}"
CONFIG="${1:-${REPO_ROOT}/configs/experiments/simul_uniss_subsecond_v2/formal_15shard.env}"
MODE="${2:-all}"
# shellcheck source=/dev/null
source "${CONFIG}"

mkdir -p "${A45_ROOT}" "${A68_ROOT}" "${FORMAL_STAGE_A_ROOT}" "${LOG_ROOT}"
TOTAL_TASKS=$((SHARD_COUNT * CHUNKS_PER_SHARD))
TOTAL_LANES=$((GPU_COUNT * LANES_PER_GPU))

run_a45_task() {
  local task="$1" gpu="$2"
  local shard=$((task / CHUNKS_PER_SHARD))
  local chunk=$((task % CHUNKS_PER_SHARD))
  local start=$((chunk * RECORDS_PER_CHUNK))
  local shard_name
  shard_name=$(printf 'train-%05d' "${shard}")
  local chunk_name
  chunk_name=$(printf 'chunk-%02d' "${chunk}")
  local input="${V1_STAGE_A_PARTS}/${shard_name}/manifest.jsonl"
  local output="${A45_ROOT}/${shard_name}/${chunk_name}"
  local log="${LOG_ROOT}/a45_${shard_name}_${chunk_name}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" HF_HOME="${HF_HOME}" HF_HUB_DISABLE_XET=1 TRANSFORMERS_OFFLINE=1 \
    "${ALIGN_PYTHON}" -m training.simul_uniss.subsecond_v2.prepare_a45 \
      --input-manifest "${input}" \
      --output-dir "${output}" \
      --forced-aligner-model "${FORCED_ALIGNER_MODEL}" \
      --whispervq-model "${WHISPERVQ_MODEL}" \
      --bicodec-checkpoint "${BICODEC_CHECKPOINT}" \
      --device cuda:0 \
      --start-index "${start}" \
      --limit-records "${RECORDS_PER_CHUNK}" \
      --worker-batch-size "${A45_WORKER_BATCH_SIZE}" \
      --alignment-batch-size "${A45_ALIGNMENT_BATCH_SIZE}" \
      --minimum-alignment-coverage "${A45_MINIMUM_COVERAGE}" \
      2>&1 | tee "${log}"
}

run_a68_task() {
  local task="$1" gpu="$2"
  local shard=$((task / CHUNKS_PER_SHARD))
  local chunk=$((task % CHUNKS_PER_SHARD))
  local shard_name
  shard_name=$(printf 'train-%05d' "${shard}")
  local chunk_name
  chunk_name=$(printf 'chunk-%02d' "${chunk}")
  local input="${A45_ROOT}/${shard_name}/${chunk_name}/a45_manifest.jsonl"
  local output="${A68_ROOT}/${shard_name}/${chunk_name}"
  local log="${LOG_ROOT}/a68_${shard_name}_${chunk_name}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" HF_HOME="${HF_HOME}" HF_HUB_DISABLE_XET=1 TRANSFORMERS_OFFLINE=1 \
    "${ALIGN_PYTHON}" -m training.simul_uniss.subsecond_v2.prepare_a68 \
      --input-manifest "${input}" \
      --output-dir "${output}" \
      --word-aligner-model "${WORD_ALIGNER_MODEL}" \
      --device cuda:0 \
      --batch-size "${A68_BATCH_SIZE}" \
      --minimum-target-link-coverage "${A68_MINIMUM_LINK_COVERAGE}" \
      2>&1 | tee "${log}"
}

run_phase() {
  local phase="$1"
  local pids=()
  for ((lane = 0; lane < TOTAL_LANES; lane++)); do
    (
      gpu=$((lane % GPU_COUNT))
      for ((task = lane; task < TOTAL_TASKS; task += TOTAL_LANES)); do
        if [[ "${phase}" == "a45" ]]; then
          run_a45_task "${task}" "${gpu}"
        else
          run_a68_task "${task}" "${gpu}"
        fi
      done
    ) &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  [[ "${failed}" -eq 0 ]]
}

if [[ "${MODE}" == "all" || "${MODE}" == "a45" ]]; then
  run_phase a45
fi
if [[ "${MODE}" == "all" || "${MODE}" == "a68" ]]; then
  run_phase a68
fi
if [[ "${MODE}" == "all" || "${MODE}" == "assemble" ]]; then
  "${TRAIN_PYTHON}" -m training.simul_uniss.subsecond_v2.assemble_stage_a \
    --a45-root "${A45_ROOT}" \
    --a68-root "${A68_ROOT}" \
    --output-dir "${FORMAL_STAGE_A_ROOT}" \
    --expected-parts "${TOTAL_TASKS}" \
    --expected-records $((SHARD_COUNT * RECORDS_PER_CHUNK * CHUNKS_PER_SHARD)) \
    2>&1 | tee "${LOG_ROOT}/assemble.log"
fi

