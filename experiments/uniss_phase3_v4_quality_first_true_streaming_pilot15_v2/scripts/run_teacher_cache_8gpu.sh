#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${EXPERIMENT_DIR}/experiment.env"
cd "${REPO_ROOT}"

: "${CACHE_PACKS:?CACHE_PACKS is required}"
: "${CACHE_ROOT:?CACHE_ROOT is required}"
: "${CACHE_COVERAGE_EPOCHS:?CACHE_COVERAGE_EPOCHS is required}"
: "${CACHE_MAX_ACOUSTICS:?CACHE_MAX_ACOUSTICS is required}"
CACHE_WORLD_SIZE=${CACHE_WORLD_SIZE:-8}
CACHE_TOPK=${CACHE_TOPK:-32}
CACHE_TEMPERATURE=${CACHE_TEMPERATURE:-1.5}
CACHE_REFERENCE_ANCHOR=${CACHE_REFERENCE_ANCHOR:-0.5}
CACHE_RECORDS_PER_BUNDLE=${CACHE_RECORDS_PER_BUNDLE:-256}
CACHE_PROGRESS_INTERVAL=${CACHE_PROGRESS_INTERVAL:-100}
CACHE_LIMIT_PACKS=${CACHE_LIMIT_PACKS:-}

[[ -f "${CACHE_PACKS}" ]] || { echo "missing teacher source packs" >&2; exit 2; }
[[ ! -e "${CACHE_ROOT}" ]] || {
  echo "refusing to overwrite teacher cache: ${CACHE_ROOT}" >&2
  exit 2
}
mkdir -p "${CACHE_ROOT}/logs" "${TMPDIR}"

pids=()
for rank in $(seq 0 $((CACHE_WORLD_SIZE - 1))); do
  printf -v tag '%02d' "${rank}"
  command=(
    "${PYTHON_BIN}" -m
    experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.build_teacher_cache
    --packs "${CACHE_PACKS}"
    --model "${PHASE3_HF_CHECKPOINT}"
    --output-dir "${CACHE_ROOT}/part_${tag}"
    --rank "${rank}"
    --world-size "${CACHE_WORLD_SIZE}"
    --device cuda:0
    --coverage-epochs "${CACHE_COVERAGE_EPOCHS}"
    --max-acoustics-per-pack "${CACHE_MAX_ACOUSTICS}"
    --topk "${CACHE_TOPK}"
    --temperature "${CACHE_TEMPERATURE}"
    --require-reference-in-topk
    --reference-anchor "${CACHE_REFERENCE_ANCHOR}"
    --records-per-bundle "${CACHE_RECORDS_PER_BUNDLE}"
    --progress-interval "${CACHE_PROGRESS_INTERVAL}"
  )
  [[ -n "${CACHE_LIMIT_PACKS}" ]] && command+=(--limit-packs "${CACHE_LIMIT_PACKS}")
  CUDA_VISIBLE_DEVICES="${rank}" "${command[@]}" \
    >"${CACHE_ROOT}/logs/part_${tag}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
if (( failed != 0 )); then
  tail -n 100 "${CACHE_ROOT}"/logs/part_??.log >&2 || true
  exit 1
fi

merge=(
  "${PYTHON_BIN}" -m
  experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.merge_teacher_cache
  --packs "${CACHE_PACKS}"
  --parts-root "${CACHE_ROOT}"
  --world-size "${CACHE_WORLD_SIZE}"
  --coverage-epochs "${CACHE_COVERAGE_EPOCHS}"
  --max-acoustics-per-pack "${CACHE_MAX_ACOUSTICS}"
)
[[ -n "${CACHE_LIMIT_PACKS}" ]] && merge+=(--limit-packs "${CACHE_LIMIT_PACKS}")
"${merge[@]}" | tee "${CACHE_ROOT}/merge.log"
echo "CACHE_MANIFEST=${CACHE_ROOT}/teacher_cache.jsonl"
