#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; shift; fi
[[ $# -eq 0 ]] || { echo "Unknown argument: $1" >&2; exit 2; }

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${EXPERIMENT_DIR}/experiment.env"

PIPELINE_DIR="${RUN_DIR}/component_pipeline_8gpu"
QWEN_MARKER="${RUN_DIR}/qwen_pipeline_8gpu/QWEN_PIPELINE_COMPLETE"
PIPELINE_MARKER="${PIPELINE_DIR}/COMPONENT_PIPELINE_COMPLETE"

if [[ "${DRY_RUN}" == "1" ]]; then
  "${EXPERIMENT_DIR}/stage00_baselines/prepare_audio.sh" --dry-run
  "${EXPERIMENT_DIR}/stage00_baselines/run_prefix_baseline.sh" --dry-run
  "${EXPERIMENT_DIR}/stage01_02_streaming_student/run_token_8gpu.sh" --dry-run
  "${EXPERIMENT_DIR}/stage01_02_streaming_student/run_audio_8gpu.sh" --dry-run
  "${EXPERIMENT_DIR}/stage05_streaming_bicodec/run_overlap_baseline.sh" --dry-run
  "${EXPERIMENT_DIR}/stage05_streaming_bicodec/run_refinement_8gpu.sh" --dry-run
  "${EXPERIMENT_DIR}/stage07_grpo/run_8gpu.sh" --dry-run
  echo "stage08_nar_optional=profiling-gated-not-auto-started"
  exit 0
fi

[[ -f "${QWEN_MARKER}" ]] || { echo "Qwen pipeline incomplete: ${QWEN_MARKER}" >&2; exit 1; }
mkdir -p "${PIPELINE_DIR}" "${LOG_DIR}"

run_step() {
  local name="$1" guarded_path="$2" artifact="$3"
  shift 3
  local marker="${PIPELINE_DIR}/${name}.complete"
  if [[ -f "${marker}" ]]; then
    echo "Skipping completed ${name}"
    return 0
  fi
  if [[ -e "${guarded_path}" ]]; then
    echo "Refusing to overwrite partial ${name} output: ${guarded_path}" >&2
    return 1
  fi
  "$@"
  [[ -e "${artifact}" ]] || { echo "${name}: missing artifact ${artifact}" >&2; return 1; }
  printf 'completed_at=%s\nartifact=%s\n' "$(date -u +%FT%TZ)" "${artifact}" > "${marker}"
}

run_step stage00_audio "${STAGE0_AUDIO_DIR}" "${STAGE0_AUDIO_DIR}/audio_manifest.jsonl" \
  "${EXPERIMENT_DIR}/stage00_baselines/prepare_audio.sh"
run_step stage00_prefix "${STAGE0_PREFIX_DIR}" "${STAGE0_PREFIX_DIR}/record_$((STAGE0_PREFIX_RECORDS - 1)).json" \
  "${EXPERIMENT_DIR}/stage00_baselines/run_prefix_baseline.sh"
run_step stage01_02_token "${STAGE1_TOKEN_OUTPUT_DIR}" "${STAGE1_TOKEN_OUTPUT_DIR}/last.pt" \
  "${EXPERIMENT_DIR}/stage01_02_streaming_student/run_token_8gpu.sh"
run_step stage01_02_audio "${STAGE1_AUDIO_OUTPUT_DIR}" "${STAGE1_AUDIO_OUTPUT_DIR}/last.pt" \
  "${EXPERIMENT_DIR}/stage01_02_streaming_student/run_audio_8gpu.sh"
run_step stage05_overlap "${STAGE5_OUTPUT_DIR}" "${STAGE5_OUTPUT_DIR}/record_0_bicodec.wav" \
  "${EXPERIMENT_DIR}/stage05_streaming_bicodec/run_overlap_baseline.sh" --record-index 0
run_step stage05_refinement "${STAGE5_REFINEMENT_OUTPUT_DIR}" "${STAGE5_REFINEMENT_OUTPUT_DIR}/bicodec_streaming_refinement.pt" \
  "${EXPERIMENT_DIR}/stage05_streaming_bicodec/run_refinement_8gpu.sh"
run_step stage07_grpo "${STAGE7_OUTPUT_DIR}" "${STAGE7_OUTPUT_DIR}/policy_grpo.pt" \
  "${EXPERIMENT_DIR}/stage07_grpo/run_8gpu.sh"

printf 'completed_at=%s\nqwen_pipeline=%s\nstage8=profiling-gated\n' \
  "$(date -u +%FT%TZ)" "${QWEN_MARKER}" > "${PIPELINE_MARKER}"
echo "Eight-GPU component pipeline completed: ${PIPELINE_MARKER}"

