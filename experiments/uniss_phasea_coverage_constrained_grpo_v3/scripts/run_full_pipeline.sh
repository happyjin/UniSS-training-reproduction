#!/usr/bin/env bash
set -euo pipefail

# Resumable, isolated v3 pipeline.  The post-R3 rollout is both a baseline
# audit and the first fresh trajectory pool; each subsequent update is followed
# by a new 64x4 rollout.  Every output ID is immutable, so a failed invocation
# can be diagnosed and resumed without overwriting a historical result.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${EXPERIMENT_ROOT}/../.." && pwd)
source "${EXPERIMENT_ROOT}/config.env"
PY=${PYTHON}
BASE_ADAPTER=${REPO_ROOT}/checkpoints/uniss_phasea_event_constrained_grpo_long_v2/event_grpo_round3_g4_w64_formal_v1
BASE_ROLLOUT=post_round3_checkpoint_g4_w64_v1

wait_for_merge() {
  local run_id=$1
  local path=${REPO_ROOT}/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/${run_id}/ROLLOUT_MERGED.json
  while [[ ! -f ${path} ]]; do
    sleep 15
  done
  echo "READY rollout=${run_id}"
}

run_train() {
  local run_id=$1 pack_id=$2 load_root=$3
  local root=${REPO_ROOT}/checkpoints/uniss_phasea_coverage_constrained_grpo_v3/${run_id}
  if [[ -f ${root}/latest_checkpointed_iteration.txt ]]; then
    echo "SKIP existing checkpoint=${run_id}"
    return
  fi
  EVENT_SMOKE=1 TRAIN_ITERS=2 bash "${SCRIPT_DIR}/run_event_grpo_8gpu.sh" \
    "${run_id}_smoke2" "${pack_id}" "${load_root}"
  TRAIN_ITERS=142 bash "${SCRIPT_DIR}/run_event_grpo_8gpu.sh" \
    "${run_id}" "${pack_id}" "${load_root}"
}

run_rollout() {
  local run_id=$1 adapter=$2 round=$3
  local merged=${REPO_ROOT}/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/${run_id}/ROLLOUT_MERGED.json
  if [[ -f ${merged} ]]; then
    echo "SKIP existing rollout=${run_id}"
    return
  fi
  ROLLOUT_WORKERS=64 ROLLOUT_GROUP_SIZE=4 bash "${SCRIPT_DIR}/run_fresh_rollout_8gpu.sh" \
    "${run_id}" "${adapter}" "${round}"
}

pack() {
  local rollout=$1 pack_id=$2
  local audit=${REPO_ROOT}/data/processed/uniss_phasea_coverage_constrained_grpo_v3/${pack_id}/AUDIT.json
  if [[ -f ${audit} ]]; then
    echo "SKIP existing pack=${pack_id}"
    return
  fi
  bash "${SCRIPT_DIR}/pack_fresh_round.sh" "${rollout}" "${pack_id}"
}

wait_for_merge "${BASE_ROLLOUT}"
pack "${BASE_ROLLOUT}" coverage_round1_pack_v1
run_train coverage_grpo_round1_v1 coverage_round1_pack_v1 "${BASE_ADAPTER}"

run_rollout coverage_rollout_round2_v1 "${REPO_ROOT}/checkpoints/uniss_phasea_coverage_constrained_grpo_v3/coverage_grpo_round1_v1/iter_0000142" 2
pack coverage_rollout_round2_v1 coverage_round2_pack_v1
run_train coverage_grpo_round2_v1 coverage_round2_pack_v1 "${REPO_ROOT}/checkpoints/uniss_phasea_coverage_constrained_grpo_v3/coverage_grpo_round1_v1"

run_rollout coverage_rollout_round3_v1 "${REPO_ROOT}/checkpoints/uniss_phasea_coverage_constrained_grpo_v3/coverage_grpo_round2_v1/iter_0000142" 3
pack coverage_rollout_round3_v1 coverage_round3_pack_v1
run_train coverage_grpo_round3_v1 coverage_round3_pack_v1 "${REPO_ROOT}/checkpoints/uniss_phasea_coverage_constrained_grpo_v3/coverage_grpo_round2_v1"

run_rollout coverage_final_post_round3_v1 "${REPO_ROOT}/checkpoints/uniss_phasea_coverage_constrained_grpo_v3/coverage_grpo_round3_v1/iter_0000142" 4
wait_for_merge coverage_final_post_round3_v1

REPORT=${REPO_ROOT}/reports/uniss_phasea_coverage_constrained_grpo_v3/final_v1
if [[ ! -e ${REPORT} ]]; then
  "${PY}" "${EXPERIMENT_ROOT}/evaluation/write_report.py" \
    --arm HistoricalStateful64x4="${REPO_ROOT}/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/formal_train64_g4_v1/ROLLOUT_MERGED.json" \
    --arm PostR3Baseline="${REPO_ROOT}/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/${BASE_ROLLOUT}/ROLLOUT_MERGED.json" \
    --arm CoverageRound2="${REPO_ROOT}/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round2_v1/ROLLOUT_MERGED.json" \
    --arm CoverageRound3="${REPO_ROOT}/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_rollout_round3_v1/ROLLOUT_MERGED.json" \
    --arm FinalPostRound3="${REPO_ROOT}/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3/coverage_final_post_round3_v1/ROLLOUT_MERGED.json" \
    --training-log CoverageRound1="${REPO_ROOT}/logs/uniss_phasea_coverage_constrained_grpo_v3/coverage_grpo_round1_v1.log" \
    --training-log CoverageRound2="${REPO_ROOT}/logs/uniss_phasea_coverage_constrained_grpo_v3/coverage_grpo_round2_v1.log" \
    --training-log CoverageRound3="${REPO_ROOT}/logs/uniss_phasea_coverage_constrained_grpo_v3/coverage_grpo_round3_v1.log" \
    --output-dir "${REPORT}"
fi
echo "PIPELINE_COMPLETE report=${REPORT}"
