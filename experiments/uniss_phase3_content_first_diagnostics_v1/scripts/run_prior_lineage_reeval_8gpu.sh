#!/usr/bin/env bash
# Experiment 0-B: is the prior lineage really stronger, or is the gap an
# evaluator artefact?
#
# Stage 1 (no GPU) runs the established write_report evaluator once over the
# archived prior-lineage rollouts and the content-first rollouts together, so
# both lineages are scored by the same code in the same artefact.
#
# Stage 2 (8 GPU, opt-in via REEVAL=1) re-runs the prior lineage's best adapter
# with today's code on the same immutable 64-episode protocol, which tests
# reproducibility on top of comparability.
#
# The prior lineage's own rollout script is used unchanged; nothing in that
# experiment is modified.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HERE}/config.env"

PRIOR_ROOT=${REPO_ROOT}/eval_outputs/uniss_phasea_coverage_constrained_grpo_v3
CONTENT_ROOT=${REPO_ROOT}/eval_outputs/uniss_phase3_content_first_joint_s2st_v1
STATEFUL_ROOT=${REPO_ROOT}/eval_outputs/uniss_phasea_stateful_longepisode_rl_v1
OUTPUT_DIR=${OUTPUT_DIR:-${REPORT_ROOT}/prior_lineage_reeval}
PRIOR_ADAPTER=${PRIOR_ADAPTER:-${REPO_ROOT}/checkpoints/uniss_phasea_event_constrained_grpo_long_v2/event_grpo_round3_g4_w64_formal_v1/iter_0000142}
REEVAL_RUN_ID=${REEVAL_RUN_ID:-prior_lineage_reeval_g4_w64_v1}
LOG=${LOG:-${LOG_ROOT}/prior_lineage_reeval.log}

ARMS=(
  "prior_stateful_archived=${STATEFUL_ROOT}/formal_train64_g4_v1/ROLLOUT_MERGED.json"
  "prior_post_round3_archived=${PRIOR_ROOT}/post_round3_checkpoint_g4_w64_v1/ROLLOUT_MERGED.json"
  "prior_final_round3_archived=${PRIOR_ROOT}/coverage_final_post_round3_v1/ROLLOUT_MERGED.json"
  "content_first_pre_grpo=${CONTENT_ROOT}/content_first_pre_grpo_g4_w64_v2/ROLLOUT_MERGED.json"
  "content_first_round1=${CONTENT_ROOT}/content_first_post_round1_g4_w64_v2/ROLLOUT_MERGED.json"
  "content_first_round2=${CONTENT_ROOT}/content_first_post_round2_g4_w64_v2/ROLLOUT_MERGED.json"
)

mkdir -p "$(dirname "${LOG}")"

if [[ ${REEVAL:-0} == 1 ]]; then
  [[ -f "${PRIOR_ADAPTER}/.metadata" ]] || {
    echo "missing prior adapter: ${PRIOR_ADAPTER}" >&2; exit 2; }
  REEVAL_OUTPUT=${PRIOR_ROOT}/${REEVAL_RUN_ID}
  [[ ! -e "${REEVAL_OUTPUT}" ]] || {
    echo "refusing to overwrite ${REEVAL_OUTPUT}" >&2; exit 3; }
  echo "stage 2: re-running the prior lineage adapter with today's code"
  ROLLOUT_WORKERS=64 ROLLOUT_GROUP_SIZE=4 \
    bash "${REPO_ROOT}/experiments/uniss_phasea_coverage_constrained_grpo_v3/scripts/run_fresh_rollout_8gpu.sh" \
    "${REEVAL_RUN_ID}" "${PRIOR_ADAPTER}" 0 2>&1 | tee -a "${LOG}"
  ARMS+=("prior_post_round3_reeval=${REEVAL_OUTPUT}/ROLLOUT_MERGED.json")
fi

for arm in "${ARMS[@]}"; do
  path=${arm#*=}
  [[ -f "${path}" ]] || { echo "missing rollout arm: ${path}" >&2; exit 2; }
done
# write_report creates OUTPUT_DIR itself and refuses to overwrite an existing
# one, so it must not be pre-created here.
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "refusing to overwrite ${OUTPUT_DIR}" >&2; exit 3; }
mkdir -p "$(dirname "${OUTPUT_DIR}")"

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
args=()
for arm in "${ARMS[@]}"; do args+=(--arm "${arm}"); done
"${PYTHON}" -m experiments.uniss_phasea_coverage_constrained_grpo_v3.evaluation.write_report \
  "${args[@]}" --output-dir "${OUTPUT_DIR}" 2>&1 | tee -a "${LOG}"

echo "OUTPUT_DIR=${OUTPUT_DIR}"
