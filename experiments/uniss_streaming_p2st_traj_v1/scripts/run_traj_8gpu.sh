#!/usr/bin/env bash
# One training run of the trajectory-supervision line, from C's iter_0004236.
#
# A sibling of uniss_streaming_p2st_pure_ce_v1/scripts/run_formal_8gpu.sh, not
# an edit to it.  That launcher hardcodes OWN_NAME to C's namespace, so reusing
# it directly would write this run's reports, log, checkpoints and tensorboard
# into C's directories.  Everything else is deliberately identical: the same
# experiment.env, the same geometry computation, and the same delegate --
# C's scripts/run_p2st_megatron.sh, invoked by absolute path so P2ST_DIR
# resolves to C's training entry point.  Step 1 changes the data pool and
# nothing else, which is what makes its result attributable.
#
# Differences from C's launcher, all of them intentional:
#   * OWN_NAME, so outputs land under this experiment;
#   * parent is C's iter_0004236 rather than B' iter_0001132;
#   * COVERAGE_EPOCHS defaults to 3.  The NIR-stratified pool holds 503,785
#     trajectories against C's 1,325,243 -- the composition is reached by
#     downsampling because build_p2st_pools rejects duplicate sequence_ids --
#     so three coverage epochs put the number of samples seen at 1.51M against
#     C's 1.33M.  That is 14% more, stated rather than hidden; the alternative
#     of two epochs would have been 24% fewer.
#   * SAVE_INTERVAL defaults to 100 so the step-100 and step-200 checkpoints
#     exist for the early external evaluations, with EVAL_INTERVAL left at 400.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BASE_EXPERIMENT=$(cd -- "${HERE}/../uniss_phase3_v4_e2e_simuls2st_pilot15_v1" && pwd)
C_EXPERIMENT=$(cd -- "${HERE}/../uniss_streaming_p2st_pure_ce_v1" && pwd)
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${BASE_EXPERIMENT}/experiment.env"
cd "${REPO_ROOT}"

: "${RUN_ID:?set a fresh immutable RUN_ID}"
: "${TRAIN_POOL_MANIFEST:?set TRAIN_POOL_MANIFEST}"
: "${VALID_POOL_MANIFEST:?set VALID_POOL_MANIFEST}"

OWN_NAME=uniss_streaming_p2st_traj_v1
PARENT_EXPERIMENT=${PARENT_EXPERIMENT:-uniss_streaming_p2st_pure_ce_v1}
PARENT_RUN_ID=${PARENT_RUN_ID:-p2st_epoch1_replay_20260902T170132Z}
PARENT_ITER=${PARENT_ITER:-0004236}
PARENT_SAVE_ROOT=${REPO_ROOT}/checkpoints/${PARENT_EXPERIMENT}/${PARENT_RUN_ID}
PARENT_CHECKPOINT=${PARENT_SAVE_ROOT}/iter_${PARENT_ITER}
# C's structural canary covers the same five families and the same pool schema,
# and the launcher only checks that it exists.  Referenced read-only.
CANARY_REPORT=${CANARY_REPORT:-${REPO_ROOT}/reports/uniss_streaming_p2st_pure_ce_v1/canaries/P2ST_STRUCTURAL_CANARY.json}

RUN_REPORT_ROOT=${REPO_ROOT}/reports/${OWN_NAME}/${RUN_ID}
RUN_LOG=${REPO_ROOT}/logs/${OWN_NAME}/${RUN_ID}.log
RUN_SAVE_DIR=${REPO_ROOT}/checkpoints/${OWN_NAME}/${RUN_ID}
RUN_TENSORBOARD_DIR=${REPO_ROOT}/runs/${OWN_NAME}/tensorboard/${RUN_ID}
RUN_FINGERPRINTS=${RUN_REPORT_ROOT}/PARENT_CHECKPOINT_FINGERPRINT.json
RUN_GEOMETRY=${RUN_REPORT_ROOT}/TRAINING_GEOMETRY.json

for path in "${TRAIN_POOL_MANIFEST}" "${VALID_POOL_MANIFEST}" "${CANARY_REPORT}" \
            "${PARENT_CHECKPOINT}/metadata.json" \
            "${C_EXPERIMENT}/scripts/run_p2st_megatron.sh"; do
  [[ -f "${path}" ]] || { echo "missing training input: ${path}" >&2; exit 3; }
done
for path in "${RUN_REPORT_ROOT}" "${RUN_LOG}" "${RUN_SAVE_DIR}" "${RUN_TENSORBOARD_DIR}"; do
  [[ ! -e "${path}" ]] || { echo "refusing to overwrite output: ${path}" >&2; exit 4; }
done
mkdir -p "${RUN_REPORT_ROOT}" "$(dirname -- "${RUN_LOG}")"

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.fingerprint \
  --checkpoint "parent=${PARENT_CHECKPOINT}" --workers 12 \
  --output "${RUN_FINGERPRINTS}" >/dev/null

# Same existence check the established launcher makes.  This pool carries zero
# teacher bindings and the run's KL denominators stay at zero throughout, so
# these audits gate a code path the run does not take.
TEACHER_RUN_ID=${TEACHER_RUN_ID:-teacher_cache_formal_p4_20260820T154500Z}
V1_TRAIN_AUDIT=${REPORT_ROOT}/v1_asr_teacher_cache/${TEACHER_RUN_ID}_v1_train/AUDIT.json
PHASE3_TRAIN_AUDIT=${REPORT_ROOT}/phase3_teacher_cache/${TEACHER_RUN_ID}_phase3_train/AUDIT.json
V1_VALID_AUDIT=${REPORT_ROOT}/v1_asr_teacher_cache/${TEACHER_RUN_ID}_v1_valid/AUDIT.json
PHASE3_VALID_AUDIT=${REPORT_ROOT}/phase3_teacher_cache/${TEACHER_RUN_ID}_phase3_valid/AUDIT.json
for path in "${V1_TRAIN_AUDIT}" "${PHASE3_TRAIN_AUDIT}" \
            "${V1_VALID_AUDIT}" "${PHASE3_VALID_AUDIT}"; do
  [[ -f "${path}" ]] || { echo "missing teacher audit: ${path}" >&2; exit 3; }
done

RUN_GBS=${RUN_GBS:-128}
RUN_SEED=${RUN_SEED:-20260819}
COVERAGE_EPOCHS=${COVERAGE_EPOCHS:-3}

PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" - \
  "${TRAIN_POOL_MANIFEST}" "${RUN_GBS}" "${COVERAGE_EPOCHS}" "${RUN_SEED}" \
  "${RUN_GEOMETRY}" <<'PY'
import json
import sys
from pathlib import Path

from experiments.uniss_streaming_p2st_pure_ce_v1.training.p2st_schedule import (
    POOL_WEIGHTS,
    family_blocks,
    required_total_blocks,
)

manifest = json.loads(Path(sys.argv[1]).read_text())
gbs, epochs, seed = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
output = Path(sys.argv[5])
rows = {family: int(value["rows"]) for family, value in manifest["families"].items()}
total = required_total_blocks(
    rows,
    global_batch_size=gbs,
    coverage_epochs=epochs,
    seed=seed,
    weights=POOL_WEIGHTS,
)
blocks = family_blocks(total, seed=seed, weights=POOL_WEIGHTS)
output.write_text(
    json.dumps(
        {
            "schema_version": "uniss_streaming_p2st_geometry_v1",
            "pool_manifest": str(Path(sys.argv[1]).resolve()),
            "global_batch_size": gbs,
            "coverage_epochs": epochs,
            "seed": seed,
            "family_rows": rows,
            "total_blocks": total,
            "train_iters": total,
            "warmup_updates": max(1, round(0.03 * total)),
            "family_blocks": {f: blocks.count(f) for f in POOL_WEIGHTS},
            "family_weights": dict(POOL_WEIGHTS),
        },
        indent=1,
        sort_keys=True,
    )
    + "\n"
)
print(f"total_blocks={total} warmup={max(1, round(0.03 * total))}")
PY

TRAIN_ITERS=${RUN_TRAIN_ITERS:-$("${PYTHON_BIN}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["train_iters"])' "${RUN_GEOMETRY}")}
WARMUP_ITERS=${RUN_WARMUP_ITERS:-$("${PYTHON_BIN}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["warmup_updates"])' "${RUN_GEOMETRY}")}
echo "train_iters=${TRAIN_ITERS} warmup=${WARMUP_ITERS}"

RUN_ID="${RUN_ID}" \
RUN_TRAIN_BUILD_REPORT="${TRAIN_POOL_MANIFEST}" \
RUN_VALID_BUILD_REPORT="${VALID_POOL_MANIFEST}" \
RUN_SAVE_DIR="${RUN_SAVE_DIR}" \
RUN_TENSORBOARD_DIR="${RUN_TENSORBOARD_DIR}" \
RUN_LOG="${RUN_LOG}" \
RUN_FINGERPRINTS="${RUN_FINGERPRINTS}" \
RUN_GEOMETRY="${RUN_GEOMETRY}" \
RUN_LOAD="${PARENT_SAVE_ROOT}" \
RUN_NPROC="${RUN_NPROC:-8}" \
RUN_MBS="${RUN_MBS:-2}" \
RUN_GBS="${RUN_GBS}" \
RUN_COVERAGE_EPOCHS="${COVERAGE_EPOCHS}" \
RUN_TRAIN_ITERS="${TRAIN_ITERS}" \
RUN_WARMUP_ITERS="${WARMUP_ITERS}" \
RUN_EXTENDED_CANARY=1 \
RUN_CANARY_REPORT="${CANARY_REPORT}" \
RUN_ALLOW_MISSING_TEACHERS=0 \
RUN_V1_TRAIN_CACHE_AUDIT="${V1_TRAIN_AUDIT}" \
RUN_PHASE3_TRAIN_CACHE_AUDIT="${PHASE3_TRAIN_AUDIT}" \
RUN_V1_VALID_CACHE_AUDIT="${V1_VALID_AUDIT}" \
RUN_PHASE3_VALID_CACHE_AUDIT="${PHASE3_VALID_AUDIT}" \
RUN_NUM_WORKERS="${NUM_WORKERS:-0}" \
RUN_SAVE_INTERVAL="${SAVE_INTERVAL:-100}" \
RUN_EVAL_INTERVAL="${EVAL_INTERVAL:-400}" \
RUN_EVAL_ITERS="${EVAL_ITERS:-8}" \
RUN_LOG_INTERVAL="${LOG_INTERVAL:-5}" \
RUN_SEED="${RUN_SEED}" \
RUN_MASTER_PORT="${MASTER_PORT:-29973}" \
  bash "${C_EXPERIMENT}/scripts/run_p2st_megatron.sh"

echo "log=${RUN_LOG}"
echo "save=${RUN_SAVE_DIR}"
