#!/usr/bin/env bash
# One coverage epoch of prefix-to-prefix training from B' iter_0001132.
#
# Mode is --e2e-extended-canary, not formal training, and the canary report it
# cites says formal_training_authorized: false.  That is accurate: C has
# produced no free-running gate result yet, so nothing has authorised it.  What
# the canary does record is the structural evidence -- the family GPU canary
# with zero terminal extensions and all three families' denominators firing,
# the 201/201 frontend prefix causality, and the cascade mechanics at 4/4.
#
# Pure cross-entropy.  Three prefix-to-prefix task families at 0.25 each plus
# the two phase3 replay families at 0.15/0.10 -- the interleaved schedule's own
# STEADY_WEIGHTS replay share, so the anti-forgetting pressure matches what
# this lineage has always used.  No teacher caches exist for this pool and
# none are needed: every weighted term the
# objective can compute other than asr_ce, mt_ce, semantic_ce and the
# boundary/EOS pair sees a zero denominator, which the family canary
# confirmed term by term.
#
# What this run has to move, registered before it starts:
#   * terminator rate on the TTS stage, 0.93 on the longest sample, to 1.00 --
#     END_SEMANTIC in the isolated single-task form;
#   * first_audible, currently equal to the source end on three of four
#     samples, well below it.  Until that moves this is a cascade, not
#     simultaneous translation.
#
# Nothing in the base experiment is modified; this sets RUN_* variables for
# the sibling launcher, which is the established launcher with the entry point
# and one weight changed.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BASE_EXPERIMENT=$(cd -- "${HERE}/../uniss_phase3_v4_e2e_simuls2st_pilot15_v1" && pwd)
# experiment.env derives every root from DATA_RUN_ID and defaults it to the
# smoke namespace, so it has to be set before sourcing, exactly as the
# established launchers do.
DATA_RUN_ID=${DATA_RUN_ID:-formal_gold_20260818T090515Z}
source "${BASE_EXPERIMENT}/experiment.env"
cd "${REPO_ROOT}"

: "${RUN_ID:?set a fresh immutable RUN_ID}"
: "${TRAIN_POOL_MANIFEST:?set TRAIN_POOL_MANIFEST}"
: "${VALID_POOL_MANIFEST:?set VALID_POOL_MANIFEST}"

OWN_NAME=uniss_streaming_p2st_pure_ce_v1
PARENT_EXPERIMENT=${PARENT_EXPERIMENT:-uniss_phase3_e2e_uniform_ce_v1}
PARENT_RUN_ID=${PARENT_RUN_ID:-uniform_ce_20260902T041721Z}
PARENT_ITER=${PARENT_ITER:-0001132}
PARENT_SAVE_ROOT=${REPO_ROOT}/checkpoints/${PARENT_EXPERIMENT}/${PARENT_RUN_ID}
PARENT_CHECKPOINT=${PARENT_SAVE_ROOT}/iter_${PARENT_ITER}
CANARY_REPORT=${CANARY_REPORT:-${REPO_ROOT}/reports/${OWN_NAME}/canaries/P2ST_STRUCTURAL_CANARY.json}

RUN_REPORT_ROOT=${REPO_ROOT}/reports/${OWN_NAME}/${RUN_ID}
RUN_LOG=${REPO_ROOT}/logs/${OWN_NAME}/${RUN_ID}.log
RUN_SAVE_DIR=${REPO_ROOT}/checkpoints/${OWN_NAME}/${RUN_ID}
RUN_TENSORBOARD_DIR=${REPO_ROOT}/runs/${OWN_NAME}/tensorboard/${RUN_ID}
RUN_FINGERPRINTS=${RUN_REPORT_ROOT}/PARENT_CHECKPOINT_FINGERPRINT.json
RUN_GEOMETRY=${RUN_REPORT_ROOT}/TRAINING_GEOMETRY.json

for path in "${TRAIN_POOL_MANIFEST}" "${VALID_POOL_MANIFEST}" "${CANARY_REPORT}" \
            "${PARENT_CHECKPOINT}/metadata.json"; do
  [[ -f "${path}" ]] || { echo "missing training input: ${path}" >&2; exit 3; }
done
for path in "${RUN_REPORT_ROOT}" "${RUN_LOG}" "${RUN_SAVE_DIR}" "${RUN_TENSORBOARD_DIR}"; do
  [[ ! -e "${path}" ]] || { echo "refusing to overwrite output: ${path}" >&2; exit 4; }
done
mkdir -p "${RUN_REPORT_ROOT}" "$(dirname -- "${RUN_LOG}")"

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.fingerprint \
  --checkpoint "v1=${PARENT_CHECKPOINT}" --workers 12 \
  --output "${RUN_FINGERPRINTS}" >/dev/null

# --e2e-allow-missing-teachers is gated to smoke mode, so an extended canary
# has to name existing teacher-cache audits.  This pool reads none of them:
# it carries zero teacher bindings, p2st_packed_task_to_runtime_item raises if
# a binding ever appears, _teacher_readers is only called from the base
# provider this entry point does not use, and the run's own
# denominator/v1_asr_kl and denominator/phase3_kl stay at zero throughout --
# which the family canary already showed term by term.  These paths therefore
# satisfy an existence check on a code path this run does not take.
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
COVERAGE_EPOCHS=${COVERAGE_EPOCHS:-1}

# compute_geometry reads the interleaved pool's schema, so the geometry comes
# from this experiment's own schedule.
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
RUN_SAVE_INTERVAL="${SAVE_INTERVAL:-400}" \
RUN_EVAL_INTERVAL="${EVAL_INTERVAL:-400}" \
RUN_EVAL_ITERS="${EVAL_ITERS:-8}" \
RUN_LOG_INTERVAL="${LOG_INTERVAL:-5}" \
RUN_SEED="${RUN_SEED}" \
RUN_MASTER_PORT="${MASTER_PORT:-29971}" \
  bash "${HERE}/scripts/run_p2st_megatron.sh"

echo "log=${RUN_LOG}"
echo "save=${RUN_SAVE_DIR}"
