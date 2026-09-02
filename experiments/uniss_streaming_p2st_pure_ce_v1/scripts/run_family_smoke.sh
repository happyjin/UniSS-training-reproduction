#!/usr/bin/env bash
# One-update GPU canary for a single prefix-to-prefix family.
#
# This is the acceptance test for the data path, not a training run.  A smoke
# is capped at one or two optimizer updates and one family owns a whole global
# batch, so the family is named explicitly; run this once per family to cover
# all three.  What it proves, in order of importance:
#
#   1. StageAObjective._inject_causal_glm does not raise.  It requires the
#      frontend's token count for each row's waveform to equal glm_lengths,
#      which is exactly what the audio cut and the closed-form count exist to
#      satisfy, and it is the one thing no CPU test can establish.
#   2. validate_family_denominators passes, i.e. the family's own losses
#      actually fired rather than the step merely not crashing.
#   3. diagnostic/causal_glm_agreement is present, so the acoustic path ran.
#
# Nothing in the base experiment is modified; this only sets RUN_* variables
# for the sibling launcher, which is the established launcher with the entry
# point and one weight changed.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BASE_EXPERIMENT=$(cd -- "${HERE}/../uniss_phase3_v4_e2e_simuls2st_pilot15_v1" && pwd)
source "${BASE_EXPERIMENT}/experiment.env"
cd "${REPO_ROOT}"

: "${RUN_ID:?set a fresh immutable RUN_ID}"
: "${SMOKE_FAMILY:?set SMOKE_FAMILY to one p2st family}"
: "${POOL_MANIFEST:?set POOL_MANIFEST to a p2st POOL_MANIFEST.json}"

OWN_NAME=uniss_streaming_p2st_pure_ce_v1
PARENT_RUN_ID=${PARENT_RUN_ID:-uniform_ce_20260902T041721Z}
PARENT_EXPERIMENT=${PARENT_EXPERIMENT:-uniss_phase3_e2e_uniform_ce_v1}
PARENT_SAVE_ROOT=${REPO_ROOT}/checkpoints/${PARENT_EXPERIMENT}/${PARENT_RUN_ID}
PARENT_ITER=${PARENT_ITER:-0001132}
PARENT_CHECKPOINT=${PARENT_SAVE_ROOT}/iter_${PARENT_ITER}

RUN_REPORT_ROOT=${REPO_ROOT}/reports/${OWN_NAME}/${RUN_ID}
RUN_LOG=${REPO_ROOT}/logs/${OWN_NAME}/${RUN_ID}.log
RUN_SAVE_DIR=${REPO_ROOT}/checkpoints/${OWN_NAME}/${RUN_ID}
RUN_TENSORBOARD_DIR=${REPO_ROOT}/runs/${OWN_NAME}/tensorboard/${RUN_ID}
RUN_FINGERPRINTS=${RUN_REPORT_ROOT}/PARENT_CHECKPOINT_FINGERPRINT.json

for path in "${POOL_MANIFEST}" "${PARENT_CHECKPOINT}/metadata.json"; do
  [[ -f "${path}" ]] || { echo "missing smoke input: ${path}" >&2; exit 3; }
done
for path in "${RUN_REPORT_ROOT}" "${RUN_LOG}" "${RUN_SAVE_DIR}" "${RUN_TENSORBOARD_DIR}"; do
  [[ ! -e "${path}" ]] || { echo "refusing to overwrite output: ${path}" >&2; exit 4; }
done
mkdir -p "${RUN_REPORT_ROOT}" "$(dirname -- "${RUN_LOG}")"

"${PYTHON_BIN}" -m \
  experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.fingerprint \
  --checkpoint "v1=${PARENT_CHECKPOINT}" --workers 12 \
  --output "${RUN_FINGERPRINTS}" >/dev/null

# compute_geometry parses the interleaved pool's BUILD_COMPLETE schema, so the
# geometry comes from this experiment's own schedule instead.  A smoke reads
# neither field -- train_iters and warmup are set explicitly below -- but the
# file has to exist and it is the right place to record the block allocation.
RUN_GEOMETRY=${RUN_REPORT_ROOT}/TRAINING_GEOMETRY.json
PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" - \
  "${POOL_MANIFEST}" "${RUN_GBS:-128}" "${RUN_COVERAGE_EPOCHS:-1}" \
  "${RUN_SEED:-20260819}" "${RUN_GEOMETRY}" <<'PY'
import json
import sys
from pathlib import Path

from experiments.uniss_streaming_p2st_pure_ce_v1.training.p2st_schedule import (
    UNIFORM_WEIGHTS,
    family_blocks,
    required_total_blocks,
)

manifest = json.loads(Path(sys.argv[1]).read_text())
gbs, epochs, seed, output = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
rows = {family: int(value["rows"]) for family, value in manifest["families"].items()}
total = required_total_blocks(
    rows, global_batch_size=gbs, coverage_epochs=epochs, seed=seed
)
blocks = family_blocks(total, seed=seed)
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
            "warmup_updates": 0,
            "family_blocks": {f: blocks.count(f) for f in UNIFORM_WEIGHTS},
            "family_weights": dict(UNIFORM_WEIGHTS),
        },
        indent=1,
        sort_keys=True,
    )
    + "\n"
)
print(f"total_blocks={total}")
PY

# --load takes the save root; the trainer resolves the iteration itself.
RUN_ID="${RUN_ID}" \
RUN_TRAIN_BUILD_REPORT="${POOL_MANIFEST}" \
RUN_SAVE_DIR="${RUN_SAVE_DIR}" \
RUN_TENSORBOARD_DIR="${RUN_TENSORBOARD_DIR}" \
RUN_LOG="${RUN_LOG}" \
RUN_FINGERPRINTS="${RUN_FINGERPRINTS}" \
RUN_GEOMETRY="${RUN_GEOMETRY}" \
RUN_LOAD="${PARENT_SAVE_ROOT}" \
RUN_NPROC="${RUN_NPROC:-2}" \
RUN_MBS="${RUN_MBS:-2}" \
RUN_GBS="${RUN_GBS:-128}" \
RUN_COVERAGE_EPOCHS="${RUN_COVERAGE_EPOCHS:-1}" \
RUN_TRAIN_ITERS="${RUN_TRAIN_ITERS:-1}" \
RUN_SMOKE=1 \
RUN_SMOKE_FAMILY="${SMOKE_FAMILY}" \
RUN_ALLOW_MISSING_TEACHERS=1 \
RUN_NUM_WORKERS="${NUM_WORKERS:-0}" \
RUN_SAVE_INTERVAL=1000 \
RUN_EVAL_INTERVAL=1000 \
RUN_EVAL_ITERS=0 \
RUN_LOG_INTERVAL=1 \
RUN_SEED="${RUN_SEED:-20260819}" \
RUN_MASTER_PORT="${MASTER_PORT:-29951}" \
  bash "${HERE}/scripts/run_p2st_megatron.sh"

echo "smoke_family=${SMOKE_FAMILY}"
echo "log=${RUN_LOG}"
