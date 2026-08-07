#!/usr/bin/env bash
set -euo pipefail

FULL198_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${FULL198_ROOT}/experiment.env"
# shellcheck source=/dev/null
source "${V6_EXPERIMENT_ROOT}/scripts/common.sh"

FULL198_TRAIN_MANIFEST="${FULL198_DATA_ROOT}/joint_train.jsonl"
FULL198_VALID_MANIFEST="${FULL198_DATA_ROOT}/joint_valid.jsonl"
FULL198_TOKENIZER_MAP_DIR="${FULL198_DATA_ROOT}/tokenizer_maps"
FULL198_DIRECTION_INDEX_DIR="${FULL198_DATA_ROOT}/direction_indices"

validate_full198_data() {
  require_file "${FULL198_TRAIN_MANIFEST}"
  require_file "${FULL198_VALID_MANIFEST}"
  require_file "${FULL198_DATA_ROOT}/manifest_summary.json"
  require_file "${FULL198_REPLAY_OFFSETS}"
  require_file "${FULL198_REPLAY_OFFSETS}.json"
  require_dir "${FULL198_TOKENIZER_MAP_DIR}"
  require_dir "${FULL198_DIRECTION_INDEX_DIR}"
  "${PYTHON}" - "${FULL198_DATA_ROOT}/manifest_summary.json" "${FULL198_REPLAY_OFFSETS}.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
replay = json.load(open(sys.argv[2]))
counts = manifest.get("counts", {})
if manifest.get("status") != "complete":
    raise SystemExit("full198 joint manifest is not complete")
if counts.get("written:train") != 19_286_004:
    raise SystemExit(f"unexpected full198 train count: {counts.get('written:train')}")
if counts.get("written:valid") != 7_965:
    raise SystemExit(f"unexpected full198 valid count: {counts.get('written:valid')}")
if not replay.get("complete") or replay.get("records") != 1_161_587:
    raise SystemExit("full198 Phase3 replay index is incomplete or unexpected")
PY
}

export_full198_inputs() {
  export TRAIN_MANIFEST="${FULL198_TRAIN_MANIFEST}"
  export VALID_MANIFEST="${FULL198_VALID_MANIFEST}"
  export TOKENIZER_MAP_DIR="${FULL198_TOKENIZER_MAP_DIR}"
  export DIRECTION_INDEX_DIR="${FULL198_DIRECTION_INDEX_DIR}"
  export REPLAY_OFFSETS="${FULL198_REPLAY_OFFSETS}"
  export BALANCE_VALIDATION=1
  export MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE NUM_WORKERS CUDA_VISIBLE_DEVICES
  export NPROC_PER_NODE MASTER_PORT
}
