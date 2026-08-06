#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"

LANE="${1:?lane 0..7 is required}"
for ((shard=LANE; shard<198; shard+=8)); do
  name="$(printf 'train-%05d' "${shard}")"
  input="${UNIST_RAW_ROOT}/${name}.parquet"
  output="${FORMAL_STAGE_A_ROOT}/parts/${name}"
  require_file "${input}"
  if [[ -f "${output}/STAGE_A_SOURCE_PART_COMPLETE.json" ]]; then
    echo "already complete: ${name}"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${LANE}" "${PYTHON}" -m training.simul_uniss.subsecond_v1.stage_a prepare-part \
    --input "${input}" \
    --output-dir "${output}" \
    --bicodec-checkpoint "${BICODEC_CHECKPOINT}" \
    --device cuda:0 \
    --side source \
    --batch-size 128 \
    --io-workers 8 \
    --progress-interval 10000 \
    --skip-sha256
done
