#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env"
mkdir -p "${PACK_PARTS_ROOT}" "${PACKED_ROOT}" "${LOG_ROOT}/pack"

printf '%s\n' $(seq 0 14) | xargs -P "${PACK_WORKERS:-8}" -I{} bash -c '
  shard="$1"; python="$2"; repo="$3"; cache="$4"; raw="$5"; parts="$6"; seq="$7"; logs="$8"
  part=$(printf "part-%03d" "$shard")
  mkdir -p "$parts/$part"
  "$python" -m experiments.uniss_true_subsecond_pilot15_epoch1_v2.data.pack_cache \
    --cache-part "$cache/$part" \
    --raw-parquet "$raw/train-$(printf "%05d" "$shard").parquet" \
    --output "$parts/$part/packed_trajectory.jsonl" \
    --marker "$parts/$part/PACK_COMPLETE.json" \
    --seq-length "$seq" > "$logs/$part.log" 2>&1
' _ {} "${PYTHON}" "${REPO_ROOT}" "${CACHE_ROOT}" "${RAW_UNIST_DIR}" "${PACK_PARTS_ROOT}" "${SEQ_LENGTH}" "${LOG_ROOT}/pack"

"${PYTHON}" -m experiments.uniss_true_subsecond_pilot15_epoch1_v2.data.assemble \
  --parts-root "${PACK_PARTS_ROOT}" \
  --output-root "${PACKED_ROOT}" \
  --shard-count 15 | tee "${LOG_ROOT}/pack/assembly.log"
