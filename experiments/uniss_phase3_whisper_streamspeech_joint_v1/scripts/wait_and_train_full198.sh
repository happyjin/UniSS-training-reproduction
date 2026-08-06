#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_ROOT}/common.sh"

EXPECTED_SHARDS="${EXPECTED_SHARDS:-198}"
POLL_SECONDS="${POLL_SECONDS:-30}"
if (( EXPECTED_SHARDS <= 0 || POLL_SECONDS <= 0 )); then
  echo "EXPECTED_SHARDS and POLL_SECONDS must be positive" >&2
  exit 1
fi

echo "Waiting for ${EXPECTED_SHARDS} full198 Stage-A shard markers."
while true; do
  complete="$(find "${FORMAL_STAGE_A_ROOT}/parts" -mindepth 2 -maxdepth 2 \
    -name STAGE_A_SOURCE_PART_COMPLETE.json -print 2>/dev/null | wc -l)"
  echo "$(date -u +%FT%TZ) stage_a_complete=${complete}/${EXPECTED_SHARDS}"
  if (( complete == EXPECTED_SHARDS )); then
    break
  fi
  if (( complete > EXPECTED_SHARDS )); then
    echo "unexpected extra Stage-A completion markers: ${complete}" >&2
    exit 1
  fi
  if ! pgrep -f '[p]repare_full198_stage_a_worker.sh' >/dev/null; then
    echo "Stage-A workers exited before all shard markers were written" >&2
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done

echo "Stage-A complete; assembling and validating the full198 joint manifest."
bash "${SCRIPT_ROOT}/prepare_full198_joint_manifest.sh"

echo "Joint manifest complete; starting formal 8-GPU Megatron training."
exec bash "${SCRIPT_ROOT}/run_full198_8gpu.sh"
