#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 GPU_ID SLOT TOTAL_SLOTS" >&2
  exit 2
fi

GPU_ID="$1"
SLOT="$2"
TOTAL_SLOTS="$3"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MANIFEST="${REPO_ROOT}/experiments/evaluation/whisper_attention_mask_v2/runs.tsv"
INDEX=0

while IFS=$'\t' read -r label relative_root; do
  [[ -n "${label}" ]] || continue
  if (( INDEX % TOTAL_SLOTS == SLOT )); then
    "${REPO_ROOT}/experiments/evaluation/whisper_attention_mask_v2/run_one.sh" \
      "${label}" "${REPO_ROOT}/${relative_root}" "${GPU_ID}"
  fi
  INDEX=$((INDEX + 1))
done < "${MANIFEST}"
