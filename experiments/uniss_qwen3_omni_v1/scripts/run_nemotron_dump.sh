#!/usr/bin/env bash
# Step D2 of docs/multi_speaker_si_plan_v7.zh-CN.md: Nemotron-3 posteriors + RTTM for all 171 AMI IHM-mix sessions,
# the offline preset and the 1.04 s streaming preset, one process per card (GPUS, default all eight), sessions
# split across them. Then, if HOLDER is set, the GPU holder on every card (the cards are idle until the next job).
# Launch detached (setsid nohup ... &); DRY_RUN=1 prints the commands.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO_ROOT}"
PY=/opt/dlami/nvme/jasonleeeli/venvs/nemo_diar/bin/python
read -r -a G <<<"${GPUS:-0 1 2 3 4 5 6 7}"
LOGS="${REPO_ROOT}/logs/multispk"; mkdir -p "${LOGS}"
log() { echo "[$(TZ=Europe/Prague date +%H:%M:%S)] $*"; }
for preset in ${PRESETS:-offline lowlat}; do
  log "preset ${preset} on cards ${G[*]}"
  for k in "${!G[@]}"; do
    cmd=(env CUDA_VISIBLE_DEVICES="${G[$k]}" "${PY}" experiments/uniss_qwen3_omni_v1/data_clean/dump_nemotron_posteriors.py
         --preset "${preset}" --shard "${k}" --num-shards "${#G[@]}")
    if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${cmd[@]}"; echo; continue; fi
    "${cmd[@]}" > "${LOGS}/nemotron_${preset}_shard${k}.log" 2>&1 &
  done
  wait
  n=$(ls data/ami_multispk_si_v1/diar/nemotron_${preset}/*.post.pt 2>/dev/null | wc -l)
  log "preset ${preset}: ${n} sessions written"
done
[ -n "${DRY_RUN:-}" ] && exit 0
log "NEMOTRON_DUMP_DONE"
if [ -n "${HOLDER:-}" ]; then bash "${HOLDER}" 0,1,2,3,4,5,6,7; fi
