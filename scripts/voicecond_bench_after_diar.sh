#!/usr/bin/env bash
# OpenBroadcast-SI v1: for each given <cond>:<dir>, wait until its Nemotron lowlat posteriors are complete (one
# post.pt per meetings.json entry, or the diar FAILED marker), build its C2 + chunk-protocol windows (bench_build
# windows) and start its baseline-B chain (voicecond_bench_run_b.sh with CHAINS=<cond>:<dir>, detached, own log).
# Sets are handled in the given order; a failed set is logged and skipped. Refuses nothing itself (the steps do).
#   [DRY_RUN=1] bash scripts/voicecond_bench_after_diar.sh <cond>:<dir> ...
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond_bench/after_diar.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
VL=logs/mist_w3po_v1/voicecond_bench
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
complete() {   # <data dir>: all posteriors present
  local n m
  m=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" $1/meetings.json)
  n=$(ls $1/diar/nemotron_lowlat/*.post.pt 2>/dev/null | wc -l)
  [ "$n" = "$m" ]
}
for x in "$@"; do
  c=${x%%:*}; d=${x##*:}; D=data/open_voicecond_v1/bench/$c/$d
  if [ -n "${DRY_RUN:-}" ]; then log "[dry] wait $D posteriors -> windows --cond $c --dir $d -> CHAINS=$c:$d run_b"; continue; fi
  until complete $D || ls $VL/DIAR_*_FAILED > /dev/null 2>&1; do sleep 30; done
  complete $D || { log "posteriors incomplete for $c:$d (diar failed), skipped"; continue; }
  PYTHONPATH=. $PY -m experiments.mist_w3po_v1.voicecond.bench_build windows --cond $c --dir $d > $VL/windows_${c}_$d.log 2>&1 \
    || { log "windows failed for $c:$d"; continue; }
  CHAINS="$c:$d" setsid nohup bash scripts/voicecond_bench_run_b.sh > $VL/run_b_${c}_$d.log 2>&1 < /dev/null 9>&- &
  log "$c:$d windows built, chain started (pid $!)"
  sleep 60
done
log "AFTER_DIAR_EXIT"
