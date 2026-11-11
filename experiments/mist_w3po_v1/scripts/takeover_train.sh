#!/usr/bin/env bash
# Take a 4-card slot from OUR placeholders and start a training in the same breath (no zero-process gap beyond the
# training's own CUDA start). Waits (event poll every 10 s) for WAIT_FILE if given. For every card in CARDS: the
# train_mist_workload.py supervisors whose cmdline has "--devices <card>" (ppid 1) are stopped first (SIGTERM), then
# their worker children and any orphan worker with the same --devices; gpu_keeper.py <dir> <card> is stopped too.
# Processes that are not ours (another user's / container) are never touched: only cmdlines matching these two
# scripts under /opt/dlami/nvme/jasonleeeli/projects/UniSS. Then LAUNCH (a command string) runs immediately.
#   CARDS=4,5,6,7 LAUNCH='CARDS=4,5,6,7 RUN_ID=... bash experiments/mist_w3po_v1/scripts/run_crf_v2p.sh' \
#     [WAIT_FILE=<path>] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/takeover_train.sh
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
CARDS=${CARDS:?}
LAUNCH=${LAUNCH:?}
log() { echo "$(date -u +%FT%TZ) $*"; }
if [ -n "${WAIT_FILE:-}" ]; then
  until [ -e "$WAIT_FILE" ]; do sleep 10; done
  log "found $WAIT_FILE"
fi
sup=(); wrk=()
for c in ${CARDS//,/ }; do
  for p in $(pgrep -f "train_mist_workload.py --devices $c( |$)" 2>/dev/null); do
    cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null) || continue
    case "$cmd" in *"/opt/dlami/nvme/jasonleeeli/projects/UniSS/scripts/gpu_load/train_mist_workload.py --devices $c "*) ;; *) continue;; esac
    if [ "$(ps -o ppid= -p $p | tr -d ' ')" = 1 ]; then sup+=("$p"); else wrk+=("$p"); fi
  done
  for p in $(pgrep -f "gpu_keeper.py .* $c$" 2>/dev/null); do sup+=("$p"); done
done
for s in "${sup[@]}"; do for p in $(pgrep -P "$s" 2>/dev/null); do wrk+=("$p"); done; done
wrk=($(printf '%s\n' "${wrk[@]}" | sort -u))
log "cards $CARDS: supervisors/keepers ${sup[*]:-none}; workers ${wrk[*]:-none}"
if [ -n "${DRY_RUN:-}" ]; then echo "LAUNCH: $LAUNCH"; exit 0; fi
for p in "${sup[@]}"; do kill "$p" 2>/dev/null && log "stopped $p"; done
sleep 2
for p in "${wrk[@]}"; do kill "$p" 2>/dev/null && log "stopped worker $p"; done
for i in $(seq 1 30); do
  busy=0
  for p in "${sup[@]}" "${wrk[@]}"; do kill -0 "$p" 2>/dev/null && busy=1; done
  [ $busy = 0 ] && break
  sleep 1
done
log "launching"
bash -c "$LAUNCH"
