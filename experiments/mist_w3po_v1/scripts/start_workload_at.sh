#!/usr/bin/env bash
# Start the user's no-checkpoint workload (scripts/gpu_load/run_mist_workload.sh) on each given card at a given UTC time.
# User request 2026-11-09 ~21:00 UTC: occupy the 8 cards from 10:30 SGT (= 02:30 UTC 2026-11-10), agreed with the colleague.
# Each card gets its own supervisor started with --wait-idle: it loads only once the card has no other process
# (it never stacks onto a foreign job), then fills memory up to the workload's own cap (>= 8 GiB free).
# Cards later used by our own experiments: stop that card's workload by the PID in $LOG/card<k>.pid first.
#   AT=2026-11-10T02:30:00Z CARDS="0 1 2 3 4 5 6 7" setsid nohup bash start_workload_at.sh > <log> 2>&1 < /dev/null &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
AT="${AT:?set AT, e.g. 2026-11-10T02:30:00Z}"
CARDS="${CARDS:-0 1 2 3 4 5 6 7}"
LOG="logs/mist_w3po_v1/workload_at_$(date -u +%Y%m%dT%H%M%SZ)"
if [ -e "$LOG" ]; then echo "refusing: $LOG exists"; exit 1; fi
mkdir -p "$LOG"
echo "$$" > "$LOG/scheduler.pid"
now=$(date -u +%s); t=$(date -u -d "$AT" +%s)
echo "$(date -u) scheduler $$: cards [$CARDS] at $AT (in $((t - now)) s); logs in $LOG"
if [ "$t" -gt "$now" ]; then sleep $((t - now)); fi
for k in $CARDS; do
  setsid -f bash -c "echo \$\$ > $LOG/card$k.pid; exec bash scripts/gpu_load/run_mist_workload.sh --devices $k --wait-idle --idle-seconds 30 --poll-seconds 5" \
    > "$LOG/card$k.log" 2>&1 < /dev/null
  sleep 10
  echo "$(date -u) card $k: supervisor pid $(cat "$LOG/card$k.pid" 2>/dev/null)"
done
echo "$(date -u) scheduler done"
