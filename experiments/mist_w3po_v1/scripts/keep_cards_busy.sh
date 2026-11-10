#!/usr/bin/env bash
# Overnight guard (user 2026-11-10 ~22:25 UTC: no card may sit idle, or a colleague may take the node's jobs away).
# Every POLL s, for each card: if the card runs NO process (nvidia-smi compute apps) on two consecutive checks AND no
# workload supervisor of this guard is alive for it, start the user's no-checkpoint workload on it exactly as
# start_workload_at.sh does (scripts/gpu_load/run_mist_workload.sh --devices k --wait-idle --idle-seconds 30). The
# supervisor loads only onto an idle card and stops by itself when another process appears (our experiments take
# precedence); the guard then restarts it once the card is idle again. Never kills anything. Stop the guard by its PID
# (guard.pid). Checked with bash -n only.
#   setsid nohup bash experiments/mist_w3po_v1/scripts/keep_cards_busy.sh > logs/mist_w3po_v1/keep_cards_busy_<UTC>.out 2>&1 < /dev/null &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
POLL=${POLL:-120}
CARDS=${CARDS:-"0 1 2 3 4 5 6 7"}
LOG="logs/mist_w3po_v1/keep_cards_busy_$(date -u +%Y%m%dT%H%M%SZ)"
if [ -e "$LOG" ]; then echo "refusing: $LOG exists"; exit 1; fi
mkdir -p "$LOG"; echo $$ > "$LOG/guard.pid"
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG/guard.log"; }
declare -A idle_once
log "guard $$ cards [$CARDS] poll ${POLL}s"
while true; do
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sort | uniq -c)
  for k in $CARDS; do
    uuid=$(nvidia-smi -i "$k" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d ' ')
    n=$(echo "$busy" | grep -c "$uuid" || true)
    sup=""; [ -e "$LOG/card$k.pid" ] && sup=$(cat "$LOG/card$k.pid")
    alive=0; [ -n "$sup" ] && [ -d "/proc/$sup" ] && alive=1
    if [ "$n" = 0 ] && [ "$alive" = 0 ]; then
      if [ "${idle_once[$k]:-0}" = 1 ]; then
        setsid -f bash -c "echo \$\$ > $LOG/card$k.pid; exec bash scripts/gpu_load/run_mist_workload.sh --devices $k --wait-idle --idle-seconds 30 --poll-seconds 5" \
          >> "$LOG/card$k.log" 2>&1 < /dev/null
        sleep 2
        log "card $k idle twice: workload supervisor started pid $(cat "$LOG/card$k.pid" 2>/dev/null)"
        idle_once[$k]=0
      else
        idle_once[$k]=1
      fi
    else
      idle_once[$k]=0
    fi
  done
  sleep "$POLL"
done
