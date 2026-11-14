#!/usr/bin/env bash
# Model-free GPU watch (user 2026-11-14 ~22:10 Prague: replaces the two-hourly LLM GPU check to save tokens).
# Every TICK s (default 300), read-only: per card used memory and compute processes. Writes one ALERT line to
# logs/mist_w3po_v1/gpu_watch/alerts.log (Prague time) only when something is wrong; a status line every 2 h goes to
# status.log. Never starts, stops or touches any process.
#   ALERT idle      card with no compute process, or only processes < 8 GiB, on two consecutive ticks (> 5 min)
#   ALERT foreign   a compute process whose owner is not root, or whose owner cannot be resolved (other namespace)
#   ALERT guard     keep_cards_busy_v3.sh (GUARD_PID, default 2726951) not alive
#   ALERT mem       card used memory above 135 GiB (138240 MiB)
# Same alert key is repeated at most once per hour.
#   [TICK=300] [GUARD_PID=2726951] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/gpu_watch_shell.sh
# DRY_RUN=1: one tick, print what would be written, write nothing. Run detached:
#   setsid nohup bash experiments/mist_w3po_v1/scripts/gpu_watch_shell.sh > logs/mist_w3po_v1/gpu_watch/run.log 2>&1 < /dev/null &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
TICK=${TICK:-300}; GUARD_PID=${GUARD_PID:-2726951}
D=logs/mist_w3po_v1/gpu_watch
mkdir -p $D
declare -A IDLE_PREV LAST_ALERT FOREIGN_PREV FOREIGN_NOW
now() { TZ=Europe/Prague date '+%F %H:%M:%S %Z'; }
emit() {   # <key> <message>
  local k=$1 t; t=$(date +%s); shift
  if [ -n "${DRY_RUN:-}" ]; then echo "ALERT $k: $*"; return; fi
  [ -n "${LAST_ALERT[$k]:-}" ] && [ $((t - LAST_ALERT[$k])) -lt 3600 ] && return
  LAST_ALERT[$k]=$t
  echo "$(now) ALERT $k: $*" >> $D/alerts.log
}
tick() {
  local q a idx uuid used small pid mem user
  q=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>/dev/null) || { emit smi "nvidia-smi failed"; return; }
  a=$(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null)
  ps -p $GUARD_PID > /dev/null 2>&1 || emit guard "guard PID $GUARD_PID not alive"
  local status=""
  FOREIGN_NOW=()
  while IFS=', ' read -r idx uuid used; do
    [ -z "$idx" ] && continue
    status="$status $idx:${used}M"
    [ "$used" -gt 138240 ] && emit mem$idx "card $idx uses $used MiB (> 135 GiB)"
    local n=0 big=0
    while IFS=', ' read -r pid u2 mem; do
      [ "$u2" = "$uuid" ] || continue
      n=$((n + 1)); [ "${mem:-0}" -ge 8192 ] && big=1
      user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
      if [ "$user" != root ]; then   # report only when seen on two consecutive ticks (short-lived processes race the query)
        [ -n "${FOREIGN_PREV[$pid]:-}" ] && emit foreign$pid "card $idx pid $pid owner '${user:-unresolved}' ${mem} MiB, seen twice"
        FOREIGN_NOW[$pid]=1
      fi
    done <<< "$a"
    if [ $n -eq 0 ] || [ $big -eq 0 ]; then
      [ -n "${IDLE_PREV[$idx]:-}" ] && emit idle$idx "card $idx idle (${n} procs, none >= 8 GiB, used $used MiB) for > ${TICK}s"
      IDLE_PREV[$idx]=1
    else
      IDLE_PREV[$idx]=""
    fi
  done <<< "$q"
  FOREIGN_PREV=(); for pid in "${!FOREIGN_NOW[@]}"; do FOREIGN_PREV[$pid]=1; done
  if [ -n "${DRY_RUN:-}" ]; then echo "status:$status (unresolved/non-root pids this tick: ${!FOREIGN_NOW[*]})"; return; fi
  [ $(( $(date +%s) / 60 % 120 )) -lt $((TICK / 60)) ] && echo "$(now) status:$status" >> $D/status.log
}
if [ -n "${DRY_RUN:-}" ]; then tick; exit 0; fi
echo "$(now) gpu_watch_shell start pid $$ tick ${TICK}s guard $GUARD_PID" >> $D/status.log
while true; do tick; sleep $TICK; done
