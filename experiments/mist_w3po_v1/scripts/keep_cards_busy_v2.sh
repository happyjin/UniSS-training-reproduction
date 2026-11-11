#!/usr/bin/env bash
# Guard v2 (user 2026-11-11 ~09:55 UTC: every card must always be held by the no-checkpoint training placeholder).
# Differences from keep_cards_busy.sh (kept unchanged):
#   * a card counts as free when none of OUR processes is on it: processes in Kubernetes pod cgroups (another user's
#     pods) are ignored, so a colleague's job cannot keep the card from being held;
#   * never starts a second placeholder: skips a card while any train_mist_workload.py supervisor for that card is
#     alive (v1 started duplicates while a placeholder was still loading, and the two then stopped each other);
#   * the placeholder runs with --ignore-k8s-pods --tolerate-keeper --smi-retries 6 --poll-seconds 10;
#   * POLL 30 s, one free check is enough.
# Never kills anything. Stop it by its PID (guard.pid). Checked with bash -n only.
#   setsid nohup bash experiments/mist_w3po_v1/scripts/keep_cards_busy_v2.sh > logs/mist_w3po_v1/keep_cards_busy_v2_<UTC>.out 2>&1 < /dev/null &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
POLL=${POLL:-30}
CARDS=${CARDS:-"0 1 2 3 4 5 6 7"}
LOG="logs/mist_w3po_v1/keep_cards_busy_v2_$(date -u +%Y%m%dT%H%M%SZ)"
if [ -e "$LOG" ]; then echo "refusing: $LOG exists"; exit 1; fi
mkdir -p "$LOG"; echo $$ > "$LOG/guard.pid"
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG/guard.log"; }
log "guard v2 $$ cards [$CARDS] poll ${POLL}s"
supervisor_for() {   # prints 1 if a train_mist_workload.py supervisor (not a --worker) for card $1 is alive
  ps -eo args= | grep "train_mist_workload.py" | grep -v -- "--worker" | grep -v grep \
    | awk -v k="$1" '{for(i=1;i<NF;i++) if($i=="--devices" && $(i+1)==k) {print 1; exit}}' | head -1
}
ours_on() {   # number of non-k8s-pod compute processes on card $1
  local n=0
  for p in $(timeout 30 nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    grep -q kubepods "/proc/$p/cgroup" 2>/dev/null || n=$((n + 1))
  done
  echo $n
}
while true; do
  for k in $CARDS; do
    [ -n "$(supervisor_for "$k")" ] && continue
    [ "$(ours_on "$k")" != 0 ] && continue
    setsid -f bash -c "echo \$\$ > $LOG/card$k.pid; exec bash scripts/gpu_load/run_mist_workload.sh --devices $k --wait-idle --ignore-k8s-pods --tolerate-keeper --smi-retries 6 --idle-seconds 5 --poll-seconds 10 --idle-memory-mib 100000" \
      >> "$LOG/card$k.log" 2>&1 < /dev/null
    sleep 2
    log "card $k free of our processes and without a placeholder: started supervisor $(cat "$LOG/card$k.pid" 2>/dev/null)"
  done
  sleep "$POLL"
done
