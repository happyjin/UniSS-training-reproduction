#!/usr/bin/env bash
# D1 training (reports/mist_w3po_v1/thinker_dagger_d1_prereg_v1.zh-CN.md §1.4): the base's recipe and data (its sched,
# lattice mode and DART manifest from logs/mist_w3po_v1/d1/base_decision.json) + the learner roll-in's rows (--d1
# <manifest of start_d1_rollin.sh>), 3000 steps, 4 cards (run_crf_v2p.sh). Waits for the manifest; R4 false (fewer than
# 5 % of the plan windows ever overdue) -> not launched. Then waits until the slot of CARDS is free (no slot owner,
# every compute process on those cards one of our placeholders), claims it (noclobber) and takes it over with
# takeover_train.sh (placeholders stopped by PID, launch in the same breath), and checks the 4 ranks.
#   RUN_ID=d1_<base>_<UTC> CARDS=0,1,2,3|4,5,6,7 [PORT=29985] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/start_d1.sh
# Run detached: setsid nohup bash .../start_d1.sh > logs/mist_w3po_v1/d1/start_d1.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
RUN_ID=${RUN_ID:?}
CARDS=${CARDS:?}
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
DEC=${DEC:-logs/mist_w3po_v1/d1/base_decision.json}      # overridable for DRY_RUN checks only
SL=logs/mist_w3po_v1/slots
case $CARDS in 0,1,2,3) SLOT=slotA;; 4,5,6,7) SLOT=slotB;; *) echo "CARDS 0,1,2,3 or 4,5,6,7"; exit 1;; esac
log() { echo "$(date -u +%FT%TZ) $*"; }
field() { $PY -c "import json,sys; d=json.load(open('$1')); print(eval(sys.argv[1], {}, {'d': d}))" "$2"; }
busy_on_cards() {   # compute processes on CARDS that are not our placeholders
  local map pid bus idx cmd
  map=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>/dev/null) || { echo "smi-failed"; return; }
  nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader 2>/dev/null | while IFS=', ' read -r pid bus; do
    idx=$(echo "$map" | awk -F', ' -v b="$bus" '$2 == b {print $1}')
    case ",$CARDS," in *",$idx,"*) ;; *) continue;; esac
    cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null) || cmd="(unreadable)"
    case "$cmd" in *scripts/gpu_load/train_mist_workload.py*|*gpu_keeper.py*) ;; *) echo "$idx:$pid";; esac
  done
}
if [ -z "${DRY_RUN:-}" ]; then until [ -s $DEC ]; do sleep 30; done; fi
[ -s $DEC ] || { echo "no $DEC yet"; exit 0; }
BASE=$(field $DEC "d['base']")
[ "$BASE" != None ] || { log "no D1 base"; exit 1; }
S=$(field $DEC "d['data']['sched']"); D=$(field $DEC "d['data']['dart']"); M=$(field $DEC "d['data']['lw_mode']")
MAN=eval_outputs/mist_w3po_v1/crf/d1_${BASE}_w10_v1
EXTRA="--sched $S --objective latent_write --lw-mode $M --dart $D --d1 $MAN"
LAUNCH="CARDS=$CARDS RUN_ID=$RUN_ID PORT=${PORT:-29985} EXTRA=\"$EXTRA\" bash experiments/mist_w3po_v1/scripts/run_crf_v2p.sh"
if [ -n "${DRY_RUN:-}" ]; then
  echo "base $BASE; manifest $MAN ($([ -e $MAN/summary.json ] && echo "R4 $(field $MAN/summary.json "d['R4_pass']")" || echo "not built yet"))"
  echo "slot $SLOT owner: $([ -e $SL/$SLOT.owner ] && head -c 150 $SL/$SLOT.owner || echo none)"
  echo "non-placeholder compute processes on $CARDS: $(busy_on_cards | tr '\n' ' ')"
  CARDS=$CARDS LAUNCH="$LAUNCH" DRY_RUN=1 bash experiments/mist_w3po_v1/scripts/takeover_train.sh
  eval "DRY_RUN=1 $LAUNCH"
  exit 0
fi
log "waiting for $MAN/summary.json"
until [ -s $MAN/summary.json ]; do sleep 30; done
[ "$(field $MAN/summary.json "d['R4_pass']")" = True ] || { log "R4 not passed: D1 not trained (prereg §4.2)"; exit 1; }
log "waiting for $SLOT ($CARDS) to hold only placeholders"
while :; do
  B=$(busy_on_cards)
  [ ! -e $SL/$SLOT.owner ] && [ -z "$B" ] && break
  sleep 30
done
( set -o noclobber; printf 'who: method-design executor (mist_w3po_v1 D1)\ntask: D1 training %s (%s) on cards %s, ~5-8 h; released by renaming to %s.released_<UTC>\nutc: %s\n' \
  "$RUN_ID" "$EXTRA" "$CARDS" "$SLOT" "$(date -u +%FT%TZ)" > $SL/$SLOT.owner ) && log "$SLOT.owner claimed" \
  || { log "$SLOT.owner claim failed: not launched"; exit 1; }
CARDS=$CARDS LAUNCH="$LAUNCH" bash experiments/mist_w3po_v1/scripts/takeover_train.sh
for i in $(seq 1 15); do
  sleep 10
  n=$(ps -eo cmd= | grep -c "[t]rain_crf_sft.py .*--out eval_outputs/mist_w3po_v1/crf/train/$RUN_ID")
  [ "$n" -ge 4 ] && break
done
log "D1 ranks running: $n"
