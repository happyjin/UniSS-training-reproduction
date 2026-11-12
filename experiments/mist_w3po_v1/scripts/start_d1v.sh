#!/usr/bin/env bash
# D1v training (reports/mist_w3po_v1/thinker_dagger_d1v_prereg_v1.zh-CN.md §1.4): VE1's recipe and data
# (crf/sched_t5_merge3_w10_v1, --lw-mode v2p, VE1's DART manifest crf/ve1_dart_merge3_w10_v1, seed 20261109) + the VE1
# learner's roll-in rows (--d1 crf/d1_ve1_w10_v1 of start_d1v_rollin.sh), 3000 steps, 4 cards (run_crf_v2p.sh).
# Waits for the manifest; R4 false (fewer than 5 % of the plan windows ever overdue) -> not launched (prereg §2.3).
# Then waits until the slot of CARDS is free (no slot owner, every compute process on those cards one of our
# placeholders), claims it (noclobber), takes it over with takeover_train.sh (placeholders stopped by PID, launch in the
# same breath) and checks the 4 ranks.
#   RUN_ID=d1v_<UTC> [CARDS=0,1,2,3] [PORT=29975] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/start_d1v.sh
# Run detached: setsid nohup bash .../start_d1v.sh > logs/mist_w3po_v1/d1v/start_d1v.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
RUN_ID=${RUN_ID:?}
CARDS=${CARDS:-0,1,2,3}
SEED=20261109
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
C=eval_outputs/mist_w3po_v1/crf
S=$C/sched_t5_merge3_w10_v1
D=$C/ve1_dart_merge3_w10_v1
MAN=$C/d1_ve1_w10_v1
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
EXTRA="--sched $S --objective latent_write --lw-mode v2p --dart $D --d1 $MAN"
LAUNCH="CARDS=$CARDS RUN_ID=$RUN_ID PORT=${PORT:-29975} EXTRA=\"$EXTRA\" bash experiments/mist_w3po_v1/scripts/run_crf_v2p.sh"
DSEED=$(field $D/summary.json "d['plan']['seed']" 2>/dev/null || echo missing)
DMODE=$(field $D/summary.json "d['lw_mode']" 2>/dev/null || echo missing)
DSCHED=$(field $D/summary.json "d['sched']" 2>/dev/null || echo missing)
if [ -n "${DRY_RUN:-}" ]; then
  echo "DART manifest: plan seed $DSEED (want $SEED), mode $DMODE (want v2p), sched $DSCHED"
  echo "D1 manifest $MAN: $([ -e $MAN/summary.json ] && echo "R4 $(field $MAN/summary.json "d['R4_pass']")" || echo 'not built yet')"
  echo "slot $SLOT owner: $([ -e $SL/$SLOT.owner ] && head -c 150 $SL/$SLOT.owner || echo none)"
  echo "non-placeholder compute processes on $CARDS: $(busy_on_cards | tr '\n' ' ')"
  CARDS=$CARDS LAUNCH="$LAUNCH" DRY_RUN=1 bash experiments/mist_w3po_v1/scripts/takeover_train.sh
  eval "DRY_RUN=1 $LAUNCH"
  exit 0
fi
[ "$DSEED" = $SEED ] && [ "$DMODE" = v2p ] && [ "$DSCHED" = $S ] || {
  log "DART manifest $D: seed $DSEED / mode $DMODE / sched $DSCHED, want $SEED / v2p / $S: not launched"; exit 1; }
log "waiting for $MAN/summary.json"
until [ -s $MAN/summary.json ]; do sleep 30; done
[ "$(field $MAN/summary.json "d['R4_pass']")" = True ] || { log "R4 not passed: D1v not trained (prereg §2.3)"; exit 1; }
log "R4 passed ($(field $MAN/summary.json "d['share_windows_with_overdue']")); waiting for $SLOT ($CARDS) to hold only placeholders"
while :; do
  B=$(busy_on_cards)
  [ ! -e $SL/$SLOT.owner ] && [ -z "$B" ] && break
  sleep 15
done
( set -o noclobber; printf 'who: method designer v4a / Thinker executor (mist_w3po_v1 D1v)\ntask: D1v training %s (%s) on cards %s, ~3.5-4 h; released by renaming to %s.released_<UTC>\nutc: %s\n' \
  "$RUN_ID" "$EXTRA" "$CARDS" "$SLOT" "$(date -u +%FT%TZ)" > $SL/$SLOT.owner ) && log "$SLOT.owner claimed" \
  || { log "$SLOT.owner claim failed: not launched"; exit 1; }
CARDS=$CARDS LAUNCH="$LAUNCH" bash experiments/mist_w3po_v1/scripts/takeover_train.sh
n=0
for i in $(seq 1 15); do
  sleep 10
  n=$(ps -eo cmd= | grep -c "[t]rain_crf_sft.py .*--out eval_outputs/mist_w3po_v1/crf/train/$RUN_ID")
  [ "$n" -ge 4 ] && break
done
log "D1v ranks running: $n"
