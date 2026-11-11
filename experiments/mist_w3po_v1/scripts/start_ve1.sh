#!/usr/bin/env bash
# VE1 training (reports/mist_w3po_v1/thinker_v2p_evidence_ve1_prereg_v1.zh-CN.md §1-§2): v2' objective (implemented as
# latent_write --lw-mode v2p: lam* = 0, posterior = serial_v2p's, checked in ve1_precheck) + T5's evidence-time labels
# at merge3 granularity (sched_t5_merge3_w10_v1) + DART (ve1_dart_merge3_w10_v1), 1.0 s reads, 3000 steps, 4 cards.
# Runs only when D1 runs (logs/mist_w3po_v1/d1/schedule.json run_D1; FORCE=1 overrides) and the CPU precheck
# (eval_outputs/mist_w3po_v1/crf/ve1_precheck_v1.json) passed. Waits until the slot of CARDS is free (no slot owner,
# only our placeholders on its cards), claims it (noclobber), takes it over with takeover_train.sh, checks 4 ranks.
#   RUN_ID=ve1_<UTC> CARDS=0,1,2,3|4,5,6,7 [PORT=29987] [FORCE=1] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/start_ve1.sh
# Run detached: setsid nohup bash .../start_ve1.sh > logs/mist_w3po_v1/ve1/start_ve1.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
RUN_ID=${RUN_ID:?}
CARDS=${CARDS:?}
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
C=eval_outputs/mist_w3po_v1/crf
SCH=${SCH:-logs/mist_w3po_v1/d1/schedule.json}      # overridable for DRY_RUN checks only
PC=$C/ve1_precheck_v1.json
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
EXTRA="--sched $C/sched_t5_merge3_w10_v1 --objective latent_write --lw-mode v2p --dart $C/ve1_dart_merge3_w10_v1"
LAUNCH="CARDS=$CARDS RUN_ID=$RUN_ID PORT=${PORT:-29987} EXTRA=\"$EXTRA\" bash experiments/mist_w3po_v1/scripts/run_crf_v2p.sh"
if [ -n "${DRY_RUN:-}" ]; then
  echo "precheck pass: $(field $PC "d['pass']" 2>/dev/null || echo missing); schedule run_D1: $(field $SCH "d['run_D1']" 2>/dev/null || echo 'not written yet')"
  echo "slot $SLOT owner: $([ -e $SL/$SLOT.owner ] && head -c 150 $SL/$SLOT.owner || echo none)"
  echo "non-placeholder compute processes on $CARDS: $(busy_on_cards | tr '\n' ' ')"
  CARDS=$CARDS LAUNCH="$LAUNCH" DRY_RUN=1 bash experiments/mist_w3po_v1/scripts/takeover_train.sh
  eval "DRY_RUN=1 $LAUNCH"
  exit 0
fi
[ "$(field $PC "d['pass']" 2>/dev/null)" = True ] || { log "VE1 CPU precheck missing or failed: not launched"; exit 1; }
if [ "${FORCE:-}" != 1 ]; then
  log "waiting for $SCH"
  until [ -s $SCH ]; do sleep 30; done
  [ "$(field $SCH "d['run_D1']")" = True ] || { log "schedule: D1 does not run -> VE1 does not run (prereg §2)"; exit 0; }
fi
log "waiting for $SLOT ($CARDS) to hold only placeholders"
while :; do
  B=$(busy_on_cards)
  [ ! -e $SL/$SLOT.owner ] && [ -z "$B" ] && break
  sleep 30
done
( set -o noclobber; printf 'who: method-design executor (mist_w3po_v1 VE1)\ntask: VE1 training %s (%s) on cards %s, ~4.5-6 h; released by renaming to %s.released_<UTC>\nutc: %s\n' \
  "$RUN_ID" "$EXTRA" "$CARDS" "$SLOT" "$(date -u +%FT%TZ)" > $SL/$SLOT.owner ) && log "$SLOT.owner claimed" \
  || { log "$SLOT.owner claim failed: not launched"; exit 1; }
CARDS=$CARDS LAUNCH="$LAUNCH" bash experiments/mist_w3po_v1/scripts/takeover_train.sh
for i in $(seq 1 15); do
  sleep 10
  n=$(ps -eo cmd= | grep -c "[t]rain_crf_sft.py .*--out eval_outputs/mist_w3po_v1/crf/train/$RUN_ID")
  [ "$n" -ge 4 ] && break
done
log "VE1 ranks running: $n"
