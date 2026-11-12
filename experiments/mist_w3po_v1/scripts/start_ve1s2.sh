#!/usr/bin/env bash
# VE1-s2 training (reports/mist_w3po_v1/thinker_ve1_s2_confirm_prereg_v1.zh-CN.md §1): start_ve1.sh's VE1 recipe
# unchanged (start_ve1.sh untouched) except the training seed --seed 20261112 (VE1: the default 20261109) and the DART
# manifest rebuilt for that plan (crf/ve1s2_dart_merge3_w10_v1: dart_aug --manifest --lw-mode v2p --plan-seed 20261112;
# the trainer re-checks every draw against its own plan at load). No schedule.json condition (coordinator decision).
# Waits until the slot of CARDS is free (no slot owner, only our placeholders on its cards), claims it (noclobber),
# takes it over with takeover_train.sh (placeholders stopped by PID, launch in the same breath), checks 4 ranks.
#   RUN_ID=ve1s2_<UTC> CARDS=0,1,2,3|4,5,6,7 [PORT=29995] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/start_ve1s2.sh
# Run detached: setsid nohup bash .../start_ve1s2.sh > logs/mist_w3po_v1/ve1s2/start_ve1s2.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
RUN_ID=${RUN_ID:?}
CARDS=${CARDS:?}
SEED=20261112
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
C=eval_outputs/mist_w3po_v1/crf
S=$C/sched_t5_merge3_w10_v1
D=$C/ve1s2_dart_merge3_w10_v1
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
EXTRA="--sched $S --objective latent_write --lw-mode v2p --dart $D --seed $SEED"
LAUNCH="CARDS=$CARDS RUN_ID=$RUN_ID PORT=${PORT:-29995} EXTRA=\"$EXTRA\" bash experiments/mist_w3po_v1/scripts/run_crf_v2p.sh"
MSEED=$(field $D/summary.json "d['plan']['seed']" 2>/dev/null || echo missing)
MSCHED=$(field $D/summary.json "d['sched']" 2>/dev/null || echo missing)
if [ -n "${DRY_RUN:-}" ]; then
  echo "VE1 precheck pass: $(field $PC "d['pass']" 2>/dev/null || echo missing); manifest plan seed: $MSEED (want $SEED); manifest sched: $MSCHED"
  echo "slot $SLOT owner: $([ -e $SL/$SLOT.owner ] && head -c 150 $SL/$SLOT.owner || echo none)"
  echo "non-placeholder compute processes on $CARDS: $(busy_on_cards | tr '\n' ' ')"
  CARDS=$CARDS LAUNCH="$LAUNCH" DRY_RUN=1 bash experiments/mist_w3po_v1/scripts/takeover_train.sh
  eval "DRY_RUN=1 $LAUNCH"
  exit 0
fi
[ "$(field $PC "d['pass']" 2>/dev/null)" = True ] || { log "VE1 CPU precheck missing or failed: not launched"; exit 1; }
[ "$MSEED" = $SEED ] && [ "$MSCHED" = $S ] || { log "manifest $D: plan seed $MSEED / sched $MSCHED, want $SEED / $S: not launched"; exit 1; }
log "VE1-s2: waiting for $SLOT ($CARDS) to hold only placeholders"
while :; do
  B=$(busy_on_cards)
  [ ! -e $SL/$SLOT.owner ] && [ -z "$B" ] && break
  sleep 15
done
( set -o noclobber; printf 'who: Thinker executor v4 (mist_w3po_v1 VE1-s2)\ntask: VE1-s2 training %s (%s) on cards %s, ~3.2 h; released by renaming to %s.released_<UTC>\nutc: %s\n' \
  "$RUN_ID" "$EXTRA" "$CARDS" "$SLOT" "$(date -u +%FT%TZ)" > $SL/$SLOT.owner ) && log "$SLOT.owner claimed" \
  || { log "$SLOT.owner claim failed: not launched"; exit 1; }
CARDS=$CARDS LAUNCH="$LAUNCH" bash experiments/mist_w3po_v1/scripts/takeover_train.sh
for i in $(seq 1 15); do
  sleep 10
  n=$(ps -eo cmd= | grep -c "[t]rain_crf_sft.py .*--out eval_outputs/mist_w3po_v1/crf/train/$RUN_ID")
  [ "$n" -ge 4 ] && break
done
log "VE1-s2 ranks running: $n"
