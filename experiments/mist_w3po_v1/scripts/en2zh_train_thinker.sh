#!/usr/bin/env bash
# En->Zh A' Thinker (reports/mist_w3po_v1/en2zh_full_plan_prereg_v1.zh-CN.md §3.1): SerialCRF v2' exactly as the zh->en
# A' (run_crf_v2p.sh: --objective serial_v2p, tau 0.5, --final-read-full-post 1, 3000 steps, ~64 read rows / step,
# lr 1e-4 -> 1e-5, warm-up 50, seed 20261109, LoRA r32 / alpha 64 + projector + new rows, checkpoints every 500 steps)
# with only the direction swapped: --sched crf/sched_en2zh_v1 (21,581 En->Zh train windows) --base the En->Zh v8
# (thinker_gstar_s2dl_20261029b) --dir en2zh. ONE seed. 4 exclusive cards.
# Waits until the slot of CARDS holds only our placeholders (no slot owner, no other compute process), claims it
# (noclobber owner file; sync_train_exclude.sh then keeps every eval orchestrator off these cards), takes it over with
# takeover_train.sh (placeholders stopped by PID, training started in the same breath) and checks 4 ranks.
#   RUN_ID=en2zh_v2p_<UTC> CARDS=0,1,2,3|4,5,6,7 [PORT=29993] [DRY_RUN=1] \
#     bash experiments/mist_w3po_v1/scripts/en2zh_train_thinker.sh
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/en2zh/train_thinker.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
RUN_ID=${RUN_ID:?}
CARDS=${CARDS:?}
C=eval_outputs/mist_w3po_v1/crf
SCH=$C/sched_en2zh_v1
BASE=eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dl_20261029b/thinker_merged
SL=logs/mist_w3po_v1/slots
case $CARDS in 0,1,2,3) SLOT=slotA;; 4,5,6,7) SLOT=slotB;; *) echo "CARDS 0,1,2,3 or 4,5,6,7"; exit 1;; esac
log() { echo "$(date -u +%FT%TZ) $*"; }
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
[ -e $SCH/summary.json ] && [ -e $BASE/sd_proj.pt ] || { log "missing $SCH or $BASE"; exit 1; }
EXTRA="--sched $SCH --base $BASE --dir en2zh"
LAUNCH="CARDS=$CARDS RUN_ID=$RUN_ID PORT=${PORT:-29993} EXTRA=\"$EXTRA\" bash experiments/mist_w3po_v1/scripts/run_crf_v2p.sh"
if [ -n "${DRY_RUN:-}" ]; then
  echo "slot $SLOT owner: $([ -e $SL/$SLOT.owner ] && head -c 150 $SL/$SLOT.owner || echo none)"
  echo "non-placeholder compute processes on $CARDS: $(busy_on_cards | tr '\n' ' ')"
  CARDS=$CARDS LAUNCH="$LAUNCH" DRY_RUN=1 bash experiments/mist_w3po_v1/scripts/takeover_train.sh
  eval "DRY_RUN=1 $LAUNCH"
  exit 0
fi
log "waiting for $SLOT ($CARDS) to hold only placeholders"
while :; do
  B=$(busy_on_cards)
  [ ! -e $SL/$SLOT.owner ] && [ -z "$B" ] && break
  sleep 30
done
( set -o noclobber; printf 'who: En->Zh preparation agent (mist_w3po_v1 en2zh A'"'"')\ntask: En->Zh SerialCRF v2'"'"' training %s (%s) on cards %s, ~3.5 h; released by renaming to %s.released_<UTC>\nutc: %s\n' \
  "$RUN_ID" "$EXTRA" "$CARDS" "$SLOT" "$(date -u +%FT%TZ)" > $SL/$SLOT.owner ) && log "$SLOT.owner claimed" \
  || { log "$SLOT.owner claim failed: not launched"; exit 1; }
CARDS=$CARDS LAUNCH="$LAUNCH" bash experiments/mist_w3po_v1/scripts/takeover_train.sh
n=0
for i in $(seq 1 15); do
  sleep 10
  n=$(ps -eo cmd= | grep -c "[t]rain_crf_sft.py .*--out eval_outputs/mist_w3po_v1/crf/train/$RUN_ID ")
  [ "$n" -ge 4 ] && break
done
log "en2zh Thinker ranks running: $n"
[ "$n" -ge 4 ] && log "EN2ZH_THINKER_TRAINING_STARTED $RUN_ID" || log "EN2ZH_THINKER_TRAINING_RANKS_MISSING $RUN_ID ($n)"
