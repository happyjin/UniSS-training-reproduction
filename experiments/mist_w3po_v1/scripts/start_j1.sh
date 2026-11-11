#!/usr/bin/env bash
# J1 training (reports/mist_w3po_v1/thinker_joint_crf_prereg_v1.zh-CN.md §4): the joint write-plan x playback CRF on the
# base picked by crf/j1_decide.py (logs/mist_w3po_v1/j1/base_decision.json: sched, DART manifest, lattice mode), T3's
# talker model, the base's recipe (run_crf_v2p.sh: 3000 steps, lr 1e-4 cosine, tok 32000, cap 120, ...), 4 cards.
#   1. waits for base_decision.json (run "python -m experiments.mist_w3po_v1.crf.j1_decide --wait" once, detached);
#   2. refuses when the CPU precheck (eval_outputs/mist_w3po_v1/crf/j1_precheck_v1.json) failed a line, unless
#      J1_PC_OVERRIDE=1 (prereg §5: PC2 / PC4 failures go to the coordinator / user; PC3 failures never launch);
#   3. waits until the slot of CARDS is free: no logs/mist_w3po_v1/slots/<slot>.owner and every compute process on
#      those cards is one of our placeholders (scripts/gpu_load/train_mist_workload.py or gpu_keeper.py), checked every
#      30 s with nvidia-smi;
#   4. claims the slot (noclobber owner file) and takes the cards over with takeover_train.sh (our placeholders stopped
#      by PID, the training launched in the same breath), then checks the 4 ranks.
# Checked with bash -n and DRY_RUN=1 only (never test-run).
#   RUN_ID=j1_<base>_<UTC> CARDS=0,1,2,3|4,5,6,7 [PORT=29981] [J1_PC_OVERRIDE=1] [DRY_RUN=1] \
#     bash experiments/mist_w3po_v1/scripts/start_j1.sh
# Run detached: setsid nohup bash .../start_j1.sh > logs/mist_w3po_v1/j1/start_j1.log 2>&1 < /dev/null &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
RUN_ID=${RUN_ID:?}
CARDS=${CARDS:?}
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
DEC=${DEC:-logs/mist_w3po_v1/j1/base_decision.json}      # overridable for DRY_RUN checks only
PC=${PC:-eval_outputs/mist_w3po_v1/crf/j1_precheck_v1.json}
TM=eval_outputs/mist_w3po_v1/crf/t3_talker_model_v1.json
SL=logs/mist_w3po_v1/slots
case $CARDS in 0,1,2,3) SLOT=slotA;; 4,5,6,7) SLOT=slotB;; *) echo "CARDS 0,1,2,3 or 4,5,6,7"; exit 1;; esac
log() { echo "$(date -u +%FT%TZ) $*"; }
# compute processes on CARDS that are NOT our placeholders (empty = the slot is free of real work)
busy_on_cards() {
  local map pid bus idx cmd
  map=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>/dev/null) || { echo "smi-failed"; return; }
  nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader 2>/dev/null | while IFS=', ' read -r pid bus; do
    idx=$(echo "$map" | awk -F', ' -v b="$bus" '$2 == b {print $1}')
    case ",$CARDS," in *",$idx,"*) ;; *) continue;; esac
    cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null) || cmd="(unreadable)"
    case "$cmd" in *scripts/gpu_load/train_mist_workload.py*|*gpu_keeper.py*) ;; *) echo "$idx:$pid";; esac
  done
}
pc_ok() {
  $PY - "$PC" "$DEC" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
bad = [k for k, ok in (("PC2", d["PC2"]["pass"]),) if not ok]
dec = json.load(open(sys.argv[2]))
b = dec.get("base")
if b:
    bad += [f"{b}.{k}" for k in ("PC3", "PC4") if not d[b][k]["pass"]]
print(" ".join(bad))
EOF
}
build_launch() {
  read -r S D M < <($PY -c "import json; d=json.load(open('$DEC'))['data']; print(d['sched'], d['dart'], d['lw_mode'])")
  EXTRA="--sched $S --objective joint_speech --lw-mode $M --dart $D --talker-model $TM"
  LAUNCH="CARDS=$CARDS RUN_ID=$RUN_ID PORT=${PORT:-29981} EXTRA=\"$EXTRA\" bash experiments/mist_w3po_v1/scripts/run_crf_v2p.sh"
}
if [ -n "${DRY_RUN:-}" ]; then
  echo "slot $SLOT; owner file: $([ -e $SL/$SLOT.owner ] && head -c 200 $SL/$SLOT.owner || echo none)"
  echo "non-placeholder compute processes on $CARDS: $(busy_on_cards | tr '\n' ' ')"
  if [ -e $DEC ]; then
    build_launch; echo "decision: $(head -c 300 $DEC)"; echo "precheck failures: $(pc_ok)"
    CARDS=$CARDS LAUNCH="$LAUNCH" DRY_RUN=1 bash experiments/mist_w3po_v1/scripts/takeover_train.sh
    eval "DRY_RUN=1 $LAUNCH"
  else
    echo "no $DEC yet (j1_decide --wait writes it after both quick checks)"
  fi
  exit 0
fi
log "waiting for $DEC"
until [ -s $DEC ]; do sleep 30; done
[ "$($PY -c "import json; print(json.load(open('$DEC')).get('base'))")" != None ] || { log "no J1 base (neither T4 nor T5 trained cleanly): not launched"; exit 1; }
build_launch
F=$(pc_ok)
if [ -n "$F" ] && [ "${J1_PC_OVERRIDE:-}" != 1 ]; then
  log "CPU precheck failed ($F): not launched (prereg §5: the coordinator / user decides; J1_PC_OVERRIDE=1 to launch)"; exit 1
fi
case "$F" in *PC3*) log "PC3 failed ($F): never launched"; exit 1;; esac
log "base $(head -c 200 $DEC); waiting for $SLOT ($CARDS) to hold only placeholders"
while :; do
  B=$(busy_on_cards)
  [ ! -e $SL/$SLOT.owner ] && [ -z "$B" ] && break
  sleep 30
done
( set -o noclobber; printf 'who: method-design executor (mist_w3po_v1 J1)\ntask: J1 training %s (%s) on cards %s, ~5-7.5 h; released by renaming to %s.released_<UTC>\nutc: %s\n' \
  "$RUN_ID" "$EXTRA" "$CARDS" "$SLOT" "$(date -u +%FT%TZ)" > $SL/$SLOT.owner ) && log "$SLOT.owner claimed" \
  || { log "$SLOT.owner claim failed (someone holds it): not launched"; exit 1; }
CARDS=$CARDS LAUNCH="$LAUNCH" bash experiments/mist_w3po_v1/scripts/takeover_train.sh
for i in $(seq 1 15); do
  sleep 10
  n=$(ps -eo cmd= | grep -c "[t]rain_crf_sft.py .*--out eval_outputs/mist_w3po_v1/crf/train/$RUN_ID")
  [ "$n" -ge 4 ] && break
done
log "J1 ranks running: $n"
