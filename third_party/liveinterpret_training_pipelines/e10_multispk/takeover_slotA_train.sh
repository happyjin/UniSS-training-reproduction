#!/usr/bin/env bash
# E10 on slot A (cards 0-3; user 2026-11-19: "用前四张卡训练"; validation stacked on the training cards):
#   1. claim logs/mist_w3po_v1/slots/slotA.owner (refuse if held);
#   2. stop the no-checkpoint placeholders of cards 0-3 by PID (supervisor + children; same rule as idea1_swap_run.sh)
#      and start the training (launch_e10_detached.sh, GPUs 0-3, GBS 64 = 8 x 4 x GA 2);
#   3. for up to 15 min stop any placeholder the guard restarts on 0-3 until each card holds >= 20 GiB of the training;
#   4. stacked on the training: E10-0 dev evaluation (devz, deve; first TTS seed; 2 decoders per card);
#   5. after the training: E10 dev-gate evaluation with the last checkpoint (prereg §3; placeholders stopped again);
#   6. release slot A (rename to slotA.released_<UTC>) and write the DONE / FAILED marker.
# Run detached:  setsid nohup bash e10_multispk/takeover_slotA_train.sh > <log> 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only (DRY_RUN prints the actions, touches nothing).
set -uo pipefail
U=/opt/dlami/nvme/jasonleeeli/projects/UniSS
LI=$U/third_party/liveinterpret_training_pipelines
E=$LI/e10_multispk
SL=$U/logs/mist_w3po_v1/slots
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
RUN_LABEL=${RUN_LABEL:-e10_thinker_v1}
OUT=$E/output
MK=$OUT/markers
INIT=$LI/checkpoints/p4mix_2n_20261117/checkpoint-29773
CARDS="0 1 2 3"
DRY=${DRY_RUN:-}
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
act() { if [ -n "$DRY" ]; then log "[dry] $*"; else "$@"; fi; }

stop_placeholders() {   # by PID: train_mist_workload supervisors (ppid 1, --devices c) of cards 0-3 and their children
  local c sup kids n=0
  for c in $CARDS; do
    for sup in $(ps -eo pid=,ppid=,args= | awk -v c="--devices $c " '$2==1 && index($0,"train_mist_workload.py") && index($0,c) {print $1}'); do
      kids=$(ps -eo pid=,ppid= | awk -v p="$sup" '$2==p {print $1}')
      log "card $c: stop placeholder supervisor $sup (children: ${kids:-none})"
      act kill -TERM $sup $kids
      n=$((n+1))
    done
  done
  [ -n "$DRY" ] || { [ "$n" -gt 0 ] && for _ in $(seq 1 60); do
      ps -eo args= | grep -q "[t]rain_mist_workload.py.*--devices [0-3] " || break; sleep 1; done; }
  return 0
}
card_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }
fail() { log "FAILED: $*"; [ -n "$DRY" ] || { mkdir -p "$MK"; echo "$*" > "$MK/E10_TRAIN_FAILED"; release; }; exit 1; }
release() { [ -e "$SL/slotA.owner" ] && command mv -f "$SL/slotA.owner" "$SL/slotA.released_$(date -u +%Y%m%dT%H%M%SZ)" \
            && log "slot A released"; }

# ---- 1. claim slot A
if [ -e "$SL/slotA.owner" ]; then log "slot A is held: $(head -3 "$SL/slotA.owner" | tr '\n' ' ')"; exit 1; fi
[ -e "$OUT/$RUN_LABEL" ] && { log "$OUT/$RUN_LABEL exists; refusing"; exit 1; }
if [ -z "$DRY" ]; then
  mkdir -p "$MK"
  ( set -o noclobber; printf 'who: E10 LiveInterpret S5 multi-speaker baseline (user 2026-11-19)\ntask: E10 Thinker-only training %s on cards 0-3 (GBS 64), E10-0 dev eval stacked, then E10 dev gate; released by renaming to slotA.released_<UTC>\nutc: %s\npid: %s\n' \
      "$RUN_LABEL" "$(date -u +%FT%TZ)" "$$" > "$SL/slotA.owner" ) || { log "could not claim slot A"; exit 1; }
fi
log "slot A claimed"

# ---- 2. hand the cards over and start the training
stop_placeholders
cd "$LI"
act env CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29641 RUN_LABEL="$RUN_LABEL" bash "$E/launch_e10_detached.sh" \
  || fail "launch_e10_detached.sh"
PIDF=$OUT/$RUN_LABEL.pid

# ---- 3. keep the guard's placeholders off until the training holds the cards
if [ -z "$DRY" ]; then
  for i in $(seq 1 90); do
    sleep 10
    kill -0 "$(cat "$PIDF")" 2>/dev/null || fail "training exited during start-up (see $OUT/$RUN_LABEL.log)"
    stop_placeholders
    ok=1; for c in $CARDS; do [ "$(card_mib "$c")" -ge 20480 ] 2>/dev/null || ok=0; done
    [ "$ok" = 1 ] && { log "training holds cards 0-3"; break; }
  done
  [ "$ok" = 1 ] || fail "training did not reach 20 GiB on every card within 15 min"
fi

# ---- 4. E10-0 dev evaluation, stacked on the training
cd "$U"
act "$PYE" "$E/eval_e10.py" --run --mode e10_0 --adapter "$INIT" --sets devz,deve --seeds first \
    --gpus 0,1,2,3 --procs-per-gpu 2 || log "E10-0 dev evaluation failed (continuing; see $OUT/eval/logs)"

# ---- 5. wait for the training, then the E10 dev gate
if [ -z "$DRY" ]; then
  while kill -0 "$(cat "$PIDF")" 2>/dev/null; do sleep 60; done
  grep -q "Traceback" "$OUT/$RUN_LABEL.log" && fail "training log has a Traceback"
fi
LAST=$(ls -d "$OUT/$RUN_LABEL"/v*/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
[ -n "$DRY" ] && LAST=${LAST:-$OUT/$RUN_LABEL/v0-XXXX/checkpoint-LAST}
[ -n "$LAST" ] && [ -e "$LAST/e10_metadata.json" -o -n "$DRY" ] || fail "no final E10 checkpoint"
log "last checkpoint $LAST"
stop_placeholders
act "$PYE" "$E/eval_e10.py" --run --mode e10 --adapter "$LAST" --sets devz,deve --seeds first \
    --gpus 0,1,2,3 --procs-per-gpu 4 || fail "E10 dev evaluation"

# ---- 6. done
[ -n "$DRY" ] || { echo "$LAST" > "$MK/E10_TRAIN_DEVGATE_DONE"; release; }
log "E10 training + dev evaluations done; test decoding waits for the dev gate check"
