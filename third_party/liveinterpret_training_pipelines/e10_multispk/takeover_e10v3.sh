#!/usr/bin/env bash
# E10 v3 on slot A (cards 0-3; user 2026-11-21 "目前前四块卡可以训练了"):
#   0. slot A must already be held for E10 v3 (logs/mist_w3po_v1/slots/slotA.owner mentions "E10 v3");
#   1. card keeper: every 10 s, stop by PID the guard's no-checkpoint placeholders on cards 0-3 (they would grab
#      the cards in the gaps between the stages); the guard itself is never touched;
#   2. stage O2  : train_e10v3.sh STAGE=o2   (2000 x GBS 64 = 8 x 4 GPUs x GA 2)
#   3. stage joint: train_e10v3.sh STAGE=joint O2_ADAPTER=<O2 last checkpoint> (6000 x GBS 24 = 6 x 4 x GA 1)
#   4. demo decode: run_e10.py --mode e10v3 on the six esports sessions of the E10 demo (gz01-03 zh2en,
#      ge01/02/04 en2zh), 3 decoders per card, first TTS seed -> output/demo_game_v3/raw_{zh2en,en2zh}
#   5. stop the keeper, release slot A (rename), write DONE / FAILED marker.
# Run detached:  setsid nohup bash e10_multispk/takeover_e10v3.sh > <log> 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only (DRY_RUN prints the actions, touches nothing).
set -uo pipefail
U=/opt/dlami/nvme/jasonleeeli/projects/UniSS
LI=$U/third_party/liveinterpret_training_pipelines
E=$LI/e10_multispk
SL=$U/logs/mist_w3po_v1/slots
PY=/opt/dlami/nvme/jasonleeeli/venvs/simuls2s/bin
OUT=$E/output
MK=$OUT/markers
DEMO=$OUT/demo_game_v3
CARDS="0 1 2 3"
DRY=${DRY_RUN:-}
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
act() { if [ -n "$DRY" ]; then log "[dry] $*"; else "$@"; fi; }

stop_placeholders() {   # by PID: train_mist_workload supervisors (ppid 1, --devices c) of cards 0-3 and their children
  local c sup kids
  for c in $CARDS; do
    for sup in $(ps -eo pid=,ppid=,args= | awk -v c="--devices $c " '$2==1 && index($0,"train_mist_workload.py") && index($0,c) {print $1}'); do
      kids=$(ps -eo pid=,ppid= | awk -v p="$sup" '$2==p {print $1}')
      log "card $c: stop placeholder supervisor $sup (children: ${kids:-none})"
      [ -n "$DRY" ] || kill -TERM $sup $kids 2>/dev/null
    done
  done
  return 0
}
release() { [ -e "$SL/slotA.owner" ] && command mv -f "$SL/slotA.owner" "$SL/slotA.released_$(date -u +%Y%m%dT%H%M%SZ)" \
            && log "slot A released"; }
KEEPER=""
stop_keeper() { [ -n "$KEEPER" ] && kill -TERM "$KEEPER" 2>/dev/null; KEEPER=""; }
fail() { log "FAILED: $*"; stop_keeper; [ -n "$DRY" ] || { mkdir -p "$MK"; echo "$*" > "$MK/E10V3_FAILED"; release; }; exit 1; }
last_ckpt() { ls -d "$OUT/$1"/v*/checkpoint-* 2>/dev/null | sort -V | tail -1; }

# ---- 0. slot A
grep -q "E10 v3" "$SL/slotA.owner" 2>/dev/null || { log "slot A is not held for E10 v3"; exit 1; }
for d in "$OUT/e10v3_o2" "$OUT/e10v3_joint" "$DEMO"; do [ -e "$d" ] && { log "$d exists; refusing"; exit 1; }; done
[ -n "$DRY" ] || mkdir -p "$MK"

# ---- 1. keeper
if [ -z "$DRY" ]; then
  ( while true; do stop_placeholders; sleep 10; done ) &
  KEEPER=$!
  log "card keeper $KEEPER"
fi
trap 'stop_keeper' EXIT

cd "$LI"
# ---- 2. stage O2
log "stage O2 start"
act env STAGE=o2 CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 MASTER_PORT=29661 OMP_NUM_THREADS=4 \
    PER_DEVICE_TRAIN_BATCH_SIZE=8 bash "$E/train_e10v3.sh" > "$OUT/e10v3_o2.log" 2>&1 || fail "stage O2 (see $OUT/e10v3_o2.log)"
O2=$(last_ckpt e10v3_o2)
[ -n "$DRY" ] && O2=${O2:-$OUT/e10v3_o2/v0-X/checkpoint-2000}
case "$O2" in *checkpoint-2000) ;; *) fail "stage O2 did not end at step 2000 ($O2)";; esac
log "stage O2 done: $O2"

# ---- 3. stage joint
log "stage joint start"
act env STAGE=joint O2_ADAPTER="$O2" CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 MASTER_PORT=29662 OMP_NUM_THREADS=4 \
    PER_DEVICE_TRAIN_BATCH_SIZE=6 bash "$E/train_e10v3.sh" > "$OUT/e10v3_joint.log" 2>&1 \
    || fail "stage joint (see $OUT/e10v3_joint.log)"
J=$(last_ckpt e10v3_joint)
[ -n "$DRY" ] && J=${J:-$OUT/e10v3_joint/v0-X/checkpoint-6000}
case "$J" in *checkpoint-6000) ;; *) fail "stage joint did not end at step 6000 ($J)";; esac
log "stage joint done: $J"

# ---- 4. demo decode (the E10 demo's sessions; 6 shards per direction, 3 decoders per card)
cd "$U"
act mkdir -p "$DEMO"
pids=""
k=0
for d in zh2en en2zh; do
  case $d in zh2en) M=gz01,gz02,gz03; S=mix; T=en;; en2zh) M=ge01,ge02,ge04; S=ihm_mix; T=zh;; esac
  for s in 0 1 2 3 4 5; do
    g=$(( k % 4 )); k=$((k+1))
    if [ -n "$DRY" ]; then
      log "[dry] card $g: run_e10 --mode e10v3 $d shard $s/6 meetings $M"
    else
      CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=4 E10_THREADS=4 PATH=$PY:$PATH "$PY/python" "$E/run_e10.py" --mode e10v3 \
        --set custom --data "data/game_esports_si_v1/$d" --sess "$S" --tgt "$T" \
        --windows "eval_outputs/mist_w3po_v1/data/windows_game_${d}_c2g06_v1/windows_test_online_${d}.jsonl" \
        --meetings "$M" --adapter "$J" --out "$DEMO/raw_$d" --shard "$s" --num-shards 6 \
        > "$DEMO/decode_${d}_$s.log" 2>&1 < /dev/null &
      pids="$pids $!"
      sleep 20
    fi
  done
done
nf=0
for p in $pids; do wait "$p" || nf=$((nf+1)); done
[ "$nf" = 0 ] || fail "$nf demo decoders failed (see $DEMO/decode_*.log)"
log "demo decode done"

# ---- 5. done
stop_keeper
[ -n "$DRY" ] || { echo "$J" > "$MK/E10V3_DONE"; release; }
log "E10 v3 done: O2 $O2, joint $J, demo raw in $DEMO"
