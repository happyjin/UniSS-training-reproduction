#!/usr/bin/env bash
# Robust wave 17 post-training chain (detached; reports/mist_w3po_v1/robust_wave17_enclora_prereg.zh-CN.md note 7):
#   wait for the trainer TPID (torchrun of run_enc) -> check run_enc/adapters_final.pt -> WiSE 0.5 copy run_enc_wise05
#   (hard link of the adapters + json with alpha 16) -> evaluation chain of enc_wise05 on cards 6,7 -> of enc on cards
#   6,7 (wave17_eval_orch.sh, CC=CA CD=CB) -> mechanism Thinker jobs on cards 6,7 (lane 6: la2 base train600, oracle enc
#   zh2en a, la2 unist train600 if wave14/merged_unist exists; lane 7: la2 enc train600, oracle enc en2zh a; each must
#   leave 600 rows) -> touch the watcher's STOP file (cards back to the guard) -> wave17_mech_score.py -> wait for wave
#   14's eval_unist_wise05/EVAL_DONE (stop on its EVAL_FAILED / CHAIN_FAILED), then up to 2 h more for wave14
#   CHAIN_DONE (its summary.json is the scorer's cross-check) -> wave17_score.py -> CHAIN_DONE (or CHAIN_FAILED).
# Never touches guard 2726951, gpu_watch 52045, sync 1222453 or cards 0-5; stops nothing itself.
#   TPID=<torchrun pid> setsid nohup bash wave17_chain.sh > logs/mist_w3po_v1/wave17/chain.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${TPID:?}"
M=eval_outputs/mist_w3po_v1/wave17
W14=eval_outputs/mist_w3po_v1/wave14
LOGD=logs/mist_w3po_v1/wave17
STOP=$LOGD/w17_noplaceholder.stop
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
CA=6; CB=7
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; echo "$*" > "$M/CHAIN_FAILED"; touch "$STOP"; exit 1; }
E=experiments/mist_w3po_v1/voicecond/wave17_eval.sh
launch() { local o; o=$(env "$@" bash "$E" 2>&1) || { log "launch $*: $o"; return 1; }; echo "$o" | sed 's/^/  /' >&2; echo "$o" | grep -o 'pid [0-9]*$' | awk '{print $2}'; }
waitp() { for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done; }
mjob() {   # card sys policy dir set
  local f p n
  [ "$3" = oracle ] && f=blocks_$4_w1_oraclecommit.jsonl || f=blocks_$4_w1.jsonl
  log "mech start $3 $2 $5 $4 on card $1"
  p=$(launch STAGE=mech GPU=$1 SYS=$2 POLICY=$3 DIR=$4 MSET=$5) && [ -n "$p" ] || { echo "$3 $2 $5 $4: launch" >> "$M/MECH_FAILED"; return 1; }
  waitp $p
  n=$(wc -l < "$M/${3}_${2}_$5/thinker_$4/$f" 2>/dev/null || echo 0)
  [ "$n" = 600 ] || { log "mech $3 $2 $5 $4: $n of 600 rows"; echo "$3 $2 $5 $4: $n rows" >> "$M/MECH_FAILED"; return 1; }
  log "mech done $3 $2 $5 $4 ($n rows)"
}
log "chain $$: waiting for trainer $TPID"
while kill -0 "$TPID" 2>/dev/null; do sleep 30; done
R=$M/run_enc
[ -s "$R/adapters_final.pt" ] && [ -s "$R/adapters_final.json" ] || fail "trainer exited without adapters_final.pt"
log "training done; WiSE 0.5 copy"
W=$M/run_enc_wise05
[ -e "$W" ] && fail "$W exists"
mkdir -p "$W"
ln "$R/adapters_final.pt" "$W/adapters_final.pt" || fail "hard link"
"$PYQ" - "$R/adapters_final.json" "$W/adapters_final.json" <<'PY' || fail "wise json"
import json, sys
m = json.load(open(sys.argv[1]))
assert m["arm"] == "enc" and m["module"] == "thinker.audio_tower" and m["alpha"] == 32.0 and m["rank"] == 16 \
    and m["steps"] == 2300 and m["world"] == 2, m
m.update({"arm": "enc_wise05", "alpha": m["alpha"] / 2,
          "wise_ft": "W0 + 0.5*(W_enc - W0): LoRA alpha halved, adapters = run_enc/adapters_final.pt"})
with open(sys.argv[2], "x") as fh:
    json.dump(m, fh)
print(json.dumps(m))
PY
O=experiments/mist_w3po_v1/voicecond/wave17_eval_orch.sh
log "evaluation enc_wise05 on cards $CA,$CB"
ARM=enc_wise05 CA=$CA CB=$CB CC=$CA CD=$CB bash $O > $LOGD/eval_orch_enc_wise05.log 2>&1 < /dev/null
log "enc_wise05 chain exit $?"
[ -e "$M/eval_enc_wise05/EVAL_DONE" ] || fail "enc_wise05 evaluation"
log "evaluation enc on cards $CA,$CB"
ARM=enc CA=$CA CB=$CB CC=$CA CD=$CB bash $O > $LOGD/eval_orch_enc.log 2>&1 < /dev/null
log "enc chain exit $?"
[ -e "$M/eval_enc/EVAL_DONE" ] || log "WARNING: enc (alpha 1) evaluation incomplete"
log "mechanism jobs"
laneA() {
  mjob $CA base la2 zh2en train600
  mjob $CA enc oracle zh2en a
  if [ -s "$W14/merged_unist/model.safetensors.index.json" ]; then mjob $CA unist la2 zh2en train600
  else log "wave14/merged_unist absent: unist train600 control skipped"; fi
}
laneB() { mjob $CB enc la2 zh2en train600; mjob $CB enc oracle en2zh a; }
laneA & la=$!
sleep 30
laneB & lb=$!
wait $la; wait $lb
touch "$STOP"
log "watcher STOP touched; cards 6,7 back to the guard"
if [ -e "$M/MECH_FAILED" ]; then log "WARNING: some mechanism jobs failed (see $M/MECH_FAILED); mechanism scoring may fail"; fi
"$PYQ" experiments/mist_w3po_v1/voicecond/wave17_mech_score.py --out "$M/mech.json" > $LOGD/mech_score.log 2>&1 \
  && log "mechanism scoring done" || log "WARNING: mechanism scoring failed (see $LOGD/mech_score.log)"
log "waiting for wave 14 unist_wise05 dev outputs"
while [ ! -e "$W14/eval_unist_wise05/EVAL_DONE" ]; do
  [ -e "$W14/eval_unist_wise05/EVAL_FAILED" ] && fail "wave 14 unist_wise05 evaluation failed"
  [ -e "$W14/CHAIN_FAILED" ] && fail "wave 14 chain failed: $(cat $W14/CHAIN_FAILED)"
  sleep 60
done
log "wave 14 unist_wise05 EVAL_DONE; waiting up to 2 h for wave 14 CHAIN_DONE (summary.json cross-check)"
t0=$(date +%s)
while [ ! -e "$W14/CHAIN_DONE" ] && [ ! -e "$W14/CHAIN_FAILED" ] && [ $(( $(date +%s) - t0 )) -lt 7200 ]; do sleep 60; done
log "scoring"
"$PYQ" experiments/mist_w3po_v1/voicecond/wave17_score.py --out "$M/summary.json" > $LOGD/score.log 2>&1 || fail "score (see $LOGD/score.log)"
date > "$M/CHAIN_DONE"
log "chain done"
