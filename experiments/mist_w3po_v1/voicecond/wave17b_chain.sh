#!/usr/bin/env bash
# Robust wave 17b full chain (reports/mist_w3po_v1/robust_wave17b_enclora_covost_prereg.zh-CN.md), detached:
#   1. wait until cards 6,7 are free of wave 17: wave17's watcher STOP (logs/mist_w3po_v1/wave17/w17_noplaceholder.stop)
#      or wave17 CHAIN_DONE / CHAIN_FAILED exists, AND for 3 consecutive polls (60 s apart) every compute process on cards
#      6 and 7 is a no-checkpoint placeholder (scripts/gpu_load/) or there is none (nvidia-smi + /proc argv);
#   2. own watcher wave17b_noplaceholder.sh (own PIDS / STOP);
#   3. smoke: wave17b_run.sh SMOKE=6 MB=32 on 6,7 (out wave17b/smoke32); pass = 'smoke stop after 6 steps', no OOM; else MB=16
#      (wave17b/smoke16); both fail -> CHAIN_FAILED;
#   4. training run_enc: wave17b_run.sh GPUS=6,7 STEPS=2300 MB=<smoke> NAME=enc (world 2);
#   5. as wave17_chain.sh: WiSE 0.5 copy run_enc_wise05 (hard link + json alpha 16) -> eval enc_wise05 on 6,7 -> eval enc on
#      6,7 (wave17b_eval_orch.sh) -> mechanism Thinker jobs (lane 6: oracle enc zh2en a; lane 7: la2 enc zh2en train600
#      (wave 13's CoVoST clips), oracle enc en2zh a; base / erm train600 reused from wave 13) -> touch STOP (cards back to
#      the guard) -> wave17b_mech_score.py -> wave17b_score.py -> CHAIN_DONE (any failure: CHAIN_FAILED + STOP).
# Cards 6,7 only; never cards 0-5; never touches guard 2726951, gpu_watch 52045, sync 1222453 or any non-placeholder job.
#   setsid nohup bash wave17b_chain.sh > logs/mist_w3po_v1/wave17b/chain.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
M=eval_outputs/mist_w3po_v1/wave17b
W17=eval_outputs/mist_w3po_v1/wave17
LOGD=logs/mist_w3po_v1/wave17b
STOP=$LOGD/w17b_noplaceholder.stop
PIDS=$LOGD/w17b_workers.pids
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
V=experiments/mist_w3po_v1/voicecond
CA=6; CB=7
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; echo "$*" > "$M/CHAIN_FAILED"; touch "$STOP"; exit 1; }
mkdir -p "$M" "$LOGD"
[ -e "$STOP" ] && { log "STOP file exists: $STOP"; exit 1; }
for f in CHAIN_DONE CHAIN_FAILED run_enc run_enc_wise05 smoke32 smoke16 summary.json mech.json; do
  [ -e "$M/$f" ] && { log "exists: $M/$f"; exit 1; }
done
E=$V/wave17b_eval.sh
launch() { local o; o=$(env "$@" bash "$E" 2>&1) || { log "launch $*: $o"; return 1; }; echo "$o" | sed 's/^/  /' >&2; echo "$o" | grep -o 'pid [0-9]*$' | awk '{print $2}'; }
waitp() { for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done; }
cards_free() {   # 0 iff every compute process on cards 6 and 7 is a gpu_load placeholder (or none)
  local idx uuid pid a
  declare -A U
  while IFS=', ' read -r idx uuid; do U[$uuid]=$idx; done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
  while IFS=', ' read -r uuid pid; do
    [ -n "${pid:-}" ] || continue
    case "${U[$uuid]:-x}" in $CA|$CB) ;; *) continue;; esac
    a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
    case "$a" in *scripts/gpu_load/*) ;; *) return 1;; esac
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)
  return 0
}
log "chain $$: waiting for wave 17 to release cards $CA,$CB"
ok=0
while :; do
  if [ -e logs/mist_w3po_v1/wave17/w17_noplaceholder.stop ] || [ -e "$W17/CHAIN_DONE" ] || [ -e "$W17/CHAIN_FAILED" ]; then
    if cards_free; then ok=$((ok + 1)); else ok=0; fi
    [ "$ok" -ge 3 ] && break
  fi
  sleep 60
done
log "cards $CA,$CB free of wave 17 (placeholders only)"
STOP=$STOP PIDS=$PIDS setsid nohup bash $V/wave17b_noplaceholder.sh > $LOGD/noplaceholder.log 2>&1 < /dev/null 8>&- 9>&- &
log "watcher pid $!"
trainpid() { echo "$1" | grep -o 'pid [0-9]*$' | awk '{print $2}'; }
MB=""
for mb in 32 16; do
  o=$(GPUS=$CA,$CB STEPS=2300 OUT=$M/smoke$mb NAME=smoke$mb MB=$mb SMOKE=6 PIDS=$PIDS bash $V/wave17b_run.sh 2>&1) || fail "smoke launch: $o"
  echo "$o" | sed 's/^/  /'
  p=$(trainpid "$o"); [ -n "$p" ] || fail "smoke pid"
  waitp "$p"
  if grep -q 'smoke stop after 6 steps' $LOGD/w17b_smoke$mb.log && ! grep -qi 'out of memory' $LOGD/w17b_smoke$mb.log; then
    MB=$mb; log "smoke mb $mb PASS: $(grep -m2 'mem ' $LOGD/w17b_smoke$mb.log | tail -n1 | cut -c1-120)"; break
  fi
  log "smoke mb $mb failed (see $LOGD/w17b_smoke$mb.log)"
  sleep 30
done
[ -n "$MB" ] || fail "smoke failed at mb 32 and 16"
sleep 20
o=$(GPUS=$CA,$CB STEPS=2300 OUT=$M/run_enc NAME=enc MB=$MB PIDS=$PIDS bash $V/wave17b_run.sh 2>&1) || fail "train launch: $o"
echo "$o" | sed 's/^/  /'
TPID=$(trainpid "$o"); [ -n "$TPID" ] || fail "train pid"
log "training run_enc (mb $MB) torchrun pid $TPID"
waitp "$TPID"
R=$M/run_enc
[ -s "$R/adapters_final.pt" ] && [ -s "$R/adapters_final.json" ] || fail "trainer exited without adapters_final.pt"
log "training done; WiSE 0.5 copy"
W=$M/run_enc_wise05
mkdir "$W" || fail "$W exists"
ln "$R/adapters_final.pt" "$W/adapters_final.pt" || fail "hard link"
"$PYQ" - "$R/adapters_final.json" "$W/adapters_final.json" <<'PY' || fail "wise json"
import json, sys
m = json.load(open(sys.argv[1]))
assert m["arm"] == "enc" and m["module"] == "thinker.audio_tower" and m["alpha"] == 32.0 and m["rank"] == 16 \
    and m["steps"] == 2300 and m["world"] == 2 and m["data"] == "eval_outputs/mist_w3po_v1/wave5/M/data", m
m.update({"arm": "enc_wise05", "alpha": m["alpha"] / 2,
          "wise_ft": "W0 + 0.5*(W_enc - W0): LoRA alpha halved, adapters = run_enc/adapters_final.pt"})
with open(sys.argv[2], "x") as fh:
    json.dump(m, fh)
print(json.dumps(m))
PY
O=$V/wave17b_eval_orch.sh
log "evaluation enc_wise05 on cards $CA,$CB"
ARM=enc_wise05 CA=$CA CB=$CB CC=$CA CD=$CB PIDS=$PIDS bash $O > $LOGD/eval_orch_enc_wise05.log 2>&1 < /dev/null
log "enc_wise05 chain exit $?"
[ -e "$M/eval_enc_wise05/EVAL_DONE" ] || fail "enc_wise05 evaluation"
log "evaluation enc on cards $CA,$CB"
ARM=enc CA=$CA CB=$CB CC=$CA CD=$CB PIDS=$PIDS bash $O > $LOGD/eval_orch_enc.log 2>&1 < /dev/null
log "enc chain exit $?"
[ -e "$M/eval_enc/EVAL_DONE" ] || log "WARNING: enc (alpha 1) evaluation incomplete"
log "mechanism jobs"
mjob() {   # card sys policy dir set
  local f p n
  [ "$3" = oracle ] && f=blocks_$4_w1_oraclecommit.jsonl || f=blocks_$4_w1.jsonl
  log "mech start $3 $2 $5 $4 on card $1"
  p=$(launch STAGE=mech GPU=$1 SYS=$2 POLICY=$3 DIR=$4 MSET=$5 PIDS=$PIDS) && [ -n "$p" ] || { echo "$3 $2 $5 $4: launch" >> "$M/MECH_FAILED"; return 1; }
  waitp $p
  n=$(wc -l < "$M/${3}_${2}_$5/thinker_$4/$f" 2>/dev/null || echo 0)
  [ "$n" = 600 ] || { log "mech $3 $2 $5 $4: $n of 600 rows"; echo "$3 $2 $5 $4: $n rows" >> "$M/MECH_FAILED"; return 1; }
  log "mech done $3 $2 $5 $4 ($n rows)"
}
laneA() { mjob $CA enc oracle zh2en a; }
laneB() { mjob $CB enc la2 zh2en train600; mjob $CB enc oracle en2zh a; }
laneA & la=$!
sleep 30
laneB & lb=$!
wait $la; wait $lb
touch "$STOP"
log "watcher STOP touched; cards $CA,$CB back to the guard"
[ -e "$M/MECH_FAILED" ] && log "WARNING: some mechanism jobs failed (see $M/MECH_FAILED)"
"$PYQ" $V/wave17b_mech_score.py --out "$M/mech.json" > $LOGD/mech_score.log 2>&1 \
  && log "mechanism scoring done" || log "WARNING: mechanism scoring failed (see $LOGD/mech_score.log)"
log "scoring"
"$PYQ" $V/wave17b_score.py --out "$M/summary.json" > $LOGD/score.log 2>&1 || fail "score (see $LOGD/score.log)"
date > "$M/CHAIN_DONE"
log "chain done"
