#!/usr/bin/env bash
# Robust wave 14 post-training chain (detached; reports/mist_w3po_v1/robust_wave14_unist_prereg.zh-CN.md):
#   wait for the trainer TPID (torchrun of run_unist) -> check run_unist/adapters_final.pt -> WiSE 0.5 copy
#   run_unist_wise05 (hard link of the adapters + json with alpha 16, wave5/M/run_erm_wise05 format) -> two evaluation
#   chains in parallel (wave14_eval_orch.sh): unist_wise05 on cards 0,1, unist on cards 2,3 -> once both have exited,
#   touch the watcher's STOP file (cards back to the guard) -> wave14_score.py -> CHAIN_DONE (or CHAIN_FAILED).
# Never touches guard 2726951, gpu_watch 52045, sync 1222453; stops nothing itself.
#   TPID=<torchrun pid> setsid nohup bash wave14_chain.sh > logs/mist_w3po_v1/wave14/chain.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${TPID:?}"
M=eval_outputs/mist_w3po_v1/wave14
LOGD=logs/mist_w3po_v1/wave14
STOP=$LOGD/w14_noplaceholder.stop
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; echo "$*" > "$M/CHAIN_FAILED"; touch "$STOP"; exit 1; }
log "chain $$: waiting for trainer $TPID"
while kill -0 "$TPID" 2>/dev/null; do sleep 30; done
R=$M/run_unist
[ -s "$R/adapters_final.pt" ] && [ -s "$R/adapters_final.json" ] || fail "trainer exited without adapters_final.pt"
log "training done; WiSE 0.5 copy"
W=$M/run_unist_wise05
[ -e "$W" ] && fail "$W exists"
mkdir -p "$W"
ln "$R/adapters_final.pt" "$W/adapters_final.pt" || fail "hard link"
"$PYQ" - "$R/adapters_final.json" "$W/adapters_final.json" <<'PY' || fail "wise json"
import json, sys
m = json.load(open(sys.argv[1]))
assert m["arm"] == "erm" and m["alpha"] == 32.0 and m["rank"] == 16 and m["steps"] == 2300, m
m.update({"arm": "unist_wise05", "alpha": m["alpha"] / 2,
          "wise_ft": "W0 + 0.5*(W_unist - W0): LoRA alpha halved, adapters = run_unist/adapters_final.pt"})
with open(sys.argv[2], "x") as fh:
    json.dump(m, fh)
print(json.dumps(m))
PY
O=experiments/mist_w3po_v1/voicecond/wave14_eval_orch.sh
ARM=unist_wise05 CA=0 CB=1 CC=0 CD=1 bash $O > $LOGD/eval_orch_unist_wise05.log 2>&1 < /dev/null &
o1=$!
sleep 30
ARM=unist CA=2 CB=3 CC=2 CD=3 bash $O > $LOGD/eval_orch_unist.log 2>&1 < /dev/null &
o2=$!
log "eval orchestrators: unist_wise05 pid $o1 (cards 0,1), unist pid $o2 (cards 2,3)"
wait $o1; r1=$?
log "unist_wise05 chain exit $r1"
wait $o2; r2=$?
log "unist chain exit $r2"
touch "$STOP"
log "watcher STOP touched; cards back to the guard"
[ -e "$M/eval_unist_wise05/EVAL_DONE" ] || fail "unist_wise05 evaluation"
[ -e "$M/eval_unist/EVAL_DONE" ] || log "WARNING: unist (alpha 1) evaluation incomplete; scoring without it"
log "scoring"
"$PYQ" experiments/mist_w3po_v1/voicecond/wave14_score.py --out "$M/summary.json" > $LOGD/score.log 2>&1 || fail "score (see $LOGD/score.log)"
date > "$M/CHAIN_DONE"
log "chain done"
