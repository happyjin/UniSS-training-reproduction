#!/usr/bin/env bash
# Robust wave 15 group 1 chain (detached; reports/mist_w3po_v1/robust_wave15_cvla_prereg.zh-CN.md), cards 4 (en2zh) and
# 5 (zh2en) only:
#   own placeholder watcher (STOP g1) -> in parallel: DeepFilterNet3 views en2zh + zh2en (CPU) and the self-check
#   Thinker same_base (v1 = v0) on cards 4, 5 -> wave15_selfcheck.py (FAIL: stop, CHAIN_G1_FAILED) -> the cvla_base
#   chain (wave15_eval_orch.sh) -> touch STOP g1 (cards back to the guard) -> wave15_score.py --group 1 ->
#   wave15_commit.py --group 1 -> CHAIN_G1_DONE.
# Never touches guard 2726951, gpu_watch 52045, sync 1222453; stops nothing itself (launches stop placeholders by PID).
#   setsid nohup bash wave15_chain_g1.sh > logs/mist_w3po_v1/wave15/chain_g1.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
M=eval_outputs/mist_w3po_v1/wave15
LOGD=logs/mist_w3po_v1/wave15
STOP=$LOGD/w15_noplaceholder_g1.stop
E=experiments/mist_w3po_v1/voicecond/wave15_eval.sh
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; echo "$*" > "$M/CHAIN_G1_FAILED"; touch "$STOP"; exit 1; }
launch() { local o; o=$(env "$@" bash "$E" 2>&1) || fail "launch $*: $o"; echo "$o" | sed 's/^/  /' >&2; echo "$o" | grep -o 'pid [0-9]*$' | awk '{print $2}'; }
waitp() { for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done; }
[ -e "$STOP" ] && { echo "STOP file exists: $STOP" >&2; exit 1; }
for f in CHAIN_G1_DONE CHAIN_G1_FAILED summary_g1.json selfcheck.json; do [ -e "$M/$f" ] && { echo "exists: $M/$f" >&2; exit 1; }; done
log "chain g1 $$"
STOP=$STOP setsid nohup bash experiments/mist_w3po_v1/voicecond/wave15_noplaceholder.sh > $LOGD/noplaceholder_g1.log 2>&1 < /dev/null 8>&- 9>&- &
log "watcher g1 pid $!"
v1=$(launch STAGE=views GPU=- ARM=views DIR=en2zh); v2=$(launch STAGE=views GPU=- ARM=views DIR=zh2en)
s1=$(launch STAGE=thinker GPU=4 ARM=same_base DIR=en2zh); sleep 30; s2=$(launch STAGE=thinker GPU=5 ARM=same_base DIR=zh2en)
waitp $s1 $s2
log "self-check"
"$PYQ" experiments/mist_w3po_v1/voicecond/wave15_selfcheck.py --run $M/eval_same_base --out $M/selfcheck.json > $LOGD/selfcheck.log 2>&1 \
  || fail "self-check (v1 = v0) did not reproduce LA2; see $LOGD/selfcheck.log"
log "self-check PASS"
waitp $v1 $v2
for d in en2zh zh2en; do [ -s "$M/views/$d/DONE.json" ] || fail "views $d"; done
log "cvla_base chain"
ARM=cvla_base CA=4 CB=5 bash experiments/mist_w3po_v1/voicecond/wave15_eval_orch.sh > $LOGD/eval_orch_cvla_base.log 2>&1 < /dev/null
r=$?
touch "$STOP"
log "cvla_base chain exit $r; watcher g1 STOP touched, cards 4-5 back to the guard"
[ -e "$M/eval_cvla_base/EVAL_DONE" ] || fail "cvla_base evaluation"
"$PYQ" experiments/mist_w3po_v1/voicecond/wave15_score.py --group 1 --out $M/summary_g1.json > $LOGD/score_g1.log 2>&1 || fail "score g1 (see $LOGD/score_g1.log)"
PYTHONPATH=. "$PYE" experiments/mist_w3po_v1/voicecond/wave15_commit.py --group 1 --out $M/commit_g1.json > $LOGD/commit_g1.log 2>&1 || fail "commit g1"
date > "$M/CHAIN_G1_DONE"
log "chain g1 done"
