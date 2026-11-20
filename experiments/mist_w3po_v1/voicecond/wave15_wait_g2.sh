#!/usr/bin/env bash
# Robust wave 15 group 2 (detached; reports/mist_w3po_v1/robust_wave15_cvla_prereg.zh-CN.md): waits for group 1
# (CHAIN_G1_DONE; CHAIN_G1_FAILED -> stop) and for wave 14's merge (wave14/merged_unist_wise05/model.safetensors.index.json
# and '"adapted"' in logs/mist_w3po_v1/wave14/w14_eval_merge_unist_wise05.log, the check wave14_eval_orch.sh makes;
# wave14 CHAIN_FAILED / eval_unist_wise05/EVAL_FAILED -> stop) -> own placeholder watcher (STOP g2) -> the cvla_unist chain
# on cards 4, 5 (wave15_eval_orch.sh) -> touch STOP g2 -> waits for wave14/eval_unist_wise05/EVAL_DONE (wave 14's own LA2
# run of unist_wise05) -> wave15_score.py --group 2 -> wave15_commit.py --group 2 -> CHAIN_G2_DONE (or CHAIN_G2_FAILED).
#   setsid nohup bash wave15_wait_g2.sh > logs/mist_w3po_v1/wave15/chain_g2.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
M=eval_outputs/mist_w3po_v1/wave15
W14=eval_outputs/mist_w3po_v1/wave14
LOGD=logs/mist_w3po_v1/wave15
STOP=$LOGD/w15_noplaceholder_g2.stop
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; echo "$*" > "$M/CHAIN_G2_FAILED"; [ -n "${WATCH:-}" ] && touch "$STOP"; exit 1; }
[ -e "$STOP" ] && { echo "STOP file exists: $STOP" >&2; exit 1; }
for f in CHAIN_G2_DONE CHAIN_G2_FAILED summary_g2.json; do [ -e "$M/$f" ] && { echo "exists: $M/$f" >&2; exit 1; }; done
log "wait g2 $$: group 1, then wave 14's merged_unist_wise05"
until [ -e "$M/CHAIN_G1_DONE" ]; do
  [ -e "$M/CHAIN_G1_FAILED" ] && fail "group 1 failed: $(cat $M/CHAIN_G1_FAILED)"
  sleep 60
done
log "group 1 done"
until [ -s "$W14/merged_unist_wise05/model.safetensors.index.json" ] && grep -q '"adapted"' logs/mist_w3po_v1/wave14/w14_eval_merge_unist_wise05.log 2>/dev/null; do
  [ -e "$W14/CHAIN_FAILED" ] && fail "wave 14 chain failed: $(cat $W14/CHAIN_FAILED)"
  [ -e "$W14/eval_unist_wise05/EVAL_FAILED" ] && fail "wave 14 unist_wise05 failed: $(cat $W14/eval_unist_wise05/EVAL_FAILED)"
  sleep 60
done
log "wave 14 merge of unist_wise05 done"
WATCH=1
STOP=$STOP setsid nohup bash experiments/mist_w3po_v1/voicecond/wave15_noplaceholder.sh > $LOGD/noplaceholder_g2.log 2>&1 < /dev/null 8>&- 9>&- &
log "watcher g2 pid $!"
ARM=cvla_unist CA=4 CB=5 bash experiments/mist_w3po_v1/voicecond/wave15_eval_orch.sh > $LOGD/eval_orch_cvla_unist.log 2>&1 < /dev/null
r=$?
touch "$STOP"
log "cvla_unist chain exit $r; watcher g2 STOP touched, cards 4-5 back to the guard"
[ -e "$M/eval_cvla_unist/EVAL_DONE" ] || fail "cvla_unist evaluation"
until [ -e "$W14/eval_unist_wise05/EVAL_DONE" ]; do
  [ -e "$W14/eval_unist_wise05/EVAL_FAILED" ] && fail "wave 14 unist_wise05 evaluation failed"
  [ -e "$W14/CHAIN_FAILED" ] && fail "wave 14 chain failed: $(cat $W14/CHAIN_FAILED)"
  sleep 60
done
log "wave 14 unist_wise05 evaluation done; scoring"
"$PYQ" experiments/mist_w3po_v1/voicecond/wave15_score.py --group 2 --out $M/summary_g2.json > $LOGD/score_g2.log 2>&1 || fail "score g2 (see $LOGD/score_g2.log)"
PYTHONPATH=. "$PYE" experiments/mist_w3po_v1/voicecond/wave15_commit.py --group 2 --out $M/commit_g2.json > $LOGD/commit_g2.log 2>&1 || fail "commit g2"
date > "$M/CHAIN_G2_DONE"
log "chain g2 done"
