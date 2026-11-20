#!/usr/bin/env bash
# Robust wave 8 AUL: everything after training, unattended. Waits for the trainer TPID to exit, checks
# run_aul/adapters_final.pt, writes run_aul_wise05 (adapters_final.pt -> ../run_aul/adapters_final.pt, the json with alpha
# halved: W0 + 0.5 delta, as wave 5's run_erm_wise05), runs wave8_aul_eval_orch.sh for aul_wise05 on cards 4,5 and for
# aul (auxiliary) on cards 6,7 in parallel, waits for both, then the scorer (wave8_aul_score.py -> summary.json) and the
# commit statistic (wave8_aul_commit.py -> commit_stats.json); writes CHAIN_DONE or CHAIN_FAILED in the root.
#   TPID=<trainer pid> bash wave8_aul_chain.sh      (itself run detached with setsid nohup)    Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${TPID:?}"
M=eval_outputs/mist_w3po_v1/wave8/AUL
LOGD=logs/mist_w3po_v1/wave8
O=experiments/mist_w3po_v1/voicecond/wave8_aul_eval_orch.sh
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; echo "$*" > "$M/CHAIN_FAILED"; exit 1; }
while kill -0 "$TPID" 2>/dev/null; do sleep 60; done
[ -s "$M/run_aul/adapters_final.pt" ] && [ -s "$M/run_aul/adapters_final.json" ] || fail "no final adapters in run_aul"
log "training finished"
W=$M/run_aul_wise05
[ -e "$W" ] && fail "$W exists (refusing to overwrite)"
mkdir -p "$W"
ln -s ../run_aul/adapters_final.pt "$W/adapters_final.pt"
/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python - "$M/run_aul/adapters_final.json" "$W/adapters_final.json" <<'EOF' || fail "wise json"
import json, sys
m = json.load(open(sys.argv[1]))
m.update(arm="aul_wise05", alpha=m["alpha"] / 2,
         wise_ft="W0 + 0.5*(W_aul - W0): LoRA alpha halved, adapters = run_aul/adapters_final.pt")
with open(sys.argv[2], "x") as fh:
    json.dump(m, fh)
EOF
log "run_aul_wise05 written; evaluation chains"
setsid nohup env ARM=aul_wise05 CA=4 CB=5 TPID=$TPID bash "$O" > "$LOGD/aul_eval_orch_aul_wise05.log" 2>&1 < /dev/null 8>&- 9>&- &
p1=$!
sleep 30   # the two merges read the same release shards; stagger them a little
setsid nohup env ARM=aul CA=6 CB=7 TPID=$TPID bash "$O" > "$LOGD/aul_eval_orch_aul.log" 2>&1 < /dev/null 8>&- 9>&- &
p2=$!
echo "none $p1 wave8_aul_orch_aul_wise05" >> "$LOGD/aul_workers.pids"
echo "none $p2 wave8_aul_orch_aul" >> "$LOGD/aul_workers.pids"
while kill -0 "$p1" 2>/dev/null || kill -0 "$p2" 2>/dev/null; do sleep 60; done
[ -e "$M/eval_aul_wise05/EVAL_DONE" ] || fail "eval aul_wise05: $(cat "$M/eval_aul_wise05/EVAL_FAILED" 2>/dev/null)"
[ -e "$M/eval_aul/EVAL_DONE" ] || log "WARNING: auxiliary eval of aul failed: $(cat "$M/eval_aul/EVAL_FAILED" 2>/dev/null)"
log "score"
/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python experiments/mist_w3po_v1/voicecond/wave8_aul_score.py \
  --out "$M/summary.json" > "$LOGD/aul_score.log" 2>&1 || fail "score"
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python experiments/mist_w3po_v1/voicecond/wave8_aul_commit.py \
  --out "$M/commit_stats.json" > "$LOGD/aul_commit.log" 2>&1 || fail "commit stats"
date > "$M/CHAIN_DONE"
log "chain done"
