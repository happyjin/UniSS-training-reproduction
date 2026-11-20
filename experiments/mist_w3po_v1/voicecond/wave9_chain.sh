#!/usr/bin/env bash
# Robust wave 9: everything after arm A's training, unattended (modelled on wave8_aul_chain.sh). Waits for the trainer
# TPID, checks run_A/adapters_final.pt, writes run_A_wise05 (adapters_final.pt -> ../run_A/adapters_final.pt; json with
# LoRA alpha halved and anchor_wise 0.5: W0 + 0.5 (W_A - W0) and E0 + 0.5 (E_A - E0)), runs wave9_eval_orch.sh for
# A_wise05 on cards CA, CB, then the scorer (wave9_score.py -> summary.json); writes CHAIN_DONE or CHAIN_FAILED in the root.
#   TPID=<trainer pid> [CA=0 CB=1] bash wave9_chain.sh      (itself run detached with setsid nohup)   Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${TPID:?}"
CA=${CA:-0}; CB=${CB:-1}
M=eval_outputs/mist_w3po_v1/wave9
LOGD=logs/mist_w3po_v1/wave9
O=experiments/mist_w3po_v1/voicecond/wave9_eval_orch.sh
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; echo "$*" > "$M/CHAIN_FAILED"; exit 1; }
while kill -0 "$TPID" 2>/dev/null; do sleep 60; done
[ -s "$M/run_A/adapters_final.pt" ] && [ -s "$M/run_A/adapters_final.json" ] || fail "no final adapters in run_A"
log "training finished"
W=$M/run_A_wise05
[ -e "$W" ] && fail "$W exists (refusing to overwrite)"
mkdir -p "$W"
ln -s ../run_A/adapters_final.pt "$W/adapters_final.pt"
/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python - "$M/run_A/adapters_final.json" "$W/adapters_final.json" <<'PY' || fail "wise json"
import json, sys
m = json.load(open(sys.argv[1]))
assert m["anchor"]
m.update(arm="A_wise05", alpha=m["alpha"] / 2, anchor_wise=0.5,
         wise_ft="W0 + 0.5*(W_A - W0): LoRA alpha halved; anchor rows E0 + 0.5 (E_A - E0); adapters = run_A/adapters_final.pt")
with open(sys.argv[2], "x") as fh:
    json.dump(m, fh)
PY
log "run_A_wise05 written; evaluation chain"
setsid nohup env ARM=A_wise05 CA=$CA CB=$CB TPID=$TPID bash "$O" > "$LOGD/w9_orch_A_wise05.log" 2>&1 < /dev/null 8>&- 9>&- &
p1=$!
echo "none $p1 wave9_orch_A_wise05" >> "$LOGD/w9_workers.pids"
while kill -0 "$p1" 2>/dev/null; do sleep 60; done
[ -e "$M/eval_A_wise05/EVAL_DONE" ] || fail "eval A_wise05: $(cat "$M/eval_A_wise05/EVAL_FAILED" 2>/dev/null)"
log "score"
/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python experiments/mist_w3po_v1/voicecond/wave9_score.py \
  --out "$M/summary.json" > "$LOGD/w9_score.log" 2>&1 || fail "score"
date > "$M/CHAIN_DONE"
log "chain done"
