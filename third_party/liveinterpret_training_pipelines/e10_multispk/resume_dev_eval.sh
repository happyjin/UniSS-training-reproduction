#!/usr/bin/env bash
# E10 dev evaluations after the training (resume of takeover_slotA_train.sh steps 4-6, 2026-11-19 ~14:05 Prague).
# takeover picked checkpoint-750 by a lexical sort (fixed: sort -V); that E10 dev run was stopped by PID and set aside,
# and the E10-0 En->Zh dev run lost one shard to a guard placeholder (OOM) after the training ended.
#   1. wait for the still-running E10-0 decode shards (WAIT_PIDS) to finish;
#   2. E10-0 deve (resume: complete shards kept, incomplete ones set aside and redone);
#   3. E10 devz + deve with the last checkpoint (LAST);
#   4. dev gate (dev_gate.py); 5. release slot A; marker E10_DEVGATE_<PASS|FAIL|MISSING>.
# Placeholders on 0-3 are stopped by PID before each launch (eval_e10.py --stop-placeholders).
# Run detached: setsid nohup bash e10_multispk/resume_dev_eval.sh > <log> 2>&1 < /dev/null 9>&- &
set -uo pipefail
U=/opt/dlami/nvme/jasonleeeli/projects/UniSS
LI=$U/third_party/liveinterpret_training_pipelines
E=$LI/e10_multispk
SL=$U/logs/mist_w3po_v1/slots
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
MK=$E/output/markers
INIT=$LI/checkpoints/p4mix_2n_20261117/checkpoint-29773
: "${LAST:?LAST checkpoint required}" "${WAIT_PIDS:=}"
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
release() { [ -e "$SL/slotA.owner" ] && command mv -f "$SL/slotA.owner" "$SL/slotA.released_$(date -u +%Y%m%dT%H%M%SZ)" \
            && log "slot A released"; }
[ -e "$LAST/e10_metadata.json" ] || { log "no E10 adapter at $LAST"; exit 1; }
[ -e "$SL/slotA.owner" ] || { log "slot A not held; refusing"; exit 1; }
echo "resumed: $(date -u +%FT%TZ) pid $$ (resume_dev_eval.sh, dev evaluations with $LAST)" >> "$SL/slotA.owner"
cd "$U"
for p in $WAIT_PIDS; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done
log "orphan E10-0 shards finished"
"$PYE" "$E/eval_e10.py" --run --stop-placeholders --mode e10_0 --adapter "$INIT" --sets deve --seeds first \
    --gpus 0,1,2,3 --procs-per-gpu 2 || log "E10-0 deve evaluation failed (gate 3/4 for deve will be MISSING)"
"$PYE" "$E/eval_e10.py" --run --stop-placeholders --mode e10 --adapter "$LAST" --sets devz,deve --seeds first \
    --gpus 0,1,2,3 --procs-per-gpu 4 || log "E10 dev evaluation failed"
"$PYE" "$E/dev_gate.py" --sets devz,deve --out "$E/output/eval/dev_gate.json"
rc=$?
case $rc in 0) m=PASS;; 3) m=FAIL;; *) m=MISSING;; esac
mkdir -p "$MK"; echo "$LAST rc=$rc" > "$MK/E10_DEVGATE_$m"
release
log "dev gate: $m"
