#!/usr/bin/env bash
# Reduced end of the tier table runs (deadline: GPU work done by 10:15 UTC, server gone at 11:30 UTC). run_tier2.sh and its
# speech queue were stopped by PID at 08:41 UTC with 13/38 done and 2 running; this queue speaks 15 of the 23 points left
# (jobs/tier3_speech_jobs.json: Zh->En first; dropped: En->Zh b0=2 @5 s, Zh->En b0=1 @0.6/0.9/1.1/1.75/5 s, b0=2 @4/5 s).
# A guard stops the queue (by PID) at 10:08 UTC; then StreamLAAL scoring of ACL LA2 0.2-0.4 s (CPU) and the holder on 0-3.
# Log: logs/si_v4/tier3.log
set -u
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
LOG=logs/si_v4/tier3.log
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
EPY=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
mkdir -p logs/si_v4/tier3_speech && touch logs/si_v4/tier3_speech/allow_empty
env OUR_MAX=120000 GPUQ_CARDS=0,1,2,3 $EPY -u scripts/gpu_queue/gpuq7.py reports/uniss_si_v4/jobs/tier3_speech_jobs.json \
  logs/si_v4/tier3_speech > logs/si_v4/tier3_speech_queue.log 2>&1 &
Q=$!
log "speech queue pid $Q"
until ! kill -0 $Q 2>/dev/null || [ "$(date -u +%H%M)" -ge 1008 ]; do sleep 20; done
if kill -0 $Q 2>/dev/null; then
  kill $Q; log "deadline: queue $Q stopped; running jobs:"
  for p in $(ps -eo pid,args | awk '$2=="bash" && $3=="-c" && /OMP_NUM_THREADS=4/ {print $1}'); do
    log "  stop $p"; pkill -TERM -P $p; kill $p
  done
fi
log "speech queue: $(tail -1 logs/si_v4/tier3_speech_queue.log)"
A=eval_outputs/uniss_qwen3_omni_v1/acl6060_dev_v1_la2low
PYTHONPATH=. $EPY -m experiments.uniss_qwen3_omni_v1.evaluation.si_streamlaal score --refs data/acl6060_dev_v1/refs.jsonl \
  --items data/acl6060_dev_v1/items.jsonl --blocks "$A/blocks_en2zh_*.jsonl" --dump $A/seg --procs 48 2>&1 | grep -v Warn \
  > $A/scores.jsonl
log "ACL la2low scored: $(grep -c '^{' $A/scores.jsonl) runs"
for p in 1897382 1897711; do while [ -e /proc/$p ]; do sleep 15; done; done
sleep 20
bash scripts/gpu_queue/holder_on.sh 0,1,2,3 >> $LOG 2>&1
log "TIER3_DONE"
