#!/usr/bin/env bash
# Tier table of paper v7, part 2 (make_tier2_jobs.py), cards 0-3, after the first tier queue (jobs/tier_jobs.json,
# logs/si_v4/tier_queue.log) has started all its jobs (two gpuq7 queues must not start engines on the same transition):
#   1. QABC with adapter s1 (chosen on dev) on RealSI text, 15 windows x b0 {-1,0,1,2}   (gpuq7, logs/si_v4/tier2_text)
#   2. the Talker speaks the s1 points, talker seed 0                                   (gpuq7, logs/si_v4/tier2_speech)
#   3. when the first tier queue is done: StreamLAAL scoring of LA2 at 0.2-0.4 s on ACL60/60 (CPU)
#   4. when everything of this branch on cards 0-3 is done (both tier queues, CVSS-T metrics): GPU holder on 0-3
# Log: logs/si_v4/tier2.log
set -u
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
LOG=logs/si_v4/tier2.log
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
EPY=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
Q="env OUR_MAX=120000 GPUQ_CARDS=0,1,2,3 $EPY -u scripts/gpu_queue/gpuq7.py"
NJ=$(python3 -c "import json;print(len(json.load(open('reports/uniss_si_v4/jobs/tier_jobs.json'))))")

log "waiting for the first tier queue to start all $NJ jobs"
until [ "$(grep -c ' start ' logs/si_v4/tier_queue.log)" -ge "$NJ" ] || grep -q QUEUE_DONE logs/si_v4/tier_queue.log; do sleep 30; done
sleep 150
mkdir -p logs/si_v4/tier2_text logs/si_v4/tier2_speech
touch logs/si_v4/tier2_text/allow_empty logs/si_v4/tier2_speech/allow_empty
log "text queue"
$Q reports/uniss_si_v4/jobs/tier2_text_jobs.json logs/si_v4/tier2_text > logs/si_v4/tier2_text_queue.log 2>&1
for d in en2zh zh2en; do
  n=$(ls eval_outputs/uniss_qwen3_omni_v1/si_v4/realsi_grid_s1/blocks_${d}_w*_sws1b*.jsonl 2>/dev/null | wc -l)
  log "realsi_grid_s1 $d: $n block files"
done
python3 reports/uniss_si_v4/make_tier2_jobs.py speech reports/uniss_si_v4/jobs/tier2_speech_jobs.json >> $LOG 2>&1 \
  || { log "speech jobs not built"; exit 1; }
log "speech queue"
$Q reports/uniss_si_v4/jobs/tier2_speech_jobs.json logs/si_v4/tier2_speech > logs/si_v4/tier2_speech_queue.log 2>&1
log "speech queue: $(tail -1 logs/si_v4/tier2_speech_queue.log)"

until grep -q QUEUE_DONE logs/si_v4/tier_queue.log; do sleep 60; done
log "first tier queue: $(tail -1 logs/si_v4/tier_queue.log)"
A=eval_outputs/uniss_qwen3_omni_v1/acl6060_dev_v1_la2low
PYTHONPATH=. $EPY -m experiments.uniss_qwen3_omni_v1.evaluation.si_streamlaal score --refs data/acl6060_dev_v1/refs.jsonl \
  --items data/acl6060_dev_v1/items.jsonl --blocks "$A/blocks_en2zh_*.jsonl" --dump $A/seg --procs 48 2>&1 | grep -v Warn \
  > $A/scores.jsonl
log "ACL la2low scored: $(grep -c '^{' $A/scores.jsonl) runs"

until grep -q CVSS_TABLE1_DONE logs/si_v4/cvss_table1.log; do sleep 60; done
bash scripts/gpu_queue/holder_on.sh 0,1,2,3 >> $LOG 2>&1
log "TIER2_DONE"
