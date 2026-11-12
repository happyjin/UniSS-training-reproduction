#!/usr/bin/env bash
# Card 3: after queue_c (xt-c1280:zh2en, xt-c5120:en2zh) exits, run xt-c5120:zh2en with 2 shards there
# (queue_d could not restart its ASR server on card 6: the freed memory was taken by other jobs).
set -uo pipefail
R=/opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${QC_PID:?}"
while kill -0 "$QC_PID" 2>/dev/null; do sleep 10; done
grep -q QUEUE_DONE "$R/eval_outputs/baselines_xtranslator_v1/logs/queue_c.log" || { echo "queue_c did not finish"; exit 1; }
QUEUE="xt-c5120:zh2en" CARD=3 NSH=2 ASR_BASE=18001 TTS_BASE=18201 MT_PORT=18100 \
  bash "$R/experiments/baselines_xtranslator_v1/run_queue2.sh"
