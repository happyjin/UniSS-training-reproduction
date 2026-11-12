#!/usr/bin/env bash
# Card 3 continuation (replaces queue_a's loop, stopped by PID): wait for the running xt-c1280:en2zh shards, log them
# in the queue format, then xt-c1280:zh2en and xt-c5120:en2zh (2 shards).  xt-c5120:zh2en moves to card 6 (queue_d.sh).
set -uo pipefail
R=/opt/dlami/nvme/jasonleeeli/projects/UniSS
E=$R/eval_outputs/baselines_xtranslator_v1
OUT=$E/xt-c1280/realsi_en2zh
for k in 0 1; do pid=$(cat "$OUT/logs/driver_shard$k.pid"); while kill -0 "$pid" 2>/dev/null; do sleep 10; done; done
have=$(cat "$OUT"/part_*.jsonl | wc -l)
errs=$(cat "$OUT"/part_*.jsonl | grep -c '"error": "' || true)
echo "$(date -u +%FT%TZ) done xt-c1280:en2zh rows=$have/346 error_rows=$errs"
[ "$have" -ge 346 ] || { echo "INCOMPLETE xt-c1280:en2zh"; exit 1; }
QUEUE="xt-c1280:zh2en xt-c5120:en2zh" CARD=3 NSH=2 ASR_BASE=18001 TTS_BASE=18201 MT_PORT=18100 \
  bash "$R/experiments/baselines_xtranslator_v1/run_queue2.sh"
