#!/usr/bin/env bash
# Run registered points x directions one after another on the card-3 services (2 shards each).
#   QUEUE="xt-default:en2zh xt-default:zh2en ..." [ASR_CHUNK=0.6] bash run_queue.sh
set -uo pipefail
R=/opt/dlami/nvme/jasonleeeli/projects/UniSS
EXP=$R/experiments/baselines_xtranslator_v1
: "${QUEUE:?}"
for pd in $QUEUE; do
  POINT=${pd%%:*}; DIR=${pd##*:}
  OUT=$R/eval_outputs/baselines_xtranslator_v1/$POINT/realsi_$DIR
  want=$(wc -l < "$R/eval_outputs/baselines_xtranslator_v1/items_$DIR.jsonl")
  have=$(cat "$OUT"/part_*.jsonl 2>/dev/null | wc -l)
  if [ "$have" -ge "$want" ]; then echo "$(date -u +%FT%TZ) $pd already complete ($have)"; continue; fi
  echo "$(date -u +%FT%TZ) start $pd (have $have / $want)"
  POINT=$POINT DIR=$DIR CARD=3 NSH=2 ASR_BASE=18001 TTS_BASE=18201 MT_PORT=18100 ASR_CHUNK=${ASR_CHUNK:-0.6} \
    bash "$EXP/run_point.sh" || { echo "LAUNCH_FAILED $pd"; exit 1; }
  for k in 0 1; do
    pid=$(cat "$OUT/logs/driver_shard$k.pid")
    while kill -0 "$pid" 2>/dev/null; do sleep 10; done
  done
  have=$(cat "$OUT"/part_*.jsonl | wc -l)
  errs=$(cat "$OUT"/part_*.jsonl | grep -c '"error": "' || true)
  echo "$(date -u +%FT%TZ) done $pd rows=$have/$want error_rows=$errs"
  [ "$have" -ge "$want" ] || { echo "INCOMPLETE $pd"; exit 1; }
done
echo "$(date -u +%FT%TZ) QUEUE_DONE"
