#!/usr/bin/env bash
# Score each queue item as soon as the queue log reports it done.  QLOG=<queue log> QUEUE="p:d ..." CARD=3 bash score_watch.sh
set -uo pipefail
: "${QLOG:?}" "${QUEUE:?}" "${CARD:?}"
EXP=/opt/dlami/nvme/jasonleeeli/projects/UniSS/experiments/baselines_xtranslator_v1
for pd in $QUEUE; do
  until grep -q -E "done $pd rows=|already complete|INCOMPLETE $pd|LAUNCH_FAILED $pd" "$QLOG" 2>/dev/null && grep -q -E "(done|INCOMPLETE|LAUNCH_FAILED) $pd" "$QLOG"; do sleep 20; done
  if grep -q -E "INCOMPLETE $pd|LAUNCH_FAILED $pd" "$QLOG"; then echo "$(date -u +%FT%TZ) STOP: $pd did not finish"; exit 1; fi
  echo "$(date -u +%FT%TZ) scoring $pd"
  POINT=${pd%%:*} DIR=${pd##*:} CARD=$CARD bash "$EXP/score_run.sh" || { echo "$(date -u +%FT%TZ) SCORE_FAILED $pd"; exit 1; }
done
echo "$(date -u +%FT%TZ) WATCH_DONE"
