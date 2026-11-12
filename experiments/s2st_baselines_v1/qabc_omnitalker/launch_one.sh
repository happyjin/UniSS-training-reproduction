#!/bin/bash
# One Omni engine on thinker card $1 / talker card $2 for ROOT (called by auto_launch.sh under gpu_launch.lock):
# pending entries in the shared ledger (the other orchestrators subtract them until visible / 300 s), then
# start_engines.sh with MAX_TRIES=1; if the engine did not start, the entries are closed with a "visible" line.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
T=$1; K=$2
ROOT=${ROOT:?}
LEDGER=logs/mist_w3po_v1/locks/pending.jsonl
O=r5_qabc_omnitalker
t=$(date +%s); utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
id="engine_e${T}${K}_$t"
if [ "$T" = "$K" ]; then
  echo "{\"t\": $t, \"utc\": \"$utc\", \"owner\": \"$O\", \"id\": \"$id\", \"card\": $T, \"mib\": 122000, \"vllm\": true, \"state\": \"pending\"}" >> $LEDGER
else
  echo "{\"t\": $t, \"utc\": \"$utc\", \"owner\": \"$O\", \"id\": \"$id\", \"card\": $T, \"mib\": 88000, \"vllm\": true, \"state\": \"pending\"}" >> $LEDGER
  echo "{\"t\": $t, \"utc\": \"$utc\", \"owner\": \"$O\", \"id\": \"${id}_talker\", \"card\": $K, \"mib\": 34000, \"vllm\": true, \"state\": \"pending\"}" >> $LEDGER
fi
out=$(ROOT=$ROOT ENGINES="$T:$K" MAX_TRIES=1 STAGGER=0 bash experiments/s2st_baselines_v1/qabc_omnitalker/start_engines.sh 2>&1)
echo "$out"
if ! echo "$out" | grep -q "engine $T:$K pid"; then
  t2=$(date +%s); utc2=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "{\"t\": $t2, \"utc\": \"$utc2\", \"owner\": \"$O\", \"id\": \"$id\", \"card\": $T, \"state\": \"visible\"}" >> $LEDGER
  [ "$T" = "$K" ] || echo "{\"t\": $t2, \"utc\": \"$utc2\", \"owner\": \"$O\", \"id\": \"${id}_talker\", \"card\": $K, \"state\": \"visible\"}" >> $LEDGER
fi
