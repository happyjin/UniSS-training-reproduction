#!/bin/bash
# 在本 Pod 上拉起若干 run_worker2：launch_shards2.sh <起始分片> <分片数>（GPU idx 从 0 起）
set -uo pipefail
START=$1; CNT=$2
E=/fsx/stevehguo/eval
O=${OUT:-$E/out2}
mkdir -p $O/logs
for i in $(seq 0 $((CNT - 1))); do
  K=$((START + i))
  (setsid env OUT=$O bash $E/code/custom/streaming_si/eval/run_worker2.sh $K $i "${SYS:-}" >> $O/logs/worker.$K.log 2>&1 < /dev/null &)
  sleep 2
done
sleep 10
pgrep -af "[r]un_worker2.sh" | awk '{print "worker", $(NF-2), "gpu", $(NF-1)}'
