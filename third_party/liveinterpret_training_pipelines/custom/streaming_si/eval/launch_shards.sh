#!/bin/bash
# 在本 Pod 上拉起若干 worker：launch_shards.sh <起始分片> <分片数> <总分片数>
# 每个 worker 绑一张卡（GPU idx 从 0 开始，对应 Pod 内可见的卡）。
set -uo pipefail
START=$1; CNT=$2; N=$3
E=/fsx/stevehguo/eval
mkdir -p $E/out/logs
for i in $(seq 0 $((CNT - 1))); do
  K=$((START + i))
  (setsid bash $E/run_worker.sh $K $N $i > $E/out/logs/worker.$K.log 2>&1 < /dev/null &)
  sleep 2
done
sleep 12
echo "本 Pod 启动的 worker："
pgrep -af "[r]un_worker.sh" | sed 's/ bash.*run_worker.sh/ worker/' | head -10
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '
echo
