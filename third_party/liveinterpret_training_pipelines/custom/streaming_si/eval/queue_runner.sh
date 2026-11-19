#!/bin/bash
# 共享任务队列（跨节点，基于 FSX 上 mkdir 的原子性，不依赖 flock 挂载参数）。
# 每张卡起一个 runner，按队列顺序逐个尝试 `mkdir <claims>/<任务>`，成功即领到，跑完写 .done。
# 队列文件每行 "<shard> <system>"，按预计耗时从长到短排。
#   queue_runner.sh <queue_file> <gpu_idx> <worker_script>
set -uo pipefail
Q=$1; G=$2; W=$3
C=$Q.claims; mkdir -p $C
while read -r k s; do
  [ -z "$k" ] && continue
  mkdir "$C/${k}_$s" 2>/dev/null || continue
  echo "$(hostname) gpu$G $(date +%H:%M:%S)" > "$C/${k}_$s/who"
  bash "$W" "$k" "$G" "$s" < /dev/null
  echo "$k $s rc=$? $(hostname) gpu$G $(date +%H:%M:%S)" >> "$Q.done"
done < "$Q"
echo "[runner $(hostname) gpu$G] 队列已空 $(date +%H:%M:%S)"
