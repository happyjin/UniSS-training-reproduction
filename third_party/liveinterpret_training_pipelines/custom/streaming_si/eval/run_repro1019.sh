#!/bin/bash
# Repro-Omni 新版（1019）流式评测：8 卡动态队列跑 7 分片 × L1～L4，再接后处理与报告。
#   推理参数与第二/三轮完全一致（run_worker3.sh），只换 STAGE3。
#   产物全部写本地 NVMe（FSx 已满）；对照系统（第二轮 7 个 + 上一版复现 out3）原样复用。
# 用法（Pod 内）：bash run_repro1019.sh
set -uo pipefail
E=/fsx/stevehguo/eval
export OUT=${OUT:-/nvme/eval_out4}
export STAGE3=${STAGE3:-/nvme/models/repro_1019}
export TMPDIR=/nvme/tmp_out4; mkdir -p $TMPDIR $OUT/logs
W3=/workspace/custom/streaming_si/eval/run_worker3.sh
P3=/workspace/custom/streaming_si/eval/run_post3.sh
NG=${NG:-8}
Q=$OUT/queue.txt; LK=$OUT/queue.lock
log() { echo "[$(date +%H:%M:%S)] $*"; }

# 1) 队列：L1 最慢先发；每行 "<shard> rep<L>"
if [ ! -f $Q ]; then
  for L in 1 2 3 4; do for K in 0 1 2 3 4 5 6; do echo "$K rep$L"; done; done > $Q
fi
: > $OUT/queue.done
claim() {   # 原子地领取下一条（mkdir 作锁）
  while ! mkdir $LK 2>/dev/null; do sleep 0.2; done
  local line; line=$(head -1 $Q); [ -n "$line" ] && sed -i '1d' $Q
  rmdir $LK; echo "$line"
}
gpu_loop() {
  local g=$1
  while true; do
    local job; job=$(claim); [ -z "$job" ] && break
    set -- $job
    log "gpu$g 开始 shard$1 $2"
    bash $W3 $1 $g $2 >> $OUT/logs/worker.gpu$g.log 2>&1
    echo "$job" >> $OUT/queue.done
    log "gpu$g 完成 shard$1 $2（已完成 $(wc -l < $OUT/queue.done)/28）"
  done
}
log "开始：STAGE3=$STAGE3 OUT=$OUT"
for g in $(seq 0 $((NG - 1))); do gpu_loop $g & done
wait
log "推理全部完成"

# 2) 后处理：合并 → 8 卡 ASR/延迟/UTMOS/副语言 → 打分（与上一版复现、第二轮官方同表）
bash $P3 merge
for g in $(seq 0 $((NG - 1))); do bash $P3 gpu $g $NG $g > $OUT/logs/post_gpu$g.log 2>&1 & done
wait
log "GPU 后处理完成"
LABEL=Repro-Omni-1019 PREV_OUT=$E/out3 PREV_LABEL=Repro-Omni-1016 bash $P3 score > $OUT/logs/score_all.log 2>&1
PYTHONPATH=/workspace/custom /opt/venv/bin/python /workspace/custom/streaming_si/eval/make_report.py \
  $OUT/_report $OUT/_report/REPORT.md
log "REPRO1019_DONE"
