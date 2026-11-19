#!/usr/bin/env bash
# bash chain.sh <下一组> <停止步数> <需要的数据标记...>
# 等本节点当前训练结束（自然跑完或被 stop_at.sh 停下）、GPU 释放、所需数据就绪后，
# 启动 <下一组> + 它的 checkpoint 上传器 + 在 <停止步数> 停下的 stop_at。
#   04ed：bash chain.sh G2 15000 s1031 ko2zh       （PC 跑完后）
#   0d22：bash chain.sh G1 15000 s1031 ko2zh       （G0 停在 15k 后）
set -uo pipefail
NEXT="${1:?下一组}"; STOP="${2:?停止步数}"; shift 2
echo "[chain] $(date -u +%FT%TZ) 排队：$NEXT（停在 $STOP），需要数据：$*"
# 1) 当前训练结束
while pgrep -f "pipelines/streaming_si/ktrain_entry.sh|swift/cli/sft.py|torchrun" >/dev/null; do sleep 120; done
# 2) GPU 显存释放
while [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1) -gt 2000 ]]; do sleep 30; done
# 3) 数据就绪（sync_data.sh / node_copy.sh 的完成标记，二者同名）
for t in "$@"; do
    case "$t" in
        s1031) need=(en2zh.wav zh2en.wav en2ja.wav ja2en.wav en2ko.wav ko2en.wav ja2zh.wav zh2ja.wav zh2ko.wav negative_noise.wav negative_noise_dns5.wav) ;;
        ko2zh) need=(ko2zh_20261025.wav) ;;
        *) need=("$t") ;;
    esac
    for m in "${need[@]}"; do
        until [[ -f /nvme/.sync_done/$m || -f /nvme/.sync_done/$t ]]; do echo "[chain] 等数据 $m"; sleep 300; done
    done
done
echo "[chain] $(date -u +%FT%TZ) 启动 $NEXT"
( setsid nohup bash /nvme/abl/run_group.sh "$NEXT" > /nvme/logs/train_"$NEXT".log 2>&1 < /dev/null & )
( setsid nohup bash /nvme/abl/ckpt_uploader.sh "$NEXT" > /nvme/logs/upload_"$NEXT".log 2>&1 < /dev/null & )
sleep 60
( setsid nohup bash /nvme/abl/stop_at.sh "$NEXT" "$STOP" > /nvme/logs/stop_"$NEXT".log 2>&1 < /dev/null & )
echo "[chain] done"
