#!/usr/bin/env bash
# bash stop_at.sh <组> <步数>
# checkpoint-<步数> 完整落盘（模型 + ZeRO 优化器分片 + trainer_state，且 180s 无写入）后停掉本节点上的训练。
# 不改 MAX_STEPS：LR schedule 仍按 56,924 步走，各组在同一步数上可比；停下的组之后可用
#   SI_EXTRA_ARGS="--resume_from_checkpoint <ckpt>" bash run_group.sh <组>
# 原样续跑。
set -uo pipefail
G="${1:?组}"; S="${2:?步数}"; OUT=/nvme/output/$G
echo "[stop_at] $(date -u +%FT%TZ) 等待 $G checkpoint-$S"
while true; do
    c=$(ls -d "$OUT"/s2_st/v*/checkpoint-"$S" 2>/dev/null | tail -1)
    if [[ -n "$c" && -f "$c/trainer_state.json" && -d "$c/global_step$S" \
          && -z "$(find "$c" -newermt '180 seconds ago' -print -quit)" ]]; then
        echo "[stop_at] $(date -u +%FT%TZ) $c 已完整，停止 $G 训练"
        pkill -TERM -f "pipelines/streaming_si/ktrain_entry.sh" || true
        pkill -TERM -f "pipelines/streaming_si/train_st.sh" || true
        pkill -TERM -f "torchrun" || true
        pkill -TERM -f "swift/cli/sft.py" || true
        for _ in $(seq 60); do
            pgrep -f "swift/cli/sft.py|torchrun" >/dev/null || break; sleep 5
        done
        pkill -KILL -f "swift/cli/sft.py|torchrun" || true
        echo "[stop_at] $(date -u +%FT%TZ) $G 已停在 step $S"
        exit 0
    fi
    sleep 120
done
