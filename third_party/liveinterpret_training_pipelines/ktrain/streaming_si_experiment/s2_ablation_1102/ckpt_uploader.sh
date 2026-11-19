#!/usr/bin/env bash
# 与训练并行常驻：把 /nvme/output/<组>/ 下新出现的 checkpoint-N 的“模型部分”传到 S3（雅加达），供评测侧拉取。
#   bash ckpt_uploader.sh <组>
# 只传推理需要的文件（safetensors / config / tokenizer / speech_decoder / trainer_state 等），
# 不传 global_step*/（ZeRO 优化器分片，约 48GB）与 rng_state —— 它们留在本地 NVMe，用于停下后续跑。
# checkpoint 写完的判据：trainer_state.json 存在且目录 120s 内无变化。传完写 _UPLOAD_DONE 标记。
# 另外每轮同步训练日志 logging.jsonl 与 abl_manifest.txt。
set -uo pipefail
G="${1:?组名}"
X=s3://stevehguo-ap-southeast-3/speech_model_backup/stevehguo/s2_ablation_1102/ckpt/$G
OUT=/nvme/output/$G
while true; do
    for c in $(ls -d "$OUT"/s2_st/v*/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
        n=$(basename "$c")
        [[ -f "$c/.uploaded" ]] && continue
        [[ -f "$c/trainer_state.json" ]] || continue
        # 目录稳定性：最近 120s 内没有文件改动
        [[ -z "$(find "$c" -newermt '120 seconds ago' -print -quit)" ]] || continue
        echo "[upload] $(date -u +%T) $G/$n"
        if aws s3 sync "$c/" "$X/$n/" --only-show-errors \
              --exclude "global_step*" --exclude "rng_state*" --exclude "*.pt" --exclude "zero_to_fp32.py" --exclude "latest"; then
            echo "$(date -u +%FT%TZ)" | aws s3 cp - "$X/$n/_UPLOAD_DONE" --only-show-errors
            touch "$c/.uploaded"; echo "[upload] done $G/$n"
        fi
    done
    v=$(ls -d "$OUT"/s2_st/v* 2>/dev/null | tail -1)
    [[ -n "$v" && -f "$v/logging.jsonl" ]] && aws s3 cp "$v/logging.jsonl" "$X/logging.jsonl" --only-show-errors
    [[ -f "$OUT/abl_manifest.txt" ]] && aws s3 cp "$OUT/abl_manifest.txt" "$X/abl_manifest.txt" --only-show-errors
    [[ -f "/nvme/logs/train_$G.log" ]] && tail -c 2000000 "/nvme/logs/train_$G.log" | aws s3 cp - "$X/train_tail.log" --only-show-errors
    sleep 300
done
