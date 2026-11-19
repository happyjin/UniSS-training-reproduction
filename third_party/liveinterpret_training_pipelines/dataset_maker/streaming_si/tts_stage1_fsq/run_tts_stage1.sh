#!/bin/bash
# ============================================================================
# TTS 语料 -> streaming_si stage1（S1）FSQ parquet，多进程 x 多卡
# ============================================================================
# 用法（在 GPU 机器上）:
#   SOURCE=yodas  LANG=jp  bash run_tts_stage1.sh 3 8     # 3 进程/卡 x 8 卡 = 24 并发
#   SOURCE=lance  LANG=ja  bash run_tts_stage1.sh 2 8
#   bash run_tts_stage1.sh --progress
#
# 为什么多进程/卡: 音频解码(mp3/webm/flac -> 16k) 与重采样是 CPU 密集的串行段，
# 期间 GPU 空转；每卡多开进程才能把 FSQ tokenizer 喂满（源项目实测 3/卡 最优）。
#
# 断点续跑: 目标 parquet 已存在即跳过（S3 上只在上载完成后才出现最终 key），
# 中断后重跑同一条命令即可。
#
# 环境变量:
#   SOURCE        lance|emilia|yodas|wenet|iscslp|ko|hifitts   （必填）
#   LANG          jp|zh|ko|en（iscslp 必填；lance 可写 ja,ko）
#   FSQ_TOKENIZER speech_tokenizer_v3.batch.onnx 本地路径（缺省从 S3 拉）
#   TEXT_TOKENIZER  Qwen 词表目录；给定时额外写 lengths 列
#   OUT_PREFIX    S3 输出前缀（默认 .../TTS_FSQ）
#   LOGDIR
# ============================================================================
set -u
cd "$(dirname "$0")"
SOURCE=${SOURCE:?需设置 SOURCE}
LANG_OPT=${LANG:-}
# ConfigMap 挂载点是只读的，日志必须落到可写目录（默认随 TMP_DIR）
LOGDIR=${LOGDIR:-${TMP_DIR:-/tmp}/tts_stage1_logs}
OUT_PREFIX=${OUT_PREFIX:-speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS_FSQ}

if [ "${1:-}" = "--progress" ]; then
    echo "已产出 parquet: $(find "$LOGDIR" -name '*.log' 2>/dev/null | wc -l) 个分片日志"
    grep -h "累计" "$LOGDIR"/*.log 2>/dev/null | tail -n "$NSHARD_DEFAULT:-8"
    exit 0
fi

NPROC=${1:-3}
NGPU=${2:-8}
NSHARD=${NSHARD:-$((NPROC * NGPU))}
SHARD_OFFSET=${SHARD_OFFSET:-0}    # Indexed Job 里填 JOB_COMPLETION_INDEX * NPROC
mkdir -p "$LOGDIR"

for g in $(seq 0 $((NGPU - 1))); do
    for p in $(seq 0 $((NPROC - 1))); do
        S=$((SHARD_OFFSET + g * NPROC + p))
        env SOURCE="$SOURCE" LANG="$LANG_OPT" \
        nohup python3 tts_to_stage1.py \
            --source "$SOURCE" ${LANG_OPT:+--lang "$LANG_OPT"} \
            --s3-out-prefix "$OUT_PREFIX" \
            ${FSQ_TOKENIZER:+--fsq-tokenizer "$FSQ_TOKENIZER"} \
            ${TEXT_TOKENIZER:+--text-tokenizer "$TEXT_TOKENIZER"} \
            ${TMP_DIR:+--tmp-dir "$TMP_DIR"} \
            ${ISCSLP_MOVIES:+--iscslp-movies "$ISCSLP_MOVIES"} \
            ${MIN_DNSMOS:+--min-dnsmos "$MIN_DNSMOS"} \
            ${BS:+--bs "$BS"} \
            ${ROWS_PER_FILE:+--rows-per-file "$ROWS_PER_FILE"} \
            ${LIMIT_ITEMS:+--limit-items "$LIMIT_ITEMS"} \
            --device "cuda:$g" --shard "$S" --nshard "$NSHARD" \
            > "$LOGDIR/${SOURCE}-s${S}.log" 2>&1 &
    done
done
echo "已启动 $NSHARD 个进程（${NPROC}/卡 x ${NGPU} 卡）source=$SOURCE lang=${LANG_OPT:-auto}"
echo "日志: $LOGDIR/${SOURCE}-s*.log"
# k8s Job 里必须 wait：否则主进程一退出容器就 Completed，子进程被杀
wait
