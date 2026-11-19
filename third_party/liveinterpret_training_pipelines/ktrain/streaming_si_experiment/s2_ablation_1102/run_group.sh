#!/usr/bin/env bash
# S2 回退根因消融（2026-11-02）：单节点 8 卡跑一组，原样复用 pipelines/streaming_si/ktrain_entry.sh。
#
#   bash run_group.sh <G0|G1|G2|PC>
#
# | 组 | 数据 | 全局 batch | LR | 步数 | 回答的问题 |
# |----|------|-----------|----|------|-----------|
# | G0 | 新 S2 实际清单：15 目录（ko2zh 用 10-25 固定版） | 1536 | 2.8e-4 | 56,924（原 schedule） | 当前环境能否复现回退；回退出现在哪一步 |
# | G1 | G0 + en↔zh（1031 en2zh/zh2en + machine_0/1/2）再复制一份 | 1536 | 2.8e-4 | 56,924 | en↔zh 占比 28%→约 44%，是否修复（变量 1） |
# | G2 | G0 去掉两个噪声负样本目录 | 1536 | 2.8e-4 | 56,924 | 空目标负样本是否参与回退（变量 4） |
# | PC | 旧 S2 清单：machine_0/1/2 | 768 | 2e-4 | 1 epoch（≈25,046） | 正对照：代码版本是否是第 5 个变量。旧 S2 用 10-10 的代码训练（git 未完整记录），新 S2 / G 组用 10-26 之后的代码，S2 链路有大改；已有的旧 S2 checkpoint 回答不了“新代码 + 旧配方是否仍然好” |
#
# 单节点用梯度累积还原原全局 batch：G0/G1/G2 每卡 48 × 8 卡 × GA4 = 1536；PC 每卡 48 × 8 × GA2 = 768。
# G0/G1/G2 显式 MAX_STEPS=56924：G1 数据变多、G2 变少，固定步数才能保证 LR schedule 与原 S2 逐步相同，
# 各组在同一步数上可直接比较。中间 checkpoint 每 2500 步保存（含优化器状态，便于停下后续跑）。
set -euo pipefail
G="${1:?用法: run_group.sh <G0|G1|G2|PC>}"
N=/nvme
CODE=/nvme/abl/code
S=s3://stevehguo-ap-southeast-3/speech_model_backup/stevehguo/data_fortrain
S1031=$S/data_fortrain_1031version/Speech2Speech
MACH=$S/open_data_0803_fortrain

pq() { echo "$1/p2_fsq_v3/"; }
wav() { echo "$1/p2_source_wav/"; }

MACHINE=("$MACH/machine_0_shard" "$MACH/machine_1_shard" "$MACH/machine_2_shard")
NEW10=("$S1031/en2zh" "$S1031/zh2en" "$S1031/en2ja" "$S1031/ja2en" "$S1031/en2ko" "$S1031/ko2en"
       "$S1031/ja2zh" "$S1031/zh2ja" "$S1031/ko2zh_20261025" "$S1031/zh2ko")
NOISE=("$S1031/negative_noise" "$S1031/negative_noise_dns5")

# ---- G1 的 en↔zh 复制份：parquet 以 dup_ 前缀软链到独立目录；tar 不复制（行内 source_wav
#      引用的是原 tar 名，合并目录里已有），只放一个空占位 tar 以满足 setup_p2 的“每个目录要有 tar”检查。
make_dup() {   # make_dup <s3 目录> -> 打印复制份的伪 s3 目录（落在 /nvme/<key>）
    local src="$1" key rest dupkey
    rest="${src#s3://}"; key="${rest#*/}"
    dupkey="abl_dup/$(basename "$key")"
    mkdir -p "$N/$dupkey/p2_fsq_v3" "$N/$dupkey/p2_source_wav"
    local f
    for f in "$N/$key/p2_fsq_v3"/*.parquet; do
        ln -sfn "$f" "$N/$dupkey/p2_fsq_v3/dup_$(basename "$f")"
    done
    [[ -f "$N/$dupkey/p2_source_wav/zz_dup_placeholder_$(basename "$key").tar" ]] \
        || tar -cf "$N/$dupkey/p2_source_wav/zz_dup_placeholder_$(basename "$key").tar" -T /dev/null
    echo "s3://bucket/$dupkey"
}

case "$G" in
    G0) DIRS=("${NEW10[@]}" "${NOISE[@]}" "${MACHINE[@]}") ;;
    G1) DIRS=("${NEW10[@]}" "${NOISE[@]}" "${MACHINE[@]}")
        for d in "$S1031/en2zh" "$S1031/zh2en" "${MACHINE[@]}"; do DIRS+=("$(make_dup "$d")"); done ;;
    G2) DIRS=("${NEW10[@]}" "${MACHINE[@]}") ;;
    PC) DIRS=("${MACHINE[@]}") ;;
    *) echo "未知组 $G"; exit 2 ;;
esac

SI_P2_DATA_DIRS=""; SI_P2_WAV_DIRS=""
for d in "${DIRS[@]}"; do SI_P2_DATA_DIRS+="$(pq "$d") "; SI_P2_WAV_DIRS+="$(wav "$d") "; done
export SI_P2_DATA_DIRS SI_P2_WAV_DIRS

export NVME_MOUNT=$N SI_STAGE=s2
export MODEL_PATH=$N/speech_model_backup/stevehguo/model_checkpoint/streaming_si_swift/base_streaming_si_4b
export OUTPUT_ROOT=$N/output/$G
export NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 NPROC_PER_NODE=8
export GROUP_BY_LENGTH=true PER_DEVICE_TRAIN_BATCH_SIZE=48 WARMUP_RATIO=0.03
export SI_MAX_AUDIO_SEC=30 MAX_LENGTH=768 SI_DROP_OVERLONG=1
export DATASET_NUM_PROC=12 DATALOADER_NUM_WORKERS=8
export SAVE_STEPS=2500 SAVE_TOTAL_LIMIT=40 SAVE_ONLY_MODEL=false LOGGING_STEPS=50
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# datasets / modelscope 的预处理缓存放 NVMe：默认在容器层，Pod 重启即丢（全量预处理约 1h）
export HF_HOME=$N/cache/hf MODELSCOPE_CACHE=$N/cache/modelscope TMPDIR=$N/tmp
mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE" "$TMPDIR"
case "$G" in
    PC) export GRADIENT_ACCUMULATION_STEPS=2 LEARNING_RATE=2e-4 MAX_STEPS=-1 ;;
    *)  export GRADIENT_ACCUMULATION_STEPS=4 LEARNING_RATE=2.8e-4 MAX_STEPS=56924 ;;
esac

mkdir -p "$OUTPUT_ROOT"
{
    echo "group=$G started=$(date -u +%FT%TZ) host=$(hostname)"
    echo "code=$(cat $CODE/.git_commit 2>/dev/null)"
    echo "lr=$LEARNING_RATE ga=$GRADIENT_ACCUMULATION_STEPS bs=$PER_DEVICE_TRAIN_BATCH_SIZE max_steps=$MAX_STEPS"
    echo "dirs(${#DIRS[@]}):"; printf '  %s\n' "${DIRS[@]}"
} | tee "$OUTPUT_ROOT/abl_manifest.txt"

cd "$CODE"
exec bash pipelines/streaming_si/ktrain_entry.sh
