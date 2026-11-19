#!/usr/bin/env bash
# 把 S3（雅加达）上的训练工件同步到本节点 /nvme/<key>（与 ktrain 的落盘约定一致）。
# 用法：bash sync_data.sh <组名...>
#   model    基座 base_streaming_si_4b
#   machine  open_data_0803_fortrain/machine_{0,1,2}_shard（旧 S2 的全部数据，正对照只需要这组）
#   s1031    1031 版九方向 + 两个噪声目录（不含 ko2zh）
#   ko2zh    ko2zh 的 10-25 固定版 ko2zh_20261025（pin_ko2zh_20261025.py 完成后再跑）
# 增量（--size-only），可重复执行；每个目录完成后写 /nvme/.sync_done/<名字>。
set -uo pipefail
B=stevehguo-ap-southeast-3
D=speech_model_backup/stevehguo/data_fortrain
N=/nvme
mkdir -p "$N/.sync_done" /nvme/logs
aws configure set default.s3.max_concurrent_requests 96
aws configure set default.s3.multipart_chunksize 64MB
aws configure set default.s3.max_queue_size 10000

one() {   # one <s3 key 前缀，不带结尾斜杠> <done 标记名>
    local key="$1" tag="$2" t0=$SECONDS
    [[ -f "$N/.sync_done/$tag" ]] && { echo "[sync] skip $tag（已完成）"; return 0; }
    for try in 1 2 3; do
        if aws s3 sync "s3://$B/$key/" "$N/$key/" --size-only --only-show-errors --region ap-southeast-3; then
            echo "[sync] done $tag $(du -sh "$N/$key" | cut -f1) $((SECONDS - t0))s"
            touch "$N/.sync_done/$tag"; return 0
        fi
        echo "[sync] retry $try $tag"; sleep 30
    done
    echo "[sync] FAIL $tag"; return 1
}

pairs() {   # pairs <目录前缀> <tag>：p2_fsq_v3 + p2_source_wav
    one "$1/p2_fsq_v3" "$2.pq" && one "$1/p2_source_wav" "$2.wav"
}

for g in "$@"; do
    case "$g" in
        model)
            one speech_model_backup/stevehguo/model_checkpoint/streaming_si_swift/base_streaming_si_4b base_model ;;
        machine)
            for m in 0 1 2; do pairs "$D/open_data_0803_fortrain/machine_${m}_shard" "machine_$m" & done; wait ;;
        s1031)
            # 4 路并行（每路内部 96 并发请求）。ko2zh 单独一组：固定版要等服务端复制完成再同步
            printf '%s\n' en2zh zh2en en2ja ja2en en2ko ko2en ja2zh zh2ja zh2ko negative_noise negative_noise_dns5 \
              | xargs -P 4 -I{} bash -c "$(declare -f one pairs); B=$B D=$D N=$N; pairs $D/data_fortrain_1031version/Speech2Speech/{} {}" ;;
        ko2zh)
            pairs "$D/data_fortrain_1031version/Speech2Speech/ko2zh_20261025" ko2zh_20261025 ;;
        *) echo "未知组 $g"; exit 2 ;;
    esac
done
echo "[sync] ALL_DONE $*"
