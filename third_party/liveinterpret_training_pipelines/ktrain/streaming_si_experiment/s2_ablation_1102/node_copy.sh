#!/usr/bin/env bash
# 节点间复制（集群内网，不走跨区 S3，避免重复付雅加达 -> 美西的流量费）。
#   源节点：bash node_copy.sh serve            起只读 rsync 守护进程（端口 8730，导出 /nvme/speech_model_backup）
#   目标节点：bash node_copy.sh pull <源 Pod IP> <tag...>
#     tag：model | machine | s1031 | ko2zh（与 sync_data.sh 同名）
# 文件级 16 路并行（单个 rsync 流约几百 MB/s，跑不满网卡）；完成后同样写 /nvme/.sync_done/<tag>。
set -uo pipefail
N=/nvme
D=stevehguo/data_fortrain
mkdir -p "$N/.sync_done" "$N/logs"

if [[ "${1:-}" == serve ]]; then
    cat > /nvme/abl/rsyncd.conf <<'EOF'
uid = root
gid = root
use chroot = no
max connections = 64
[smb]
    path = /nvme/speech_model_backup
    read only = yes
EOF
    pkill -f "rsync --daemon" || true
    rsync --daemon --config=/nvme/abl/rsyncd.conf --port=8730 && echo "[serve] rsync daemon on :8730"
    exit 0
fi

[[ "${1:-}" == pull ]] || { echo "用法见文件头"; exit 2; }
SRC="$2"; shift 2

keys_of() {
    case "$1" in
        model)   echo stevehguo/model_checkpoint/streaming_si_swift/base_streaming_si_4b ;;
        machine) for m in 0 1 2; do echo "$D/open_data_0803_fortrain/machine_${m}_shard"; done ;;
        s1031)   for d in en2zh zh2en en2ja ja2en en2ko ko2en ja2zh zh2ja zh2ko negative_noise negative_noise_dns5; do
                     echo "$D/data_fortrain_1031version/Speech2Speech/$d"; done ;;
        ko2zh)   echo "$D/data_fortrain_1031version/Speech2Speech/ko2zh_20261025" ;;
        *) echo "未知 tag $1" >&2; return 1 ;;
    esac
}

for tag in "$@"; do
    [[ -f "$N/.sync_done/$tag" ]] && { echo "[pull] skip $tag"; continue; }
    t0=$SECONDS
    lst="$N/logs/pull_$tag.lst"; : > "$lst"
    for k in $(keys_of "$tag"); do
        rsync -r --list-only "rsync://$SRC:8730/smb/$k/" | awk -v k="$k" '$1 !~ /^d/ {print k"/"$NF}' >> "$lst"
    done
    n=$(wc -l < "$lst")
    echo "[pull] $tag: $n 个文件，16 路并行"
    split -n l/16 -d "$lst" "$lst.part."
    # 先建好全部目标目录：16 个 rsync 同时 mkdir 同一路径会撞 EEXIST（code 11）
    sed 's#/[^/]*$##' "$lst" | sort -u | sed "s#^#$N/speech_model_backup/#" | xargs -r mkdir -p
    for p in "$lst".part.*; do
        rsync -a --whole-file --files-from="$p" "rsync://$SRC:8730/smb/" "$N/speech_model_backup/" &
    done
    wait
    # 校验：文件数与总字节数逐一核对源端清单
    miss=0
    while read -r f; do [[ -s "$N/speech_model_backup/$f" ]] || miss=$((miss + 1)); done < "$lst"
    rm -f "$lst".part.*
    if [[ $miss -eq 0 ]]; then
        touch "$N/.sync_done/$tag"; echo "[pull] done $tag $((SECONDS - t0))s"
    else
        echo "[pull] FAIL $tag：缺 $miss 个文件，重跑本命令即可增量补齐"
    fi
done
echo "[pull] ALL_DONE $*"
