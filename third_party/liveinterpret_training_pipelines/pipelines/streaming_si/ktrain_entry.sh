#!/usr/bin/env bash
# streaming_si 的 ktrain 入口（KtrainJob framework: pytorch，yaml 里 TRAIN_SCRIPT 指向本文件）。
#
# ktrain/launch_train.sh 在每个 Pod 上执行本脚本一次（每节点一个 Pod），此时已导出
# NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT / NPROC_PER_NODE；本脚本只做三件事：
#   1. 把 yaml 里 s3:// 形式的数据变量转成节点 NVMe 路径（submit.sh 的同步 Job 已把它们
#      同步到 ${NVME_MOUNT}/<key>；launch_train.sh 只转换固定几个变量，不认识 SI_*）；
#   2. S5：在本节点准备数据（解压音频 tar、切 val、建分层索引），全部幂等、结果留在 NVMe；
#   3. 按 SI_STAGE 调用对应训练脚本（launch_train.sh 不给 TRAIN_SCRIPT 传参数）。
#
# 阶段（SI_STAGE，必填）-> 训练脚本 / 数据：
#   s1 | s3-w | s3   train_tts.sh    Emilia TTS 数据（SI_S1_DATA）
#   s2               train_st.sh     UniST P2 数据（SI_P2_*）
#   s4-w | s4        train_s2st.sh   UniST P2 数据（SI_P2_*）
#   s5               train_simul.sh  流式 records（SI_S4_*，变量名沿用源项目 S4）
# 2026-10 起阶段改名（新旧对照见 README 第 3 节）。旧名 s1-0 / s1-1w / s1-1a / s1-1 与新名不冲突，
# 自动映射为 s1 / s3-w / s3 / s3 并打印警告；旧名 s3-w / s3 / s4 与新名冲突，无法自动识别：
# 旧 yaml 会因数据变量对不上而在入口直接报错，须手动改成 s4-w / s4 / s5。
#
# 环境变量（yaml spec.env）：
#   SI_S1_DATA          S1/S3-w/S3：parquet 目录的 s3:// URI（或 NVMe 上的 glob）
#   SI_P2_DATA_DIRS     S2/S4-w/S4：各机器 p2_fsq_v3/ 的 s3:// URI（空格分隔，可多台机器）
#   SI_P2_WAV_DIRS      S2/S4-w/S4：各机器 p2_source_wav/ 的 s3:// URI（与上面按顺序一一对应）
#                       多台机器时用软链合并成一个 parquet 目录 + 一个 tar 目录；tar / parquet
#                       跨机器重名直接报错（数据集要求一个 tar 目录，重名会让音频配错行）
#   GROUP_BY_LENGTH     S1-S4：true = 按 parquet 里的 lengths 列分桶组 batch（common_args.sh 透传
#                       --group_by_length；减少 padding）。lengths 列由 p2_parquet_prep.py / s1_parquet_prep.py 生成
#   SI_S4_RECORDS       S5：各方向 records 目录的 s3:// URI（空格分隔；须同在一个 .../records/ 下）
#   SI_S4_AUDIO_TARS    S5：音频 tar 目录的 s3:// URI（.../audio/_tars/），解压到其上一级 audio/
#   SI_S4_AUDIO_ROOT    S5：节点上已解压好的音频根目录（设了就不用 SI_S4_AUDIO_TARS，也不解压）
#   SI_PREP_ONLY        1 = 只准备数据后退出（多节点首次在新节点上跑时先用它，见下）
#   SI_SHARED_OUTPUT    1 = OUTPUT_ROOT 在多节点共享存储上（不再加 --save_on_each_node）
#   SI_EXTRA_ARGS       追加给 swift 的参数（空格分隔），如 "--resume_from_checkpoint /nvme/..."
#   SI_CODE_CHECK       1 = 训练前检查 import / 模型变体 / 4B 加速核 / ray / GPU 数（不加载权重，秒级）
#   USE_RAY             true = KubeRay framework: ray（目前只支持 s1 / s3-w / s3，见下方 ray 段）
#
# 为什么多节点要先单独跑一次 SI_PREP_ONLY=1：数据准备在 torchrun 之前、各节点独立进行。
# 若一个节点已有数据、另一个要解压几小时，先完成的节点会在 rendezvous 处等到超时。
# 因此多节点训练时本脚本只校验、不做耗时准备；缺数据直接报错，不白占 GPU。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NVME_MOUNT="${NVME_MOUNT:-/nvme}"
PYTHON="${PYTHON:-python}"
STAGES="s1 | s2 | s3-w | s3 | s4-w | s4 | s5"
: "${SI_STAGE:?需要 SI_STAGE（${STAGES}）}"
: "${CHUNK_MS:=640}"
export CHUNK_MS
RENAME_HINT="（2026-10 阶段改名：旧 s3-w / s3 / s4 现为 s4-w / s4 / s5，旧 s1-1w / s1-1a 现为 s3-w / s3；见 README 第 3 节）"

log() { echo "[streaming_si/ktrain] $*" >&2; }
die() { echo "[streaming_si/ktrain][FATAL] $*" >&2; exit 1; }

# 与新名不冲突的旧名（源项目编号）自动映射
case "${SI_STAGE}" in
    s1-0)       _new=s1 ;;
    s1-1w)      _new=s3-w ;;
    s1-1a|s1-1) _new=s3 ;;
    *)          _new="" ;;
esac
if [[ -n "${_new}" ]]; then
    log "[WARN] SI_STAGE=${SI_STAGE} 是旧名，按 ${_new} 执行；请把 yaml 改成新名 ${RENAME_HINT}"
    SI_STAGE="${_new}"
fi

# 与 launch_train.sh / derive_sync_jobs.py 同一规则：s3://<bucket>/<key> -> ${NVME_MOUNT}/<key>
to_local() {
    local v="$1"
    case "${v}" in
        s3://*)
            local rest="${v#s3://}"; rest="${rest#*/}"
            v="${NVME_MOUNT%/}/${rest}"
            while [[ "${v}" == *"//"* ]]; do v="${v//\/\///}"; done
            [[ ${#v} -gt 1 ]] && v="${v%/}"
            ;;
    esac
    echo "${v}"
}

# 目录 -> <目录>/*.parquet（数据集注册要求以 .parquet 结尾）
parquet_glob() {
    local p; p="$(to_local "$1")"
    if [[ -d "${p}" ]]; then echo "${p%/}/*.parquet"; else echo "${p}"; fi
}

MULTI_NODE=false
[[ "${NNODES:-1}" -gt 1 ]] && MULTI_NODE=true
PREP_ONLY="${SI_PREP_ONLY:-0}"

# ---------------------------------------------------------------------------
# ray 模式（USE_RAY=true，KubeRay framework: ray）
#   * 训练脚本只在 driver 节点执行一次；其他 GPU worker 只经 launch_train.sh 的
#     NODE_PREP_ONLY（需 NODE_PREP_RUN_TRAIN_SCRIPT=true）以 NODE_PREP_ONLY=true 调本脚本，
#     此时只做本节点数据准备 / 校验 + 代码检查后退出。
#   * NNODES/NPROC_PER_NODE 被刻意 unset，多节点改由 ray_entrypoint.py 注入的
#     RAY_WORLD_SIZE / RAY_NPROC_PER_NODE 判定。
#   * 目前只适配 Emilia 数据的 s1 / s3-w / s3（setup_s1 每节点独立、无跨机合并）。
#     S2 / S4-w / S4 的 P2 合并、S5 的解压 / 建索引与全局 batch 计算尚未迁到 ray —— 直接拒绝。
# ---------------------------------------------------------------------------
IS_RAY=false
[[ "${USE_RAY:-false}" == true || "${USE_RAY:-false}" == 1 ]] && IS_RAY=true
NODE_PREP="${NODE_PREP_ONLY:-false}"
if [[ "${IS_RAY}" == true ]]; then
    case "${SI_STAGE}" in
        s1|s3-w|s3) ;;
        *) die "USE_RAY=true 目前只支持 s1 / s3-w / s3，收到 SI_STAGE=${SI_STAGE}；其他阶段请用 framework: pytorch" ;;
    esac
    [[ -n "${RAY_WORLD_SIZE:-}" ]] || die "ray 模式缺 RAY_WORLD_SIZE（应由 ktrain/ray_entrypoint.py 注入）"
    _ray_nodes=$(( RAY_WORLD_SIZE / ${RAY_NPROC_PER_NODE:-8} ))
    [[ "${_ray_nodes}" -gt 1 ]] && MULTI_NODE=true
fi

# ---------------------------------------------------------------------------
# S5 节点数据准备（函数名 / SI_S4_* 变量沿用源项目 S4 的命名）
# ---------------------------------------------------------------------------
_extract_one() {
    # $1 = <tar_dir>/<dir>_shardNNN.tar；tar 内是 shardNNN/，解到 <audio>/<dir>/
    local tar="$1" audio="$2" name dir shard marker
    name="$(basename "${tar}" .tar)"
    dir="${name%%_shard*}"
    shard="${name#"${dir}"_}"
    marker="${audio}/${dir}/.extracted_${shard}"
    [[ -f "${marker}" ]] && return 0
    mkdir -p "${audio}/${dir}"
    tar -xf "${tar}" -C "${audio}/${dir}" && touch "${marker}"
}
export -f _extract_one

s4_missing() {
    # 输出缺失项（空 = 已就绪）。tars 为空 = 使用已解压的 SI_S4_AUDIO_ROOT，不检查解压进度
    local audio="$1" tars="$2" d n_tar n_done
    if [[ -n "${tars}" ]]; then
        n_tar=$(find "${tars}" -maxdepth 1 -name '*.tar' | wc -l | tr -d ' ')
        n_done=$(find "${audio}" -mindepth 2 -maxdepth 2 -name '.extracted_*' 2>/dev/null | wc -l | tr -d ' ')
        [[ "${n_tar}" -gt 0 ]] || echo "audio tar 目录为空: ${tars}"
        [[ "${n_done}" -ge "${n_tar}" ]] || echo "音频未解压完: ${n_done}/${n_tar}"
    fi
    for d in en2zh zh2en; do
        [[ -f "${SI_S4_VAL_DIR}/${d}_val_keys.json" ]] || echo "缺 val: ${SI_S4_VAL_DIR}/${d}_val_keys.json"
        [[ -f "${SI_S4_INDEX_DIR}/${d}_index_meta.json" ]] || echo "缺索引: ${SI_S4_INDEX_DIR}/${d}_index_meta.json"
    done
}

s4_prepare() {
    # $1 = records 根（含 {en2zh,zh2en}/chunk_*/）；$2 = 音频根；$3 = tar 目录（空 = 不解压）
    local rec_root="$1" audio="$2" tars="$3" d rec
    local workers="${SI_S4_PREP_WORKERS:-32}"
    # 同一节点上的并发任务互斥（NVMe 为节点持久 hostPath）
    exec 9>"${SI_S4_PREP_DIR}/.swift_prep.lock"
    flock 9
    if [[ -n "${tars}" ]]; then
        log "S5 prep @ $(hostname): 解压音频 tar 到 ${audio}（${workers} 并发，已解压的跳过）"
        find "${tars}" -maxdepth 1 -name '*.tar' -print0 \
            | xargs -0 -P "${workers}" -I{} bash -c '_extract_one "$1" "$2"' _ {} "${audio}"
    fi
    for d in en2zh zh2en; do
        rec="${rec_root}/${d}/chunk_${CHUNK_MS}ms"
        if [[ ! -f "${SI_S4_VAL_DIR}/${d}_val_keys.json" ]]; then
            log "S5 prep: 切 val ${d}"
            "${PYTHON}" "${REPO_ROOT}/dataset_maker/streaming_si/build_s4_val_split.py" \
                --records "${rec}/val/records" --direction "${d}" --n_val "${SI_S4_N_VAL:-1500}" \
                --out "${SI_S4_VAL_DIR}" --data_root "${audio}"
        fi
        if [[ ! -f "${SI_S4_INDEX_DIR}/${d}_index_meta.json" ]]; then
            log "S5 prep: 建分层索引 ${d}"
            "${PYTHON}" "${REPO_ROOT}/dataset_maker/streaming_si/build_record_index.py" \
                --records "${rec}/train/records" --direction "${d}" --data_root "${audio}" \
                --out "${SI_S4_INDEX_DIR}" --workers "${SI_S4_INDEX_WORKERS:-64}"
        fi
    done
    flock -u 9
}

setup_s4() {
    : "${SI_S4_RECORDS:?S5 需要 SI_S4_RECORDS（各方向 records 目录的 s3:// URI）${RENAME_HINT}}"
    local tars="" audio rec_root="" tok rec d
    # records 根：取第一项的 .../records 前缀，其余项必须在同一根下
    for tok in ${SI_S4_RECORDS}; do
        rec="$(to_local "${tok}")"
        [[ "${rec}" == */records/* ]] || die "SI_S4_RECORDS 项 ${tok} 路径中没有 /records/"
        local this_root="${rec%%/records/*}/records"
        [[ -z "${rec_root}" ]] && rec_root="${this_root}"
        [[ "${this_root}" == "${rec_root}" ]] || die "SI_S4_RECORDS 各项须在同一 records 根下（${rec_root} vs ${this_root}）"
    done
    # 音频根：已解压目录优先；否则由 tar 目录推出（.../audio/_tars -> .../audio）
    if [[ -n "${SI_S4_AUDIO_ROOT:-}" ]]; then
        audio="$(to_local "${SI_S4_AUDIO_ROOT}")"
        [[ -d "${audio}/en2zh" && -d "${audio}/zh2en" ]] || die "SI_S4_AUDIO_ROOT 下缺 en2zh/ 或 zh2en/: ${audio}"
    else
        : "${SI_S4_AUDIO_TARS:?S5 需要 SI_S4_AUDIO_TARS（.../audio/_tars/ 的 s3:// URI）或 SI_S4_AUDIO_ROOT}"
        tars="$(to_local "${SI_S4_AUDIO_TARS}")"
        [[ "${tars}" == */audio/_tars ]] || die "SI_S4_AUDIO_TARS 应以 /audio/_tars/ 结尾，收到 ${SI_S4_AUDIO_TARS}"
        [[ -d "${tars}" ]] || die "音频 tar 目录不存在: ${tars}（同步 Job 未完成？）"
        audio="${tars%/_tars}"
    fi
    local train_dirs=()
    for d in en2zh zh2en; do
        rec="${rec_root}/${d}/chunk_${CHUNK_MS}ms"
        [[ -f "${rec}/train/records/_manifest.json" ]] || die "缺 ${rec}/train/records/_manifest.json（SI_S4_RECORDS 是否包含 ${d}/chunk_${CHUNK_MS}ms/train？）"
        [[ -d "${rec}/val/records" ]] || die "缺 ${rec}/val/records"
        train_dirs+=("${rec}/train/records")
    done
    # val / 索引产物按 records 版本落在 records 根旁（不同 rechunk 版本互不覆盖）
    export SI_S4_PREP_DIR="${SI_S4_PREP_DIR:-${rec_root%/records}/_swift_prep/chunk_${CHUNK_MS}ms}"
    export SI_S4_VAL_DIR="${SI_S4_VAL_DIR:-${SI_S4_PREP_DIR}/val}"
    export SI_S4_INDEX_DIR="${SI_S4_INDEX_DIR:-${SI_S4_PREP_DIR}/index}"
    mkdir -p "${SI_S4_VAL_DIR}" "${SI_S4_INDEX_DIR}"
    # 显式给出 records 与音频根，train_simul.sh 不再从 SI_S4 推导（records 与音频可在不同目录）
    SI_S4_DATA="$(IFS=,; echo "${train_dirs[*]}")"
    export SI_S4_DATA SI_S4_DATA_ROOT="${audio}"

    if [[ "${PREP_ONLY}" == 1 || "${MULTI_NODE}" == false ]]; then
        s4_prepare "${rec_root}" "${audio}" "${tars}"
    fi
    local missing; missing="$(s4_missing "${audio}" "${tars}")"
    if [[ -n "${missing}" ]]; then
        echo "${missing}" >&2
        die "本节点 S5 数据未就绪。多节点训练前先用同一份 yaml 加 SI_PREP_ONLY=1 提交一次（见本文件头注释）"
    fi
    log "S5 数据就绪 @ $(hostname): records=${SI_S4_DATA} audio=${SI_S4_DATA_ROOT} val=${SI_S4_VAL_DIR} index=${SI_S4_INDEX_DIR}"
}

# ---------------------------------------------------------------------------
# S2 / S4-w / S4：多台机器的 P2 数据合并
# ---------------------------------------------------------------------------
# 数据集要求「一个 parquet glob + 一个 tar 目录」（register_streaming_si.resolve_p2_wav_dir）。
# 多台机器时在节点 NVMe 上建软链目录：merged/p2_fsq_v3/*.parquet 与 merged/p2_source_wav/*.tar，
# 行里的 source_wav 是「tar 名::偏移」，只要 tar 名跨机器唯一就能正确配对；重名直接报错。
setup_p2() {
    : "${SI_P2_DATA_DIRS:?S2/S4-w/S4 需要 SI_P2_DATA_DIRS（各机器 p2_fsq_v3/ 的 s3:// URI）${RENAME_HINT}}"
    : "${SI_P2_WAV_DIRS:?S2/S4-w/S4 需要 SI_P2_WAV_DIRS（各机器 p2_source_wav/ 的 s3:// URI，与上面一一对应）}"
    local pq_dirs=() wav_dirs=() tok i
    for tok in ${SI_P2_DATA_DIRS}; do pq_dirs+=("$(to_local "${tok}")"); done
    for tok in ${SI_P2_WAV_DIRS}; do wav_dirs+=("$(to_local "${tok}")"); done
    [[ ${#pq_dirs[@]} -eq ${#wav_dirs[@]} ]] || die "SI_P2_DATA_DIRS 与 SI_P2_WAV_DIRS 数量不同（${#pq_dirs[@]} vs ${#wav_dirs[@]}）"
    for i in "${!pq_dirs[@]}"; do
        compgen -G "${pq_dirs[$i]}/*.parquet" > /dev/null || die "没有 parquet: ${pq_dirs[$i]}（同步 Job 未完成？）"
        compgen -G "${wav_dirs[$i]}/*.tar" > /dev/null || die "没有 tar: ${wav_dirs[$i]}（同步 Job 未完成？）"
        if [[ -f "${pq_dirs[$i]}/.cosy_version" ]]; then
            [[ "$(tr -d '[:space:]' < "${pq_dirs[$i]}/.cosy_version")" == v3 ]] \
                || die "${pq_dirs[$i]} 不是 v3 代 FSQ（.cosy_version=$(cat "${pq_dirs[$i]}/.cosy_version")）"
        fi
    done
    # 单台机器也走合并目录：要往 parquet 里加元数据列（超长过滤 / 分桶用），不能改原始数据
    [[ -n "${MODEL_PATH:-}" && -f "${MODEL_PATH}/config.json" ]] \
        || die "MODEL_PATH 下没有 config.json: ${MODEL_PATH:-<unset>}（P2 预处理要用它的 tokenizer）"
    # 合并目录名由输入列表决定：换了机器组合就是新目录，不会复用旧软链
    local tag merged
    tag="$(printf '%s\n' "${pq_dirs[@]}" "${wav_dirs[@]}" | cksum | cut -d' ' -f1)"
    merged="${NVME_MOUNT%/}/_swift_prep/p2_merged_${tag}"
    exec 8>"${NVME_MOUNT%/}/.swift_p2_merge_${tag}.lock"
    flock 8
    rm -rf "${merged}.tmp" && mkdir -p "${merged}.tmp/p2_fsq_v3" "${merged}.tmp/p2_source_wav"
    local f
    for i in "${!pq_dirs[@]}"; do
        for f in "${pq_dirs[$i]}"/*.parquet; do
            [[ -e "${merged}.tmp/p2_fsq_v3/$(basename "$f")" ]] && die "parquet 跨机器重名: $(basename "$f")"
            ln -s "$f" "${merged}.tmp/p2_fsq_v3/"
        done
        for f in "${wav_dirs[$i]}"/*.tar; do
            [[ -e "${merged}.tmp/p2_source_wav/$(basename "$f")" ]] && die "tar 跨机器重名: $(basename "$f")（同名不同内容会让音频配错行）"
            ln -s "$f" "${merged}.tmp/p2_source_wav/"
        done
    done
    # 统一各机器 parquet 的列（datasets 要求列名一致），并加元数据列（音频采样点数 / token 数 / lengths），
    # 供 SI_DROP_OVERLONG 的预先过滤与 GROUP_BY_LENGTH 分桶。只把软链替换成实体文件，原始数据不动
    "${PYTHON}" "${SCRIPT_DIR}/p2_parquet_prep.py" "${merged}.tmp/p2_fsq_v3" "${merged}.tmp/p2_source_wav" \
        --lengths --tokenizer "${MODEL_PATH}" || die "P2 parquet 预处理失败"
    rm -rf "${merged}" && mv "${merged}.tmp" "${merged}"
    flock -u 8
    export SI_P2_DATA="${merged}/p2_fsq_v3/*.parquet" SI_P2_WAV_DIR="${merged}/p2_source_wav"
    log "P2 合并 ${#pq_dirs[@]} 台机器 -> ${merged}（$(ls "${merged}/p2_fsq_v3" | wc -l | tr -d ' ') parquet / $(ls "${merged}/p2_source_wav" | wc -l | tr -d ' ') tar）"
}

# ---------------------------------------------------------------------------
# S1 / S3-w / S3（Emilia）：分桶时给 parquet 加 lengths 列（写副本，原始数据不动）
# ---------------------------------------------------------------------------
# 结果按「源目录 + 预处理脚本 + tokenizer + SI_MAX_SPEECH_LEN」缓存：同一组节点上 S3-w -> S3
# 第二次直接复用（约 20 GB 副本，生成一次约几分钟）。
setup_s1() {
    : "${SI_S1_DATA:?S1/S3-w/S3 需要 SI_S1_DATA${RENAME_HINT}}"
    local src
    src="$(to_local "${SI_S1_DATA}")"
    # 嵌套目录（如 TTS_FSQ_FILTERED/{zh,jp,ja-ko,ko,en}/...）：顶层 glob 匹配不到，
    # 且 s1_parquet_prep.py 也只认平铺目录 -> 先软链拍平，GROUP_BY_LENGTH 两种模式共用。
    # 文件名加相对路径前缀，避免跨子目录重名（zh/iscslp-* 与 en/iscslp-*）。
    if [[ -d "${src}" ]] && [[ -z "$(find "${src}" -maxdepth 1 -name '*.parquet' -print -quit)" ]]; then
        local tag flat f rel name
        tag="$( { echo "${src}"; find "${src}" -name '*.parquet' | wc -l; } | cksum | cut -d' ' -f1)"
        flat="${NVME_MOUNT%/}/_swift_prep/s1_flat_${tag}"
        exec 9>"${NVME_MOUNT%/}/.swift_s1_flat_${tag}.lock"
        flock 9
        if [[ -f "${flat}/.done" ]]; then
            log "S1 嵌套目录拍平复用 ${flat}"
        else
            rm -rf "${flat}.tmp" && mkdir -p "${flat}.tmp"
            while IFS= read -r f; do
                rel="${f#"${src%/}"/}"
                name="$(echo "${rel}" | tr '/' '_')"
                [[ -e "${flat}.tmp/${name}" ]] && die "拍平后 parquet 重名: ${rel}"
                ln -s "${f}" "${flat}.tmp/${name}"
            done < <(find "${src}" -name '*.parquet')
            touch "${flat}.tmp/.done"
            rm -rf "${flat}" && mv "${flat}.tmp" "${flat}"
            log "S1 嵌套目录已拍平 -> ${flat}"
        fi
        flock -u 9
        src="${flat}"
    fi
    if [[ "${GROUP_BY_LENGTH:-false}" != "true" ]]; then
        export SI_S1_DATA="$(parquet_glob "${src}")"
        compgen -G "${SI_S1_DATA}" > /dev/null || die "没有 Emilia parquet: ${SI_S1_DATA}（同步 Job 未完成？）"
        return
    fi
    [[ -d "${src}" ]] || die "GROUP_BY_LENGTH=true 时 SI_S1_DATA 须为目录: ${src}"
    compgen -G "${src}/*.parquet" > /dev/null || die "没有 Emilia parquet: ${src}（同步 Job 未完成？）"
    [[ -n "${MODEL_PATH:-}" && -f "${MODEL_PATH}/tokenizer.json" ]] \
        || die "MODEL_PATH 下没有 tokenizer.json: ${MODEL_PATH:-<unset>}（Emilia 长度预处理要用它）"
    local tag out
    tag="$( { echo "${src}"; ls "${src}" | wc -l; cat "${SCRIPT_DIR}/s1_parquet_prep.py" "${MODEL_PATH}/tokenizer.json"; \
              echo "speech=${SI_MAX_SPEECH_LEN:-750}"; } | cksum | cut -d' ' -f1)"
    out="${NVME_MOUNT%/}/_swift_prep/s1_len_${tag}"
    exec 9>"${NVME_MOUNT%/}/.swift_s1_len_${tag}.lock"
    flock 9
    if [[ -f "${out}/.done" ]]; then
        log "Emilia lengths 复用 ${out}"
    else
        rm -rf "${out}.tmp" && mkdir -p "${out}.tmp"
        "${PYTHON}" "${SCRIPT_DIR}/s1_parquet_prep.py" "${src}" "${out}.tmp" --tokenizer "${MODEL_PATH}" \
            || die "Emilia parquet 长度预处理失败"
        touch "${out}.tmp/.done"
        rm -rf "${out}" && mv "${out}.tmp" "${out}"
    fi
    flock -u 9
    export SI_S1_DATA="${out}/*.parquet"
}

# ---------------------------------------------------------------------------
# 按阶段分发
# ---------------------------------------------------------------------------
case "${SI_STAGE}" in
    s1|s3-w|s3)
        setup_s1
        cmd=(bash "${SCRIPT_DIR}/train_tts.sh" "${SI_STAGE}") ;;
    s2)
        setup_p2
        cmd=(bash "${SCRIPT_DIR}/train_st.sh") ;;
    s4-w|s4)
        setup_p2
        cmd=(bash "${SCRIPT_DIR}/train_s2st.sh" "${SI_STAGE}") ;;
    s5)
        setup_s4
        cmd=(bash "${SCRIPT_DIR}/train_simul.sh") ;;
    *) die "未知 SI_STAGE=${SI_STAGE}（${STAGES}）${RENAME_HINT}" ;;
esac

if [[ "${PREP_ONLY}" == 1 ]]; then
    log "SI_PREP_ONLY=1：数据准备完成 @ $(hostname)，不启动训练"
    exit 0
fi

# checkpoint 保存策略
#   torchrun 多节点：落各节点本地 NVMe，--save_on_each_node 保证每节点都有完整 checkpoint。
#   ray：16 个 actor 的 LOCAL_RANK 全是 0，--save_on_each_node 会让每个 actor 都当
#        「本节点主进程」并发写同一目录，不能开。只有全局 rank0（第一个 GPU worker）落盘，
#        续训 / 取产出从该节点拿（或设 SI_SHARED_OUTPUT=1 并把 OUTPUT_ROOT 指到共享存储）。
extra=()
if [[ "${MULTI_NODE}" == true && "${SI_SHARED_OUTPUT:-0}" != 1 && "${IS_RAY}" != true ]]; then
    extra+=(--save_on_each_node true)
fi
# shellcheck disable=SC2206
[[ -n "${SI_EXTRA_ARGS:-}" ]] && extra+=(${SI_EXTRA_ARGS})

[[ -n "${MODEL_PATH:-}" && -f "${MODEL_PATH}/config.json" ]] \
    || die "MODEL_PATH 下没有 config.json: ${MODEL_PATH:-<unset>}（同步 Job 未完成或路径不对）"

# ---------------------------------------------------------------------------
# 代码检查（SI_CODE_CHECK=1；只 import + 读 config，不加载权重，秒级）
#   - 插件与 streaming_si 模块能 import（代码同步完整、依赖齐全）
#   - 模型 thinker_arch 与 SI_MODEL_TYPE 一致
#   - 4B（qwen3_5）时镜像必须有 fla / causal-conv1d，否则线性注意力退回纯 torch、极慢
#   - USE_RAY 时镜像必须有 ray
#   - 本节点可见 GPU 数与 NPROC_PER_NODE 一致（仅 torchrun；ray 下跳过）
# torchrun：每节点在 rendezvous 前各跑一次；ray：每个 GPU worker 在 NODE_PREP_ONLY 里各跑一次，
# driver 再跑一次。任一失败直接退出，不让其他节点空等。
# ---------------------------------------------------------------------------
si_code_check() {
    # ray 的逐节点准备任务以 num_gpus=0 运行（ray_entrypoint.py::_node_prep），可能看不到 GPU；
    # ray driver 只看 1 张卡 —— 都跳过 GPU 数检查
    local want_gpus="" check_gpu=1
    if [[ "${IS_RAY}" == true ]]; then
        check_gpu=0
    else
        want_gpus="${NPROC_PER_NODE:-}"
    fi
    log "SI_CODE_CHECK @ $(hostname)（ray=${IS_RAY} node_prep=${NODE_PREP} gpu_check=${check_gpu}）"
    REPO_ROOT="${REPO_ROOT}" SI_WANT_GPUS="${want_gpus}" SI_CHECK_GPU="${check_gpu}" SI_IS_RAY="${IS_RAY}" \
        "${PYTHON}" - <<'PY' || die "SI_CODE_CHECK 失败 @ $(hostname)"
import json, os, sys
sys.path.insert(0, os.path.join(os.environ["REPO_ROOT"], "custom"))
cfg = json.load(open(os.path.join(os.environ["MODEL_PATH"], "config.json")))
arch = cfg.get("thinker_arch", "qwen3")
mtype = {"qwen3": "streaming_si", "qwen3_5": "streaming_si_4b"}.get(arch)
want = os.environ.get("SI_MODEL_TYPE")
assert mtype, f"未知 thinker_arch={arch}"
assert not want or want == mtype, f"SI_MODEL_TYPE={want} 与模型 thinker_arch={arch}（{mtype}）不一致"
import streaming_si.register_streaming_si  # noqa: F401  注册 model/template/dataset
if arch == "qwen3_5":
    import fla, causal_conv1d  # noqa: F401
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
    print(f"[code-check] fla {fla.__version__} / causal_conv1d {causal_conv1d.__version__}")
if os.environ.get("SI_IS_RAY") == "true":
    import ray
    print(f"[code-check] ray {ray.__version__}")
# liger：STREAMING_SI_LIGER_OPS=1 时必须真能挂上，否则直接失败。
# 不这么断言的话会重演 --use_liger_kernel 的坑：liger 查不到 model_type 就只打一条
# warning、训练照跑，跑完才发现这一轮白费（见 integration/liger_ops.py docstring）。
if os.environ.get("STREAMING_SI_LIGER_OPS") == "1":
    from streaming_si.integration.liger_ops import (apply_liger_to_streaming_si, required_patch_fns,
                                                    _liger_version)
    from liger_kernel.transformers import monkey_patch as _mp
    # 只断言「必需」目标（talker=Qwen3，S1/S3/S3-w 训的就是它）。thinker(Qwen3Next) 是可选：
    # 它全程冻结、只有前向，liger 0.5.10 也还没有 apply_liger_kernel_to_qwen3_next，
    # 缺了只少一块前向收益，不该让整个实验失败。
    _miss = [f for f in required_patch_fns() if getattr(_mp, f, None) is None]
    assert not _miss, (f"STREAMING_SI_LIGER_OPS=1 但 liger-kernel {_liger_version()} "
                       f"缺必需的 {_miss}")
    _done = apply_liger_to_streaming_si()
    assert _done, "STREAMING_SI_LIGER_OPS=1 但没有任何目标被融合"
    print(f"[code-check] liger {_liger_version()} -> {_done}")
import torch
n = torch.cuda.device_count()
if os.environ.get("SI_CHECK_GPU") == "1":
    assert n >= 1, "没有可见 GPU"
    w = os.environ.get("SI_WANT_GPUS")
    assert not w or n == int(w), f"可见 GPU {n} 与期望 {w} 不一致"
print(f"[code-check] OK model_type={mtype} talker={cfg.get('talker_preset')} visible_gpus={n}")
PY
}

if [[ "${NODE_PREP}" == true ]]; then
    # ray 逐节点准备：上面的 case 已在本节点完成数据准备 / 校验，这里做代码检查后退出（不训练）
    [[ "${SI_CODE_CHECK:-0}" == 1 ]] && si_code_check
    log "NODE_PREP_ONLY：本节点数据与代码检查完成 @ $(hostname)"
    exit 0
fi
[[ "${SI_CODE_CHECK:-0}" == 1 ]] && si_code_check

# SI_SKIP_THINKER 守卫：只有 s1 是 cond_mode=text_only + text_w=0（见 train_tts.sh 的 case），
# 其余阶段 thinker hidden 真的要用。这里按阶段提前拦掉误用 —— 模型侧 can_skip_thinker 还会
# 再校验一次实际 config，但那时已在训练进程内，不如在入口就失败。
# 注意不能检查 SI_COND_MODE：它由 train_tts.sh 导出，此刻（exec 之前）还没设。
if [[ "${SI_SKIP_THINKER:-0}" == 1 || "${SI_SKIP_THINKER:-0}" == true ]]; then
    case "${SI_STAGE}" in
        s1) log "SI_SKIP_THINKER=1：s1 将跳过 thinker 主干前向（cond 路本就被 zeros_like 丢弃）" ;;
        *)  die "SI_SKIP_THINKER=1 只适用于 SI_STAGE=s1（收到 ${SI_STAGE}）；s3-w/s3/s4-w/s4/s5 用 cond_mode=full，thinker hidden 不可跳过" ;;
    esac
fi

if [[ "${IS_RAY}" == true ]]; then
    log "stage=${SI_STAGE} ray world=${RAY_WORLD_SIZE} nodes=${_ray_nodes} model=${MODEL_PATH}"
else
    log "stage=${SI_STAGE} nodes=${NNODES:-1} node_rank=${NODE_RANK:-0} model=${MODEL_PATH}"
fi
exec "${cmd[@]}" ${extra[@]+"${extra[@]}"}
