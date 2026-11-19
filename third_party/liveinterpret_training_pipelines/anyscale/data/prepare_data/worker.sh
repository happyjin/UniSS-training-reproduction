#!/usr/bin/env bash
# =============================================================================
# Data-prep worker script — SOLE ORCHESTRATOR for the prepare_data pipeline.
#
# Runs on a SINGLE CPU head node (8CPU-32GB, see compute_config.yaml). Responsibilities:
#   0. FULL PRELOAD: at task start, lfs hsm_restore EVERY audio file under
#      TTS_DATA_PATH / S2S_DATA_PATH / TTS_RAW_PATH_PREFIX into the FSx SSD cache
#      (per AWS "preload file contents via HSM DRA" recipe) so the makers read
#      warm cache instead of triggering per-file on-demand recalls.
#   1. TTS  : ${TTS_DATA_PATH}  -> ${LOCAL_WORK_DIR}/text_to_speech[.jsonl]
#   2. S2S  : ${S2S_DATA_PATH}  -> ${LOCAL_WORK_DIR}/speech_to_speech_translation[.jsonl]
#   3. Merge: TTS:S2S = 5:1     -> ${LOCAL_WORK_DIR}/bridge_alignment_joint[.jsonl]
#   4. After EACH stage, upload that stage's result to
#        s3://${S3_BUCKET}/datasets/${TASK_NAME}/<stage>/
#      (LOCAL_WORK_DIR is the node-local ephemeral temp; S3 is the durable copy).
#   5. Audit: total byte size of all audio referenced by the final jsonl.
#   6. FINAL PREWARM: per PREWARM_TARGETS, lfs hsm_restore every audio referenced
#      by the corresponding stage's jsonl into FSx (so downstream training hits
#      the cache). Single-node (atomic lock on the shared FSX_MOUNT).
#
# Concurrency: each maker is invoked ONCE per stage with --num-workers = all CPU
# cores, so every core is saturated while that stage runs. No multi-node fan-out
# (the node is CPU-only and the dataset is processed sequentially per stage).
#
# FSx-specific work (preload / final prewarm / audit) is done by prewarm.py;
# non-FSx plumbing (find-inputs/concat/manifest/ready) by tools.py.
# All configuration is injected via env (job.yaml).
# =============================================================================
set -euo pipefail

# ---- Python 依赖 (不要在此处 pip install) ----
# pandas / soundfile / pyarrow / imageio-ffmpeg / awscli 等全部已在
# anyscale/image/warm-fsx:<N> 镜像内预装 (见 Dockerfile.warm_fsx),
# 由 compute_config 的 image_uri 引用. 禁止在 sh 脚本里安装依赖.
# 注: shardio 是仓库 dataset_maker/ 下的本地模块 (maker 经 sys.path 导入), 非 pip 包.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PD="${REPO_ROOT}/anyscale/data/prepare_data"
PREWARM_PY="${PD}/prewarm.py"
TOOLS_PY="${PD}/tools.py"
TTS_SCRIPT="${REPO_ROOT}/dataset_maker/make_text_to_speech.py"
S2S_SCRIPT="${REPO_ROOT}/dataset_maker/make_speech_to_speech_translation.py"
MERGE_SCRIPT="${REPO_ROOT}/dataset_maker/merge_datasets.py"

_log() { echo "[worker] $(date +%H:%M:%S) $*" >&2; }

# 让每个 maker 子进程内的数值库保持单线程, 避免 8 个 worker 进程各自的
# OpenMP/MKL 线程相互抢占 CPU 核心 (8 进程 × 1 线程 = 恰好打满 8 核).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---- 必填环境变量校验 ----
check_required() {
    local missing=()
    for var in SMALL_DATA_LIMIT TTS_DATA_PATH S2S_DATA_PATH TTS_RAW_PATH_PREFIX; do
        [[ -n "${!var:-}" ]] || missing+=("$var")
    done
    if ((${#missing[@]} > 0)); then
        echo "[worker][FATAL] missing env vars: ${missing[*]}" >&2
        exit 1
    fi
}
check_required

# ---- 配置参数 (env 注入, 此处只给默认值) ----
FSX_MOUNT="${FSX_MOUNT:-/mnt/fsx}"
S3_BUCKET="${S3_BUCKET:-anyscale-poc-v2-bucket-124355679795-ap-northeast-1}"
S3_REGION="${S3_REGION:-ap-northeast-1}"
# 任务名: 作为 S3 上的根目录, 默认取时间戳保证每次运行独立.
TASK_NAME="${TASK_NAME:-prepare_$(date +%Y%m%d_%H%M%S)}"
S3_BASE="s3://${S3_BUCKET}/datasets/${TASK_NAME}"
# 本地临时目录 (节点 ephemeral-storage): 各阶段产物先写这里, 结束即上传 S3.
LOCAL_WORK_DIR="${LOCAL_WORK_DIR:-/tmp/prepare_data_${TASK_NAME}}"
OUT_DIR="${LOCAL_WORK_DIR}"
mkdir -p "${OUT_DIR}"

SAMPLING_RATE="${SAMPLING_RATE:-16000}"
LATENCY_BUDGET_MS="${LATENCY_BUDGET_MS:-2000}"
SMALL_DATA_LIMIT="${SMALL_DATA_LIMIT:-0}"
MAX_AUDIO_DURATION_SEC="${MAX_AUDIO_DURATION_SEC:-120}"
SHARD_MAX_LINES="${SHARD_MAX_LINES:-0}"
PREWARM_DIR_BATCH="${PREWARM_DIR_BATCH:-2000}"
PREWARM_POLL_INTERVAL="${PREWARM_POLL_INTERVAL:-15}"
PREWARM_FULL_WAIT_TIMEOUT="${PREWARM_FULL_WAIT_TIMEOUT:-7200}"
AUDIO_SIZE_REPORT="${AUDIO_SIZE_REPORT:-}"
BRIDGE_RATIOS="${BRIDGE_RATIOS:-1 5}"
read -ra BRIDGE_RATIOS_ARR <<< "$BRIDGE_RATIOS"

# 并发: 尽量占满 head 节点所有 CPU 核心.
# 容器内 nproc 不可靠 (cgroup 限制/视图问题), 用 cgroup 的 cpu quota/period
# 推导实际可用核数 (get_cpu_count); 仍可用 PREP_PARALLEL 环境变量强制覆盖.
get_cpu_count() {
    local quota period cpu_count=1
    if [ -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
        quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
        period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
        [ "$quota" -gt 0 ] && cpu_count=$((quota / period))
    elif [ -f /sys/fs/cgroup/cpu.max ]; then
        read -r quota period < /sys/fs/cgroup/cpu.max
        [ "$quota" != "max" ] && cpu_count=$((quota / period))
    fi
    [ "$cpu_count" -lt 1 ] && cpu_count=$(nproc 2>/dev/null || grep -c ^processor /proc/cpuinfo 2>/dev/null || echo 1)
    echo "$cpu_count"
}
NPROC="$(get_cpu_count)"
PREP_PARALLEL="${PREP_PARALLEL:-$NPROC}"

# 最终预热 (PREWARM_TARGETS) 相关
PREWARM_TASK_JSONL="${PREWARM_TASK_JSONL:-auto}"
PREWARM_TARGETS="${PREWARM_TARGETS:-bridge_alignment_joint}"
PREWARM_PARALLEL="${PREWARM_PARALLEL:-16}"
PREWARM_PRODUCER_POLL="${PREWARM_PRODUCER_POLL:-15}"
PREWARM_WAIT_TIMEOUT="${PREWARM_WAIT_TIMEOUT:-7200}"

# 单文件模式产物为 *.jsonl; 分片模式为子目录.
if [[ "${SHARD_MAX_LINES}" -gt 0 ]]; then
    TTS_OUT="${OUT_DIR}/text_to_speech"
    S2S_OUT="${OUT_DIR}/speech_to_speech_translation"
    FINAL_OUT="${OUT_DIR}/bridge_alignment_joint"
else
    TTS_OUT="${OUT_DIR}/text_to_speech.jsonl"
    S2S_OUT="${OUT_DIR}/speech_to_speech_translation.jsonl"
    FINAL_OUT="${OUT_DIR}/bridge_alignment_joint.jsonl"
fi

_log "TASK_NAME=${TASK_NAME}"
_log "S3_BASE=${S3_BASE}"
_log "LOCAL_WORK_DIR=${LOCAL_WORK_DIR}"
_log "PREP_PARALLEL=${PREP_PARALLEL} (nproc=${NPROC})"

# ---- S3 上传工具 (aws cli 由镜像预装; 见 Dockerfile.warm_fsx) ----
ensure_aws() {
    if ! command -v aws >/dev/null 2>&1; then
        _log "ERROR: aws CLI not found. 它必须在镜像内预装 (Dockerfile.warm_fsx), 禁止在 sh 里 pip install."
        return 1
    fi
}
s3_upload() {
    # $1 = local path (file or dir); $2 = s3 dest (dir-like, must end with /)
    local src="$1" dst="$2"
    if [[ ! -e "$src" ]]; then
        _log "s3_upload: skip (not found): ${src}"
        return 0
    fi
    ensure_aws || return 1
    aws configure set default.s3.max_concurrent_requests 100 2>/dev/null || true
    if [[ -d "$src" ]]; then
        aws s3 sync "${src}" "${dst}" --region "${S3_REGION}" --only-show-errors
    else
        aws s3 cp "${src}" "${dst}" --region "${S3_REGION}" --only-show-errors
    fi
}

# ---- FORCE_UPDATE: 强制重新生成所有数据 ----
if [[ -n "${FORCE_UPDATE:-}" ]]; then
    _log "FORCE_UPDATE=1 — removing existing outputs for full regeneration."
    rm -rf "${TTS_OUT}" "${S2S_OUT}" "${FINAL_OUT}"
    rm -f "${TTS_OUT}.inputs_manifest.json" "${S2S_OUT}.inputs_manifest.json"
    rm -f "${OUT_DIR}/.worker_prepare_data_done"
    # 最终任务 jsonl 预热的跨节点互斥标记 (位于共享 FSX_MOUNT), 清掉以触发重新预热
    rm -f "${FSX_MOUNT}/.prewarm_task_jsonl_done"
fi

# ---- 幂等 (pipeline 级): 整轮主流程已完成则跳过 ----
DONE_MARKER="${OUT_DIR}/.worker_prepare_data_done"
if [[ -f "${DONE_MARKER}" ]]; then
    _log "$(hostname) already done — skip pipeline."
else
    # =========================================================================
    # process_stage: 单个数据集的增量发现 + 单次 maker 调用 (占满所有核心) +
    # 拼接 + 上传 S3.
    # 参数: <kind tts|s2s> <input_dir> <out_base> <status_file> <script>
    #   status_file 写入 "1" 表示本次有新数据, "0" 表示无.
    # =========================================================================
    process_stage() {
        local kind="$1" input_dir="$2" out_base="$3" status_file="$4" script="$5"
        local name="${kind^^}"

        # 1) 发现当前输入
        local cur_file; cur_file="$(mktemp)"
        python3 "$TOOLS_PY" find-inputs --kind "$kind" --input-dir "$input_dir" > "$cur_file"
        local n_cur; n_cur="$(wc -l < "$cur_file" | tr -d ' ')"

        # 2) 增量 diff: 当前 - 已处理
        local manifest="$out_base.inputs_manifest.json"
        local delta_file; delta_file="$(mktemp)"
        python3 "$TOOLS_PY" manifest-diff --manifest "$manifest" --inputs-file "$cur_file" > "$delta_file"
        local n_delta; n_delta="$(wc -l < "$delta_file" | tr -d ' ')"

        local fresh=0
        [[ ! -f "$manifest" ]] && fresh=1

        if [[ "$n_delta" -eq 0 ]]; then
            if python3 "$TOOLS_PY" ready --path "$out_base"; then
                _log "[$name] no new inputs; skip (keep $(python3 "$TOOLS_PY" count-lines --path "$out_base") lines)."
                echo 0 > "$status_file"
                rm -f "$cur_file" "$delta_file"
                return 0
            fi
            # manifest 存在但产物缺失 -> 视同半成品, 全量重算(processed 清零)
            _log "[$name] manifest present but output missing; full reprocess."
            rm -f "$manifest"
            cp "$cur_file" "$delta_file"
            n_delta="$n_cur"
            fresh=1
        fi

        # 新鲜运行(manifest 丢失)或全量重算: 清掉旧产物, 避免追加到遗留文件
        if [[ "$fresh" -eq 1 ]]; then
            if [[ "$SHARD_MAX_LINES" -gt 0 ]]; then
                [[ -d "$out_base" ]] && rm -rf "$out_base"
            else
                [[ -f "$out_base" ]] && rm -f "$out_base"
            fi
        fi

        # 清理上一轮中断遗留的 part 文件
        rm -f "${out_base}.part_*.jsonl"

        if [[ "$n_delta" -eq 0 ]]; then
            _log "[$name] ERROR: no inputs to process."
            echo 0 > "$status_file"
            rm -f "$cur_file" "$delta_file"
            return 0
        fi

        # 3) 单次 maker 调用, 内部以 --num-workers 占满所有核心
        _log "[$name] processing ${n_delta} new / ${n_cur} total parquet inputs (--num-workers ${PREP_PARALLEL})"
        local part; part="$(printf '%s.part_%05d.jsonl' "$out_base" 0)"
        local -a mcmd=(python3 "$script"
            --input-files $(cat "$delta_file")
            --input-dir "$input_dir"
            --output "$part"
            --shard-max-lines 0
            --num-workers "${PREP_PARALLEL}")
        if [[ "$kind" == "tts" ]]; then
            mcmd+=(--sampling-rate "$SAMPLING_RATE" --fsx-mount "$FSX_MOUNT")
        else
            mcmd+=(--latency-budget-ms "$LATENCY_BUDGET_MS")
        fi
        if [[ "$SMALL_DATA_LIMIT" -gt 0 ]]; then
            mcmd+=(--limit "$SMALL_DATA_LIMIT")
        fi
        if [[ "$MAX_AUDIO_DURATION_SEC" -gt 0 ]]; then
            mcmd+=(--max-audio-duration-sec "$MAX_AUDIO_DURATION_SEC")
        fi
        # 默认丢弃未知时长音频 (硬条件); 不再暴露开关.
        _log "[$name] maker -> ${part}"
        "${mcmd[@]}"

        # 4) 拼接 (增量追加) + 写 manifest
        python3 "$TOOLS_PY" concat --out "$out_base" \
            --shard-max-lines "$SHARD_MAX_LINES" "$part"
        python3 "$TOOLS_PY" manifest-save --manifest "$manifest" --inputs-file "$delta_file"
        _log "[$name] produced $(python3 "$TOOLS_PY" count-lines --path "$out_base") lines -> ${out_base}"
        echo 1 > "$status_file"
        rm -f "$cur_file" "$delta_file"
    }

    # ---- 预热与处理并行 (per-stage, 不串行等待) ----
    # 每个阶段的输入 root 在"该阶段 maker 启动前"于后台发起 lfs hsm_restore
    # (--no-wait 仅下发召回请求并立即返回), maker 读音频与 FSx/HSM 服务端异步召回
    # 天然重叠 —— 预热始终"领先"当前阶段所需数据, 而不是任务开始先全量扫描整卷、
    # 再串行等满。这样处理全程打满, 预热在后台悄悄追平。
    #
    # 关键: 绝不为"预热整卷"而扫描整个 FSX_MOUNT —— TTS_RAW_PATH_PREFIX 若被设成
    # /mnt/fsx/ (整卷根), os.walk 会枚举上千万文件, 与 maker 抢 FSx 元数据, 反而
    # 把处理卡死。因此整卷根直接跳过, 只预热具体的 curated 子目录。
    prewarm_bg() {
        local root="$1"
        [[ -z "$root" ]] && return 0
        if [[ "$root" == "${FSX_MOUNT}" || "$root" == "${FSX_MOUNT}/" ]]; then
            _log "prewarm: skip '${root}' (== FSX_MOUNT, 跳过整卷预热)"
            return 0
        fi
        _log "prewarm: background preload -> ${root} (--no-wait)"
        python3 "$PREWARM_PY" preload --roots "$root" \
            --dir-batch "$PREWARM_DIR_BATCH" \
            --poll-interval "$PREWARM_POLL_INTERVAL" \
            --no-wait &
    }

    tts_status="$(mktemp)"; s2s_status="$(mktemp)"
    : > "$tts_status"; : > "$s2s_status"
    _log "=== prepare_data start: tts=${TTS_DATA_PATH} s2s=${S2S_DATA_PATH} ==="
    _log "  max_audio_duration_sec=${MAX_AUDIO_DURATION_SEC} (unknown-duration audio dropped by default)"

    # ---- Step 1: TTS (预热 TTS 输入 + 可选 raw 前缀, 与 maker 并行) ----
    prewarm_bg "$TTS_DATA_PATH"
    prewarm_bg "$TTS_RAW_PATH_PREFIX"
    process_stage tts "$TTS_DATA_PATH" "$TTS_OUT" "$tts_status" "$TTS_SCRIPT"
    _log "=== (1/4) uploading TTS stage -> ${S3_BASE}/text_to_speech/ ==="
    s3_upload "${TTS_OUT}" "${S3_BASE}/text_to_speech/"

    # ---- Step 2: S2S (预热 S2S 输入, 与 maker 并行) ----
    prewarm_bg "$S2S_DATA_PATH"
    process_stage s2s "$S2S_DATA_PATH" "$S2S_OUT" "$s2s_status" "$S2S_SCRIPT"
    _log "=== (2/4) uploading S2S stage -> ${S3_BASE}/speech_to_speech_translation/ ==="
    s3_upload "${S2S_OUT}" "${S3_BASE}/speech_to_speech_translation/"

    tts_new="$(cat "$tts_status")"
    s2s_new="$(cat "$s2s_status")"
    rm -f "$tts_status" "$s2s_status"

    # ---- Step 3: Merge S2S:TTS by configurable ratios ----
    need_merge=0
    [[ "$tts_new" -eq 1 || "$s2s_new" -eq 1 ]] && need_merge=1
    if ! python3 "$TOOLS_PY" ready --path "$FINAL_OUT"; then
        need_merge=1
    fi

    if [[ "$need_merge" -eq 1 ]]; then
        _log "(3/4) Merge TTS:S2S = ${BRIDGE_RATIOS_ARR[1]:-5}:${BRIDGE_RATIOS_ARR[0]:-1} -> bridge_alignment_joint"
        if [[ "$SHARD_MAX_LINES" -gt 0 && -d "$FINAL_OUT" ]]; then
            rm -rf "$FINAL_OUT"
        elif [[ -f "$FINAL_OUT" ]]; then
            rm -f "$FINAL_OUT"
        fi
        merge_cmd=(python3 "$MERGE_SCRIPT"
            --inputs "$S2S_OUT" "$TTS_OUT"
            --ratios "${BRIDGE_RATIOS_ARR[@]}"
            --output-name bridge_alignment_joint
            --output-dir "$OUT_DIR"
            --shuffle-output)
        if [[ "$SHARD_MAX_LINES" -gt 0 ]]; then
            merge_cmd+=(--shard-max-lines "$SHARD_MAX_LINES")
        fi
        "${merge_cmd[@]}"
    else
        _log "(3/4) Merge skip: bridge_alignment_joint unchanged."
    fi
    _log "=== (3/4) uploading bridge stage -> ${S3_BASE}/bridge_alignment_joint/ ==="
    s3_upload "${FINAL_OUT}" "${S3_BASE}/bridge_alignment_joint/"

    # ---- Step 4: audio-size audit of the final training jsonl ----
    if [[ "$need_merge" -eq 1 ]]; then
        if [[ -n "$AUDIO_SIZE_REPORT" ]]; then
            python3 "$PREWARM_PY" audit --jsonl "$FINAL_OUT" --report "$AUDIO_SIZE_REPORT"
        else
            python3 "$PREWARM_PY" audit --jsonl "$FINAL_OUT"
        fi
    else
        _log "AUDIO SIZE AUDIT skipped (bridge unchanged)."
    fi

    _log "ALL STAGES DONE. final lines = $(python3 "$TOOLS_PY" count-lines --path "$FINAL_OUT") -> $FINAL_OUT"
    _log "stage artifacts uploaded to ${S3_BASE}/"
    touch "${DONE_MARKER}"
fi

# ---- 最终预热: 将 PREWARM_TARGETS 列出的阶段结果 jsonl 的全部音频预热到 FSX ----
# 跨节点互斥: 共享 FSX_MOUNT 上的原子标记, 只有一个节点实际执行预热.
# PREWARM_TASK_JSONL:
#   留空    = 跳过;
#   "auto"/"1" = 预热 PREWARM_TARGETS 列出的本流程产物 (默认 bridge_alignment_joint),
#               取值为 text_to_speech / speech_to_speech_translation /
#               bridge_alignment_joint 的任意逗号组合;
#   其它非空 = 直接作为 jsonl 文件或分片目录路径 (两者皆支持, 可多次指定).
# PREWARM_TARGETS (仅 auto 时生效): 逗号分隔的产物名, 默认 "bridge_alignment_joint".
_prewarm_mode="${PREWARM_TASK_JSONL:-}"
_prewarm_targets="${PREWARM_TARGETS:-bridge_alignment_joint}"
_resolve_targets=()
if [[ "${_prewarm_mode}" == "auto" || "${_prewarm_mode}" == "1" ]]; then
    # 逻辑名 -> 实际产物路径. FINAL_OUT/TTS_OUT/S2S_OUT 已按 SHARD_MAX_LINES 解析:
    # 单文件模式为 <name>.jsonl, 分片模式为 <name>/ 目录. 直接复用, 避免硬编码
    # ".jsonl" 缺失导致 final prewarm 找不到目标、整段 no-op (曾报
    # "skip missing target / no valid local targets to warm").
    declare -A _name2path=(
        ["bridge_alignment_joint"]="${FINAL_OUT}"
        ["text_to_speech"]="${TTS_OUT}"
        ["speech_to_speech_translation"]="${S2S_OUT}"
    )
    IFS=',' read -ra _names <<< "${_prewarm_targets}" || true
    for _n in "${_names[@]}"; do
        _n="$(echo -n "${_n}" | tr -d '[:space:]')"
        [[ -z "${_n}" ]] && continue
        if [[ -n "${_name2path[$_n]:-}" ]]; then
            _resolve_targets+=("${_name2path[$_n]}")
        else
            _resolve_targets+=("${OUT_DIR}/${_n}")
        fi
    done
elif [[ -n "${_prewarm_mode}" ]]; then
    _resolve_targets+=("${_prewarm_mode}")
fi

if [[ ${#_resolve_targets[@]} -gt 0 ]]; then
    _fsx="${FSX_MOUNT:-/mnt/fsx}"
    _prewarm_lock="${_fsx}/.prewarm_task_jsonl_done"
    # 只预热本地确实存在的阶段产物 (CPU 节点本地临时目录)
    _valid_targets=()
    for _t in "${_resolve_targets[@]}"; do
        if python3 "$TOOLS_PY" ready --path "$_t"; then
            _valid_targets+=("$_t")
        else
            _log "final prewarm: skip missing target ${_t}"
        fi
    done
    if [[ ${#_valid_targets[@]} -gt 0 ]]; then
        if ( set -C; : > "${_prewarm_lock}" ) 2>/dev/null; then
            echo "[worker] $(hostname) acquired prewarm lock — warming: ${_valid_targets[*]}"
            python3 "${PD}/prewarm_task_jsonl.py" \
                --parallel "${PREWARM_PARALLEL:-16}" \
                --poll-interval "${PREWARM_PRODUCER_POLL:-15}" \
                --timeout "${PREWARM_WAIT_TIMEOUT:-7200}" \
                --jsonl "${_valid_targets[@]}"
        else
            echo "[worker] $(hostname) skipped final prewarm (another node already warmed)."
        fi
    else
        _log "final prewarm: no valid local targets to warm."
    fi
fi

echo "[worker] $(hostname) DONE."
