#!/usr/bin/env bash
# 单卡跑测入口：在一个 Pod 上顺序跑完一个系统的推理 + 全部打分指标（口径同 README_self）。
#
# 由 ktrain/launch_train.sh 执行（framework: pytorch，1 节点）。每一步都打印
# `[bench] ==== STEP i/N ...` 并写进 ${OUT}/status.log，方便边跑边看。
#
# 环境变量：
#   SYSTEM        s4 | s5 | rep_off | rep_L1 | rep_L2 | rep_L3 | rep_L4（默认 s4）
#                   s4      S4-offline：infer_s2st.py
#                   s5      S5-streaming：infer_s5_streaming.py（S4 底座 + S5 adapter）
#                   rep_off Repro-Omni-offline：官方 SimulS2ST-Omni 离线 e2e + 复现 stage2
#                   rep_LN  Repro-Omni-LN：官方 SimulEval agent，先合复现 stage2 再挂 stage3v2，L=N
#   GPU_TAG       硬件标签（h200 / a10 / l40s），只用于输出目录与文档归属
#   OUT           输出根目录（默认 ${FSX_MOUNT}/junyaoyang/infer_bench/<RUN_ID>）
#   BENCH_EPOCHS  跑几遍推理取最好（默认 1）
#   SCORE_GPUS    打分阶段并行张数（默认 1，只在只有一张卡时用 1）
#   LIMIT         只跑前 N 句（调试用）
#   ASSETS_ROOT   资产根目录（含 eval/ 与 tts_bench/，即 H200 FSX 上 stevehguo 评测资产的副本）。
#                 设了就不读 ${FSX}/stevehguo：A10 / L40S 集群没有 FSX，资产经 S3 PVC 挂载进来。
#   PUBLISH_TO    跑完把 OUT（不含 _cache）拷到这里。S3 挂载不支持追加写（`tee -a`），所以 OUT 要放本地盘，
#                 最后再整体拷过去；此时 OUT 必须显式指到本地目录。
#   STOP_AFTER    infer = 推理跑完即结束（跳过合并/打分/汇总，用于 profile 与冒烟）
#   THINKER_BEAMS 只对 rep_L1~L4 生效：thinker 的 beam 数（默认 4，与官方默认一致）
#   SIMULS2ST_SHARED_MODEL=1  双方向在单进程顺序执行，共享已加载模型（默认 0）
#   SHARED_LIMIT_PER_DIRECTION 共享模式每方向取前 N 句，0=全部；冒烟配 STOP_AFTER=infer
#   SIMULS2ST_FM_STEPS / SIMULS2ST_FM_CFG / SIMULS2ST_FM_RESCALE_CFG / SIMULS2ST_VOICEBOX_DTYPE /
#   SIMULS2ST_FAST_TALKER（talker 逐 token 解码里的 penalty/ngram 向量化） / SIMULS2ST_PROFILE（分阶段计时）
#                 只对 rep_L1~L4 生效：给官方仓库的本地副本打补丁，改 VoiceBox 的 Euler 步数、
#                 CFG 强度与半精度（默认不设 = 原样，16 步 / cfg 2.0 / fp32）
#   LAT_BATCH     latency_eval.py 的批大小（默认 8）。词级时间戳要 output_attentions，32 层 encoder
#                 自注意力会同时留在显存里（8 条 × 32 层 × 1500² × 20 头 × 2B ≈ 22GB），H200 放得下、
#                 A10G 23GB 放不下（2026-10-23 S5/A10G 就在这一步 OOM）。A10G 用 2，L40S 用 4。
#   OMNI_THINKER_GRAPH=graph  只对 rep_L1~L4 且 SIMULS2ST_SHARED_MODEL=1 生效：thinker 的单 token
#                 解码步换成静态 KV 缓存 + CUDA Graph（Talker 早就这么做了，thinker 一直没做）。
#                 只接管 HF beam search 下 q_len=1 的前向，prefill / 全序列前向 / 音频塔不变；
#                 beams=1、batch 不等于 beams、序列超出 OMNI_THINKER_GRAPH_BUCKETS 最大档、或捕获失败
#                 都会原样退回 eager（best-effort，不影响正确性）。
#                 2026-11-13 全量实测（1193 句，三数据集，beams=4）：
#                   L40S  RTF 模型内 0.339->0.289（-14.7%）、整体墙钟 0.413->0.333（-19.4%）
#                   A10G  RTF 模型内 0.422->0.373（-11.6%）、整体墙钟 0.489->0.416（-14.9%）
#                 质量（ASR/Text BLEU、COMET、UTMOS、SIM-o、AutoPCP delta、LAAL）三数据集全部在
#                 ±0.01 量级内摆动，无方向性漂移，判定为与基线同分布；显存峰值 +7~8%（静态缓存常驻）。
#   OMNI_THINKER_GRAPH_BUCKETS  缓存容量档位，默认 384,512,768,1024,2048（取最小够用档）
#   OMNI_THINKER_GRAPH_STATS    命中/未命中/捕获耗时统计写到这个 json 路径（可选）
set -uo pipefail

# 脚本自保护：bash 是边读边执行的。脚本所在的 S3 挂载上的对象被覆盖后，正在运行的实例读下一段时会报
# "error reading input file: Stale file handle" 直接退出（2026-10-23 S5 / A10G 就这样在跑完 GPU 指标后丢了
# 2.5 小时）。先复制到本地盘再执行，之后覆盖桶里的脚本不再影响已启动的实例。
if [[ -z "${BENCH_REEXEC:-}" ]]; then
    _orig="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    _local="$(mktemp /tmp/bench_entry.XXXXXX.sh)" && cp "${_orig}" "${_local}" \
        && BENCH_REEXEC=1 BENCH_ORIG_SCRIPT="${_orig}" exec bash "${_local}" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BENCH_ORIG_SCRIPT:-${BASH_SOURCE[0]}}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY="${PYTHON:-/opt/venv/bin/python}"
FSX="${FSX_MOUNT:-/mnt/fsx}"
NVME="${NVME_MOUNT:-/nvme}"
SYSTEM="${SYSTEM:-s4}"
GPU_TAG="${GPU_TAG:-h200}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT="${OUT:-${FSX}/junyaoyang/infer_bench/${RUN_ID}}"
SCORE_GPUS="${SCORE_GPUS:-1}"
BENCH_EPOCHS="${BENCH_EPOCHS:-1}"
LIMIT="${LIMIT:-}"
STOP_AFTER="${STOP_AFTER:-}"
# LAT_BATCH：词级时间戳的 whisper 要 output_attentions，32 层 encoder 自注意力同时驻留（8 条 ≈ 22GB），
# A10G 23GB 放不下。按卡给默认值，可用环境变量覆盖。
if [[ -n "${LAT_BATCH:-}" ]]; then
    :
elif [[ "${GPU_TAG}" == "a10" ]]; then
    LAT_BATCH=2
elif [[ "${GPU_TAG}" == "l40s" ]]; then
    LAT_BATCH=4
else
    LAT_BATCH=8
fi

# 只读复用 stevehguo 的评测资产（测试清单 / 打分模型 / 依赖），不往里写任何东西
ASSETS_ROOT="${ASSETS_ROOT:-}"
PUBLISH_TO="${PUBLISH_TO:-}"
if [[ -n "${ASSETS_ROOT}" ]]; then
    SRC="${ASSETS_ROOT}/eval"
    EMOTION2VEC="${ASSETS_ROOT}/tts_bench/models/emotion2vec_plus_large"
else
    SRC="${FSX}/stevehguo/eval"
    EMOTION2VEC="${FSX}/stevehguo/tts_bench/models/emotion2vec_plus_large"
fi
# 基线报告目录：README_self 的 7 列都在第三轮报告里（S4/S5 沿用第二轮产物，Repro 为第三轮）
BASE_REPORT_DIR="${BASE_REPORT_DIR:-${SRC}/out3/_report}"
MAN="${SRC}/work/manifests/all.jsonl"
MODEL="${MODEL_DIR:-${SRC}/models/s4_ck}"
WHISPER="${SRC}/models/whisper-large-v3"
COMET_DIR="${SRC}/models/comet"
WAVLM="${SRC}/models/wavlm-base-plus-sv"
W2V="${SRC}/models/wav2vec2-large-xlsr-53"
AUTOPCP="${SRC}/models/autopcp/model.config"
SITE_COMET="${SRC}/code/site_comet"
SITE_PARA="${SRC}/code/site_para"
EVAL_PY="${REPO_ROOT}/custom/streaming_si/eval"
S5_ADAPTER="${SRC}/models/s5_adapter"
OMNI_REPO="${SRC}/code/SimulS2ST-Omni-main"          # 官方仓库（只读；models -> /fsx/stevehguo/eval/models）
OMNI_ROOT="${SRC}/models/SimulS2ST-Omni"
SITE_OMNI="${SRC}/code/site_omni"                    # transformers 4.52.3 等，官方 Omni 需要
REP_STAGE2="${SRC}/models/repro/stage2_multitask_full"
REP_STAGE3="${SRC}/models/repro/stage3v2_fluency_retrain"

case "${SYSTEM}" in
  s4)      LABEL="S4-offline";         STREAM=0 ;;
  s5)      LABEL="S5-streaming";       STREAM=1 ;;
  rep_off) LABEL="Repro-Omni-offline"; STREAM=0 ;;
  rep_L[1-4]) LABEL="Repro-Omni-L${SYSTEM#rep_L}"; STREAM=1 ;;
  *) echo "[bench][FATAL] 未知 SYSTEM=${SYSTEM}" >&2; exit 2 ;;
esac

log() { echo "[bench] $*" | tee -a "${OUT}/status.log" >&2; }
die() { echo "[bench][FATAL] $*" | tee -a "${OUT}/status.log" >&2; exit 1; }

# 续跑：
#   RESUME_FROM=score  复用已有 OUT（必须显式给 OUT），跳过清单/推理/合并/GPU 指标，只重跑打分 + 汇总
#   RESUME_FROM=lat    同上跳过清单/推理/合并，但在 GPU 指标阶段只补跑 latency（LAAL），然后重跑打分 + 汇总
RESUME_FROM="${RESUME_FROM:-}"
mkdir -p "${OUT}"/{infer,asr,utmos,para,lat,report,logs}

# site_omni（transformers 4.52.3 的 site-packages 覆盖层）在 S3 挂载上是 148MB / 6134 个小文件，
# Python 逐个 import 时每次 open 都要走 mountpoint-s3 的网络往返：实测 `import transformers`
# 99.6s，解到本地盘只要 0.57s（2026-11-11，A10G，见 docs 里冷启动拆解）。
# 挂了 site_omni.tar（单对象，一次性打包）就先解到本地盘再用；没有就退回原来的挂载路径，行为不变。
if [[ "${SYSTEM}" == rep_* ]]; then
    SITE_OMNI_TAR="${SITE_OMNI}.tar"
    if [[ -f "${SITE_OMNI_TAR}" ]]; then
        _site_local="${OUT}/_cache/site_omni"
        if [[ ! -f "${_site_local}/.extracted" ]]; then
            _t0=$(date +%s)
            rm -rf "${_site_local}" && mkdir -p "${_site_local}" \
                && tar -C "${_site_local}" -xf "${SITE_OMNI_TAR}" && touch "${_site_local}/.extracted" \
                || { rm -rf "${_site_local}"; log "[WARN] site_omni.tar 解压失败，回退到 S3 挂载路径"; }
            [[ -f "${_site_local}/.extracted" ]] \
                && log "site_omni 解到本地盘：$(( $(date +%s) - _t0 ))s -> ${_site_local}"
        fi
        [[ -f "${_site_local}/.extracted" ]] && SITE_OMNI="${_site_local}"
    fi
fi
if [[ -n "${RESUME_FROM}" ]]; then
    echo "[bench] ---- 续跑 RESUME_FROM=${RESUME_FROM}（$(date '+%F %T')）----" >> "${OUT}/status.log"
else
    : > "${OUT}/status.log"
fi
skip() { [[ "${RESUME_FROM}" == "score" || "${RESUME_FROM}" == "lat" ]]; }
only_lat() { [[ "${RESUME_FROM}" == "lat" ]]; }
only_score() { [[ "${RESUME_FROM}" == "score" ]]; }

PUBLISHED=0

publish() {
    [[ -n "${PUBLISH_TO}" ]] || return 0
    log "拷贝产物 ${OUT} -> ${PUBLISH_TO}（不含 _cache）"
    mkdir -p "${PUBLISH_TO}"
    # S3 挂载只能顺序写新文件、不能改时间戳/权限：用 cp -r（不保留属性），不用 tar / rsync -a
    cp_rc=0
    for f in $(ls -A "${OUT}"); do
        [[ "${f}" == "_cache" ]] && continue
        cp -r "${OUT}/${f}" "${PUBLISH_TO}/" || cp_rc=1
    done
    [[ ${cp_rc} -eq 0 ]] && { log "拷贝完成"; PUBLISHED=1; } || log "[WARN] 部分文件拷贝失败，产物仍在本地 ${OUT}"
}
publish_partial() {
    local rc=$?
    [[ -n "${PUBLISH_TO:-}" && "${PUBLISHED}" != 1 ]] || return 0
    echo "[bench] 未正常结束（rc=${rc}），保存日志与已有的中间结果 -> ${PUBLISH_TO}/_partial" >&2
    mkdir -p "${PUBLISH_TO}/_partial/infer" 2>/dev/null || return 0
    for d in logs asr utmos para lat report; do
        [[ -d "${OUT}/${d}" ]] && cp -r "${OUT}/${d}" "${PUBLISH_TO}/_partial/" 2>/dev/null
    done
    for f in status.log summary.md manifest.jsonl; do
        [[ -f "${OUT}/${f}" ]] && cp "${OUT}/${f}" "${PUBLISH_TO}/_partial/" 2>/dev/null
    done
    for f in "${OUT}"/infer/*.jsonl; do [[ -f "${f}" ]] && cp "${f}" "${PUBLISH_TO}/_partial/infer/" 2>/dev/null; done
    [[ -f "${OUT}/infer/_seg/results.jsonl" ]] && cp "${OUT}/infer/_seg/results.jsonl" "${PUBLISH_TO}/_partial/infer/seg_results.jsonl" 2>/dev/null
    if [[ -n "${SIMULS2ST_PROBE_MODES:-}${SIMULS2ST_BOUNDARY_PROBE:-}" && -d "${OUT}/infer" ]]; then
        cp -r "${OUT}/infer/." "${PUBLISH_TO}/_partial/infer/" 2>/dev/null
    fi
    return 0
}
trap publish_partial EXIT

# ---------------------------------------------------------------- 0. 环境
step=0; total=7
step=$((step + 1))
log "==== STEP ${step}/${total} 环境检查（$SYSTEM = $LABEL / $GPU_TAG / run $RUN_ID）"
log "OUT=${OUT}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | tee -a "${OUT}/status.log"
# 清单里的音频路径是 /fsx/stevehguo/eval/...：H200 上 Pod 挂在 /mnt/fsx -> 软链 /fsx；
# 没有 FSX 的集群把 /fsx/stevehguo/eval 指到资产副本。都不改任何清单文件。
if [[ -n "${ASSETS_ROOT}" ]]; then
    mkdir -p /fsx/stevehguo && ln -sfn "${SRC}" /fsx/stevehguo/eval && log "建软链 /fsx/stevehguo/eval -> ${SRC}"
elif [[ ! -e /fsx ]]; then ln -s "${FSX}" /fsx && log "建软链 /fsx -> ${FSX}"; fi

# Repro-Omni 的官方仓库里 models/ 是指向 eval/models 的符号链接；没有 FSX 时 ASSETS_ROOT 经 S3
# 挂载（mountpoint-s3），S3 没有符号链接这个对象类型，挂载层也建不出符号链接（S4/S5 不用这层，
# 不受影响）。官方仓库本身很小，复制到本地可写目录，在本地重建这个链接，仅 rep_off / rep_L* 需要。
if [[ -n "${ASSETS_ROOT}" ]] && [[ "${SYSTEM}" == rep_off || "${SYSTEM}" == rep_L[1-4] ]]; then
    LOCAL_OMNI="/tmp/omni_repo_local"
    if [[ ! -f "${LOCAL_OMNI}/.staged" ]]; then
        log "复制官方 Omni 仓库到本地（$(du -sh "${OMNI_REPO}" 2>/dev/null | cut -f1)）-> ${LOCAL_OMNI}（重建 models/ 符号链接）"
        mkdir -p "${LOCAL_OMNI}"
        cp -r "${OMNI_REPO}/." "${LOCAL_OMNI}/"
        ln -sfn "${OMNI_ROOT%/}/.." "${LOCAL_OMNI}/models"   # -> ${SRC}/models（与源项目 models -> eval/models 语义一致）
        touch "${LOCAL_OMNI}/.staged"
    fi
    OMNI_REPO="${LOCAL_OMNI}"
    # 可选：给本地副本打运行期补丁（VoiceBox 步数 / CFG / 半精度），只在设了环境变量时生效
    if [[ -n "${SIMULS2ST_FM_STEPS:-}${SIMULS2ST_FM_CFG:-}${SIMULS2ST_FM_RESCALE_CFG:-}${SIMULS2ST_VOICEBOX_DTYPE:-}${SIMULS2ST_FAST_TALKER:-}${SIMULS2ST_PROFILE:-}${SIMULS2ST_PROFILE_TALKER:-}${SIMULS2ST_TALKER_ACCEL:-}${SIMULS2ST_TRACE_CODES:-}" ]]; then
        "${PY}" "${REPO_ROOT}/pipelines/streaming_si/patch_omni_repo.py" "${OMNI_REPO}" \
            || die "patch_omni_repo.py 失败"
    fi
fi
[[ -f "${MAN}" ]] || die "测试清单不存在：${MAN}"
[[ -d "${MODEL}" ]] || die "模型目录不存在：${MODEL}"
[[ -d "${WHISPER}" ]] || die "Whisper 不存在：${WHISPER}"
# 只读目录：不写 __pycache__、HF/COMET 不往人家的目录里写东西
export PYTHONDONTWRITEBYTECODE=1
# HF / torch.hub 缓存：复制到我们自己的目录再用。
#   * 不能直接用 ${SRC}/hf：打分时 tokenizer 会去 HF 查 mistral regex 补丁（联网），
#     开 HF_HUB_OFFLINE 会直接抛 OfflineModeIsEnabled；不开就会写进别人的目录。
#   * 也不能只读软链：同理，一旦需要下载就会写。
OUR_HF="${OUT}/_cache/hf"
OUR_TORCH="${OUT}/_cache/torch_hub"
if [[ ! -f "${OUR_HF}/.copied" ]]; then
  log "复制 HF 缓存（$(du -sh "${SRC}/hf" | cut -f1)）-> ${OUR_HF}"
  mkdir -p "${OUR_HF}"
  rsync -a "${SRC}/hf/" "${OUR_HF}/" && touch "${OUR_HF}/.copied"
fi
if [[ ! -f "${OUR_TORCH}/.copied" ]]; then
  log "复制 torch.hub 缓存（$(du -sh "${SRC}/models/torch_hub" | cut -f1)）-> ${OUR_TORCH}"
  mkdir -p "${OUR_TORCH}"
  rsync -a "${SRC}/models/torch_hub/" "${OUR_TORCH}/" && touch "${OUR_TORCH}/.copied"
fi
export HF_HOME="${OUR_HF}" TORCH_HOME="${OUR_TORCH}"
unset HF_HUB_OFFLINE || true
export PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export COSYVOICE_REPO="${COSYVOICE_REPO:-/opt/CosyVoice}"
export COSYVOICE_SPEECH_TOKENIZER="${COSYVOICE_SPEECH_TOKENIZER:-${SRC}/models/cosyvoice3/speech_tokenizer_v3.onnx}"
export PYTHONPATH="${REPO_ROOT}/custom${PYTHONPATH:+:${PYTHONPATH}}"
log "PYTHONPATH=$PYTHONPATH"
log "MODEL=${MODEL}"

if skip; then
    if only_lat; then
        step=$((step + 3))          # 清单 / 推理 / 合并
        log "跳过 清单 / 推理 / 合并（复用 ${OUT} 已有结果），只补跑 latency"
    else
        step=$((step + 4))
        log "跳过 清单 / 推理 / 合并 / GPU 指标（复用 ${OUT} 已有结果）"
    fi
    t_infer=$(grep -oE "推理总用时 [0-9]+" "${OUT}/status.log" | tail -1 | grep -oE "[0-9]+" || echo 0)
else
# ---------------------------------------------------------------- 1. 清单
step=$((step + 1))
log "==== STEP ${step}/${total} 生成本次清单（预热句 + 全部 1193 句）"
"${PY}" - "${MAN}" "${OUT}/manifest.jsonl" "$LIMIT" <<'PY'
import json, sys
src, dst, limit = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in open(src, encoding='utf-8') if l.strip()]
if limit:
    rows = rows[:int(limit)]
w = dict(rows[0]); w['id'] = '_warmup_' + w['id']
with open(dst, 'w', encoding='utf-8') as f:
    for r in [w] + rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'[bench] 清单 {len(rows)} 句 + 1 预热句 -> {dst}')
PY
[[ $? -eq 0 ]] || die "生成清单失败"

# ---------------------------------------------------------------- 2/3. 推理
step=$((step + 1))
log "==== STEP ${step}/${total} 推理（$SYSTEM，$BENCH_EPOCHS 遍）"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 2000 \
    > "${OUT}/logs/gpu_infer.csv" 2>/dev/null &
GPUMON=$!
t_infer0=$(date +%s)
for ep in $(seq 1 "${BENCH_EPOCHS}"); do
    d="${OUT}/infer"
    [[ "${BENCH_EPOCHS}" -gt 1 ]] && d="${OUT}/infer_e${ep}"
    log "推理第 ${ep}/${BENCH_EPOCHS} 遍 -> ${d}"
    t_ep0=$(date +%s)
    case "${SYSTEM}" in
      s4)
        "${PY}" "${REPO_ROOT}/custom/streaming_si/inference/infer_s2st.py" \
            --model_dir "${MODEL}" \
            --input_jsonl "${OUT}/manifest.jsonl" \
            --out_dir "${d}" \
            --cosyvoice_repo "${COSYVOICE_REPO}" \
            --device cuda:0 > "${OUT}/logs/infer.${ep}.log" 2>&1
        rc=$?
        ;;
      s5)
        "${PY}" "${REPO_ROOT}/custom/streaming_si/inference/infer_s5_streaming.py" \
            --model_dir "${MODEL}" --adapter "${S5_ADAPTER}" \
            --input_jsonl "${OUT}/manifest.jsonl" \
            --out_dir "${d}" \
            --cosyvoice_repo "${COSYVOICE_REPO}" \
            --device cuda:0 > "${OUT}/logs/infer.${ep}.log" 2>&1
        rc=$?
        ;;
      rep_off)
        # 与第三轮 run_worker3.sh repoff 逐字相同的调用，只换输出目录
        ( cd "${OMNI_REPO}" && PYTHONPATH="${SITE_OMNI}:${OMNI_REPO}:${OMNI_REPO}/external/SimulEval:${OMNI_REPO}/src/train:${PYTHONPATH}" \
          "${PY}" "${EVAL_PY}/run_omni_offline.py" \
            --manifest "${OUT}/manifest.jsonl" --shard 0 --num_shards 1 \
            --out_dir "${d}" --model_root "${OMNI_ROOT}" --checkpoint_path "${REP_STAGE2}" \
            --device cuda:0 ) > "${OUT}/logs/infer.${ep}.log" 2>&1
        rc=$?
        ;;
      rep_L[1-4])
        L="${SYSTEM#rep_L}"; rc=0
        if [[ "${SIMULS2ST_SHARED_MODEL:-0}" == 1 ]]; then
          SHARED_RUNNER="${EVAL_PY}/omni_shared_run.py"
          [[ -n "${SIMULS2ST_PROBE_MODES:-}" ]] && SHARED_RUNNER="${EVAL_PY}/omni_graph_probe.py"
          [[ "${SIMULS2ST_BOUNDARY_PROBE:-0}" == 1 ]] && SHARED_RUNNER="${EVAL_PY}/omni_kv_boundary_probe.py"
          [[ "${OMNI_THINKER_GRAPH:-off}" == graph ]] && SHARED_RUNNER="${EVAL_PY}/omni_thinker_graph_run.py"
          ( cd "${OMNI_REPO}" && SIMULS2ST_MERGE_ADAPTERS_FROM="${REP_STAGE2}" \
            PYTHONPATH="${SITE_OMNI}:${OMNI_REPO}:${OMNI_REPO}/external/SimulEval:${OMNI_REPO}/src/train" \
            "${PY}" "${SHARED_RUNNER}" \
              --manifest "${OUT}/manifest.jsonl" --out-dir "${d}" \
              --checkpoint-path "${REP_STAGE3}" --latency-multiplier "${L}" \
              --thinker-num-beams "${THINKER_BEAMS:-4}" \
              --limit-per-direction "${SHARED_LIMIT_PER_DIRECTION:-0}" \
          ) > "${OUT}/logs/infer.${ep}.shared.log" 2>&1
          rc=$?
        else
        for dir in en2zh zh2en; do
          W="${d}/run_${dir}"; rm -rf "${W}"; mkdir -p "${W}"
          # 每个方向单独起进程、单独预热一句（同 run_worker3.sh）
          "${PY}" - "${OUT}/manifest.jsonl" "${dir}" "${W}" <<'PY3'
import json, sys
m, d, w = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in open(m, encoding='utf-8') if l.strip()]
rows = [r for r in rows if f"{r['src_lang']}2{r['tgt_lang']}" == d and not r['id'].startswith('_warmup_')]
if rows:
    wu = dict(rows[0]); wu['id'] = '_warmup_' + wu['id']; rows = [wu] + rows
with open(f'{w}/ids.txt', 'w') as fi, open(f'{w}/source.txt', 'w') as fs, open(f'{w}/target.txt', 'w') as ft:
    for r in rows:
        fi.write(r['id'] + '\n'); fs.write(r['audio'] + '\n'); ft.write(r['reference'].replace('\n', ' ') + '\n')
print(f'[bench] {d}: {len(rows)} 条（含预热）')
PY3
          n=$(wc -l < "${W}/ids.txt"); [[ "${n}" -eq 0 ]] && continue
          if [[ ${dir} == en2zh ]]; then SL=English; TL=Chinese; TOK=zh; UNIT=char; else SL=Chinese; TL=English; TOK=13a; UNIT=word; fi
          log "  ${dir}: ${n} 条，L=${L}"
          # 参数与 run_worker3.sh 逐字相同（官方 README「Streaming S2ST」默认值，只改 L 与语言口径）
          ( cd "${OMNI_REPO}" && SIMULS2ST_MERGE_ADAPTERS_FROM="${REP_STAGE2}" \
            PYTHONPATH="${SITE_OMNI}:${OMNI_REPO}:${OMNI_REPO}/external/SimulEval:${OMNI_REPO}/src/train" \
            "${PY}" "${EVAL_PY}/omni_simuleval_wrap.py" \
              --agent src/agents/simuleval_omni_talker_s2st_agent.py \
              --source-segment-size 1000 \
              --model-name-or-path models/SimulS2ST-Omni/offline \
              --checkpoint-path "${REP_STAGE3}" \
              --source "${W}/source.txt" --target "${W}/target.txt" --output "${W}" \
              --source-lang ${SL} --target-lang ${TL} \
              --latency-multiplier ${L} \
              --sacrebleu-tokenizer ${TOK} --eval-latency-unit ${UNIT} \
              --history-window-turns 28 --history-overlap-turns 16 \
              --prompt-source-chunks 2 --prompt-generated-chunks 1 \
              --thinker-max-new-tokens 256 --talker-max-new-tokens 500 \
              --thinker-no-sample --talker-no-sample \
              --thinker-num-beams ${THINKER_BEAMS:-4} \
              --thinker-repetition-penalty 1.2 --thinker-no-repeat-ngram-size 5 \
              --talker-repetition-penalty 1.4 --talker-no-repeat-ngram-size 5 \
              --computation-aware \
              --no-scoring ) > "${OUT}/logs/infer.${ep}.${dir}.log" 2>&1
          r=$?; [[ ${r} -ne 0 ]] && rc=${r}
          log "  ${dir}: rc=${r}，实例 $(grep -c . "${W}/instances.log" 2>/dev/null || echo 0)/${n}"
        done
        fi
        ;;
    esac
    log "推理第 ${ep} 遍 rc=${rc}，用时 $(( $(date +%s) - t_ep0 ))s"
    [[ ${rc} -eq 0 ]] || { tail -30 "${OUT}"/logs/infer.${ep}*.log | tee -a "${OUT}/status.log"; die "推理失败"; }
done
kill ${GPUMON} 2>/dev/null
t_infer=$(( $(date +%s) - t_infer0 ))
log "推理总用时 ${t_infer}s"

if [[ "${STOP_AFTER}" == "infer" ]]; then
    log "STOP_AFTER=infer：推理完成，跳过合并 / GPU 指标 / 打分 / 汇总"
    publish
    log "==== 提前结束（STOP_AFTER=infer），产物在 ${PUBLISH_TO:-${OUT}}"
    exit 0
fi

# ---------------------------------------------------------------- 3. 合并（去预热）
step=$((step + 1))
log "==== STEP ${step}/${total} 合并结果（剔除预热句）"
case "${SYSTEM}" in
  s4)      RAW="${OUT}/infer/results.jsonl" ;;
  rep_off) RAW="${OUT}/infer/results.00.jsonl" ;;
  s5)      # 整理成统一分段表示（LAAL 用），同 run_post2.sh
           "${PY}" "${EVAL_PY}/stream_segments.py" s5 --results "${OUT}/infer/results.jsonl" \
               --out "${OUT}/infer/_seg" 2>&1 | tee -a "${OUT}/status.log"
           RAW="${OUT}/infer/_seg/results.jsonl" ;;
  rep_L*)  "${PY}" "${EVAL_PY}/stream_segments.py" omni --runs "${OUT}/infer/run_*" \
               --out "${OUT}/infer/_seg" 2>&1 | tee -a "${OUT}/status.log"
           RAW="${OUT}/infer/_seg/results.jsonl" ;;
esac
[[ -s "${RAW}" ]] || die "推理结果为空：${RAW}"
"${PY}" - "${OUT}" "${MAN}" "${RAW}" <<'PY' | tee -a "${OUT}/status.log"
import json, sys
out, man_p, raw = sys.argv[1], sys.argv[2], sys.argv[3]
man = {json.loads(l)['id']: json.loads(l) for l in open(man_p, encoding='utf-8') if l.strip()}
rows = [json.loads(l) for l in open(raw, encoding='utf-8') if l.strip()]
rows = [r for r in rows if r['id'] in man or r['id'].startswith('_warmup_')]
rows = [r for r in rows if not r['id'].startswith('_warmup_')]
for r in rows:
    r.setdefault('tgt_lang', man[r['id']]['tgt_lang'])
    r.setdefault('source_sec', man[r['id']]['source_sec'])
with open(f'{out}/infer/merged.jsonl', 'w', encoding='utf-8') as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
ok = sum(1 for r in rows if r.get('status') in (None, 'ok'))
print(f'合并 {len(rows)} 条（ok {ok}，缺 {len(set(man) - {r["id"] for r in rows})}）')
PY
fi     # 结束「完整跑」分支：GPU 指标单独一段，lat 续跑时跳过 asr/utmos/para 只跑 latency

# ---------------------------------------------------------------- 4. GPU 指标（ASR / UTMOS / 副语言）
if only_score; then
  step=$((step + 1))
  log "==== STEP ${step}/${total} ASR / UTMOS / 副语言（跳过：RESUME_FROM=score）"
else
step=$((step + 1))
log "==== STEP ${step}/${total} ASR / UTMOS / 副语言（${SCORE_GPUS} 卡并行）"
for ((g = 0; g < SCORE_GPUS; g++)); do
  (
    export CUDA_VISIBLE_DEVICES=$g
    # 词级时间戳的 whisper 要 output_attentions，显存随 batch 线性涨；A10G 上必须调小（LAT_BATCH=2）
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    if ! only_lat; then
    # 转写（ASR-BLEU / ASR-COMET）
    "${PY}" "${EVAL_PY}/run_asr.py" --wav_jsonl "${OUT}/infer/merged.jsonl" \
        --manifest "${MAN}" --out "${OUT}/asr/${SYSTEM}.$g.jsonl" \
        --model "${WHISPER}" --device cuda:0 --shard $g --num_shards "${SCORE_GPUS}" --batch 8 \
        > "${OUT}/logs/asr.$g.log" 2>&1
    echo "[bench] asr shard $g rc=$?" | tee -a "${OUT}/status.log"
    # UTMOS（合成语音 + 源音频参照）
    "${PY}" "${EVAL_PY}/run_utmos.py" --wav_jsonl "${OUT}/infer/merged.jsonl" --wav_key wav \
        --out "${OUT}/utmos/${SYSTEM}.$g.jsonl" --shard $g --num_shards "${SCORE_GPUS}" \
        > "${OUT}/logs/utmos.$g.log" 2>&1
    echo "[bench] utmos shard $g rc=$?" | tee -a "${OUT}/status.log"
    "${PY}" "${EVAL_PY}/run_utmos.py" --wav_jsonl "${MAN}" --wav_key audio \
        --out "${OUT}/utmos/source.$g.jsonl" --shard $g --num_shards "${SCORE_GPUS}" \
        > "${OUT}/logs/utmos_source.$g.log" 2>&1
    echo "[bench] utmos-source shard $g rc=$?" | tee -a "${OUT}/status.log"
    # 副语言（SIM-o / ESim / AutoPCP）
    PYTHONPATH="${SITE_PARA}:${PYTHONPATH}" "${PY}" "${EVAL_PY}/run_para_metrics.py" \
        --results "${OUT}/infer/merged.jsonl" --manifest "${MAN}" --out "${OUT}/para/${SYSTEM}.$g.jsonl" \
        --wavlm "${WAVLM}" --emotion2vec "${EMOTION2VEC}" --autopcp "${AUTOPCP}" --w2v "${W2V}" \
        --device cuda:0 --batch 8 --shard $g --num_shards "${SCORE_GPUS}" \
        > "${OUT}/logs/para.$g.log" 2>&1
    echo "[bench] para shard $g rc=$?" | tee -a "${OUT}/status.log"
    fi
    # 流式系统：带词级时间戳的转写 -> LAAL / StartOffset / EndOffset（CA 与非 CA）
    if [[ "${STREAM}" == 1 ]]; then
      "${PY}" "${EVAL_PY}/latency_eval.py" --results "${OUT}/infer/merged.jsonl" --manifest "${MAN}" \
          --model "${WHISPER}" --out "${OUT}/lat/${SYSTEM}.$g.jsonl" --shard $g --num_shards "${SCORE_GPUS}" \
          --batch "${LAT_BATCH}" \
          > "${OUT}/logs/lat.$g.log" 2>&1
      echo "[bench] latency shard $g rc=$?" | tee -a "${OUT}/status.log"
    fi
  ) &
done
wait
log "GPU 指标阶段完成"
fi     # 结束 GPU 指标段

# ---------------------------------------------------------------- 5. 打分（BLEU / chrF / COMET）
step=$((step + 1))
log "==== STEP ${step}/${total} 打分（三个数据集各一份报告）"
CK=$(ls "${COMET_DIR}"/models--Unbabel--wmt22-comet-da/snapshots/*/checkpoints/model.ckpt 2>/dev/null | head -1)
[[ -n "${CK}" ]] || log "[WARN] 找不到 COMET checkpoint，COMET 分数会是空"
j() { ls $1 2>/dev/null | paste -sd, -; }   # glob 展开即完整路径，逗号拼接
for ds in realsi_en2zh realsi_zh2en acl6060_en2zh; do
  SCORE_MAN="${SRC}/work/manifests/${ds}.jsonl"
  if [[ -n "${LIMIT}" ]]; then
    # 冒烟：只保留本次实际跑过的 id，否则缺句会被当成空译文把 BLEU 压到 0
    SCORE_MAN="${OUT}/manifest.${ds}.jsonl"
    "${PY}" - "${SRC}/work/manifests/${ds}.jsonl" "${OUT}/infer/merged.jsonl" "${SCORE_MAN}" <<'PY2'
import json, sys
man, done, out = sys.argv[1], sys.argv[2], sys.argv[3]
ids = {json.loads(l)['id'] for l in open(done, encoding='utf-8') if l.strip()}
rows = [l for l in open(man, encoding='utf-8') if l.strip() and json.loads(l)['id'] in ids]
open(out, 'w', encoding='utf-8').writelines(rows)
print(f'[bench] LIMIT 模式：{out} 保留 {len(rows)} 句')
PY2
    [[ -s "${SCORE_MAN}" ]] || { log "LIMIT 模式下 ${ds} 没有命中任何句子，跳过"; continue; }
  fi
  ( export CUDA_VISIBLE_DEVICES=0
    PYTHONPATH="${SITE_COMET}:${PYTHONPATH}" "${PY}" "${EVAL_PY}/score_eval.py" \
      --manifest "${SCORE_MAN}" \
      --sys "${LABEL}-${GPU_TAG}=${OUT}/infer/merged.jsonl:$(j "${OUT}/asr/${SYSTEM}.*.jsonl"):$(j "${OUT}/utmos/${SYSTEM}.*.jsonl"):$(j "${OUT}/lat/${SYSTEM}.*.jsonl"):$(j "${OUT}/para/${SYSTEM}.*.jsonl")" \
      --source_utmos "$(j "${OUT}/utmos/source.*.jsonl")" \
      --comet_model "${CK}" --out "${OUT}/report/${ds}.json" \
      > "${OUT}/logs/score.${ds}.log" 2>&1
    echo "[bench] score ${ds} rc=$?" | tee -a "${OUT}/status.log"
    grep -E "comet =|^${LABEL}" "${OUT}/logs/score.${ds}.log" | tee -a "${OUT}/status.log"
  ) &
done
wait

# ---------------------------------------------------------------- 6. 汇总
step=$((step + 1))
log "==== STEP ${step}/${total} 汇总"
"${PY}" - "${OUT}" "${GPU_TAG}" "${LABEL}" "${t_infer}" "${BASE_REPORT_DIR}" <<'PY' | tee "${OUT}/summary.md"
import json, os, statistics, sys
out, tag, system, t_infer, base_dir = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
rows = [json.loads(l) for l in open(f'{out}/infer/merged.jsonl', encoding='utf-8') if l.strip()]
walls = [r['wall_sec'] for r in rows if r.get('wall_sec')]
src_sec = sum(r.get('source_sec') or 0 for r in rows)
peak = None
g = f'{out}/logs/gpu_infer.csv'
if os.path.isfile(g):
    mem = [int(x.split(',')[1]) for x in open(g) if x.count(',') >= 2]
    peak = max(mem) if mem else None
print(f'# {system} @ {tag} 单卡跑测\n')
print(f'- 句数 {len(rows)}，源音频 {src_sec/60:.1f} 分钟')
print(f'- 推理总墙钟 {t_infer}s（含读文件/写 wav），语料级 RTF {t_infer/src_sec:.3f}')
if walls:
    print(f'- 模型内计时：Σwall {sum(walls):.1f}s，RTF {sum(walls)/src_sec:.3f}，'
          f'单句均值 {statistics.fmean(walls):.2f}s，中位 {statistics.median(walls):.2f}s，'
          f'p90 {sorted(walls)[int(len(walls)*0.9)]:.2f}s，最大 {max(walls):.2f}s')
if peak:
    print(f'- 显存峰值 {peak} MiB')
print(f'- 失败 {sum(1 for r in rows if r.get("status") not in (None, "ok"))}，'
      f'空译文 {sum(1 for r in rows if not (r.get("text") or "").strip())}，'
      f'空语音 {sum(1 for r in rows if (r.get("output_sec") or 0) <= 0.05)}')
print('\n## 打分明细\n')
for ds, label in (('realsi_en2zh', 'RealSI en→zh'), ('realsi_zh2en', 'RealSI zh→en'),
                  ('acl6060_en2zh', 'ACL 60/60 en→zh')):
    p = f'{out}/report/{ds}.json'
    if not os.path.isfile(p):
        print(f'### {label}\n(缺)\n'); continue
    rep = json.load(open(p, encoding='utf-8'))
    s = next(iter(rep['systems'].values()))
    print(f'### {label}（{rep["n_utt"]} 句）\n')
    print('| 指标 | 本机 | README_self 基线 |\n|---|---|---|')
    base_p = f'{base_dir}/{ds}.json'
    b = {}
    if os.path.isfile(base_p):
        b = json.load(open(base_p, encoding='utf-8'))['systems'].get(system, {})
    for k, name in (('asr_bleu', 'ASR-BLEU'), ('asr_chrf', 'ASR-chrF'), ('asr_comet', 'ASR-COMET'),
                    ('text_bleu', 'Text-BLEU'), ('text_chrf', 'Text-chrF'), ('text_comet', 'Text-COMET'),
                    ('utmos', 'UTMOS'), ('sim_o', 'SIM-o'), ('sim_o_delta', 'SIM-o delta'),
                    ('esim_delta', 'ESim delta'), ('autopcp_delta', 'AutoPCP delta'),
                    ('laal_nca_bow', 'LAAL'), ('laal_ca_bow', 'LAAL (CA)'), ('laal_ca_eow', 'LAAL-EOW (CA)'),
                    ('start_offset_ca', 'StartOffset (CA)'), ('end_offset_ca', 'EndOffset (CA)'),
                    ('rtf_corpus', 'RTF（语料级）'), ('rtf_median', 'RTF（中位）'),
                    ('mean_wall_sec', '平均单句耗时'), ('len_ratio', '时长比')):
        v, bv = s.get(k), b.get(k)
        if v is None and bv is None:
            continue                                  # 离线系统没有延迟指标
        fv = '-' if v is None else (f'{v:.4f}' if isinstance(v, float) and abs(v) < 10 else f'{v}')
        fb = '-' if bv is None else (f'{bv:.4f}' if isinstance(bv, float) and abs(bv) < 10 else f'{bv}')
        print(f'| {name} | {fv} | {fb} |')
    print()
print(f'\n基线来源：`{base_dir}/*.json` 的 `{system}` 列（README_self.md 同源；7 卡各跑一片）。')
PY

publish
log "==== 全部完成，产物在 ${PUBLISH_TO:-${OUT}}"
