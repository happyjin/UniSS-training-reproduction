#!/usr/bin/env bash
# 一键导出 streaming_si 的两个基座模型（HF 目录，swift 可直接 --model 加载），并可选上传 S3。
#
#   streaming_si     Thinker Qwen3-0.6B  + Talker 0.1B（8 层，随机初始化）
#   streaming_si_4b  Thinker Qwen3.5-4B  + Talker 0.6B（28 层，用 Qwen3-0.6B 的 transformer 层初始化）
#   两者共用 Voxtral-Mini-4B-Realtime-2602 的音频塔（只下 HF 格式的 model.safetensors，不下 consolidated）
#
# 下载量：Qwen3-0.6B 1.4G + Qwen3.5-4B 8.7G + Voxtral 8.3G + Fun-CosyVoice3（flow/hift/campplus）约 1.5G
#
# 用法（在 GPU 服务器或任意有 ~40GB 空闲磁盘、能访问 HuggingFace 的机器上）：
#   bash pipelines/streaming_si/export_base_models.sh                       # 只导出
#   S3_DEST=s3://stevehguo-ap-southeast-3/speech_model_backup/stevehguo/model_checkpoint/streaming_si_swift \
#       bash pipelines/streaming_si/export_base_models.sh                   # 导出后上传
#
# 可选环境变量：
#   WORK_DIR           下载与导出的根目录（默认 ./streaming_si_export）
#   VARIANTS           要导出的变体，空格分隔（默认 "streaming_si streaming_si_4b"）
#   THINKER_REPO / THINKER_4B_REPO / VOXTRAL_REPO   HF 仓库 ID（见下方默认值）
#   HF_ENDPOINT        HF 镜像（如 https://hf-mirror.com）
#   COSYVOICE_DIR      Fun-CosyVoice3-0.5B 目录；给定时把 flow/HiFT 打包进模型目录（仅推理用）
#   COSYVOICE_REPO_ID  不给 COSYVOICE_DIR 时从 HF 下载 Fun-CosyVoice3（设为空串则不打包声码器）
#   DTYPE              导出精度（默认 bfloat16）
#   S3_DEST            上传目的地前缀；各变体上传到 ${S3_DEST}/base_<变体>/
#   AWS_REGION         默认 ap-southeast-3
#   KEEP_DOWNLOADS=1   上传后保留下载的基座
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"
: "${WORK_DIR:=${PWD}/streaming_si_export}"
: "${VARIANTS:=streaming_si streaming_si_4b}"
: "${THINKER_REPO:=Qwen/Qwen3-0.6B}"
: "${THINKER_4B_REPO:=Qwen/Qwen3.5-4B}"
: "${VOXTRAL_REPO:=mistralai/Voxtral-Mini-4B-Realtime-2602}"
: "${COSYVOICE_REPO_ID=FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
: "${DTYPE:=bfloat16}"
: "${AWS_REGION:=ap-southeast-3}"
PRE="${WORK_DIR}/pretrained"
OUT="${WORK_DIR}/models"
mkdir -p "${PRE}" "${OUT}"

log() { echo "[export_base] $*" >&2; }
die() { echo "[export_base][FATAL] $*" >&2; exit 1; }

"${PYTHON}" -c 'import huggingface_hub, safetensors, transformers' 2>/dev/null \
    || die "需要 huggingface_hub / safetensors / transformers（pip install -r requirements.txt）"
avail_gb=$(df -Pk "${WORK_DIR}" | awk 'NR==2 {printf "%d", $4/1024/1024}')
[[ "${avail_gb}" -ge 40 ]] || log "WARN: ${WORK_DIR} 只剩 ${avail_gb}GB，两个变体全量约需 40GB"

# ---------------------------------------------------------------------------
# 1. 下载基座（已下载的跳过；Voxtral 只取音频塔所在分片，约为整仓的 1/5）
# ---------------------------------------------------------------------------
fetch() {   # fetch <repo> <dir> [allow_patterns...]
    local repo="$1" dir="$2"; shift 2
    if [[ -f "${dir}/.fetched" ]]; then log "已存在，跳过: ${dir}"; return; fi
    log "下载 ${repo} -> ${dir}"
    "${PYTHON}" - "${repo}" "${dir}" "$@" <<'PY'
import json, os, sys
from huggingface_hub import hf_hub_download, snapshot_download
repo, out, *allow = sys.argv[1:]
os.makedirs(out, exist_ok=True)
if allow == ['__voxtral__']:
    small = ['config.json', 'generation_config.json', 'preprocessor_config.json', 'processor_config.json',
             'params.json', 'model.safetensors.index.json']
    for name in small:
        try:
            hf_hub_download(repo, name, local_dir=out)
        except Exception:
            pass                                   # 仓库不一定每个文件都有
    index = os.path.join(out, 'model.safetensors.index.json')
    if os.path.exists(index):                      # 分片仓库：只取音频塔所在分片
        wm = json.load(open(index))['weight_map']
        need = sorted({v for k, v in wm.items() if k.startswith(('audio_tower.', 'multi_modal_projector.'))})
        if not need:
            raise SystemExit('index 里没有 audio_tower.* —— 仓库结构与预期不符')
    else:
        # 单文件仓库（Voxtral-Mini-4B-Realtime-2602 即是）：只下 HF 格式的 model.safetensors（8.25GiB）；
        # consolidated.safetensors 是 mistral 原生格式的同一份权重，键名不同，导出不读
        need = ['model.safetensors']
    print(f'[voxtral] 下载 {need}', flush=True)
    for shard in need:
        hf_hub_download(repo, shard, local_dir=out)
else:
    snapshot_download(repo, local_dir=out, allow_patterns=allow or None,
                      ignore_patterns=['*.pth', '*.bin', '*.gguf', 'original/*'])
PY
    touch "${dir}/.fetched"
}

need_4b=false; need_06=false
for v in ${VARIANTS}; do
    case "${v}" in
        streaming_si) need_06=true ;;
        streaming_si_4b) need_4b=true ;;
        *) die "未知变体 ${v}（streaming_si | streaming_si_4b）" ;;
    esac
done
QWEN3_DIR="${PRE}/$(basename "${THINKER_REPO}")"
QWEN35_DIR="${PRE}/$(basename "${THINKER_4B_REPO}")"
VOX_DIR="${PRE}/$(basename "${VOXTRAL_REPO}")"
fetch "${THINKER_REPO}" "${QWEN3_DIR}"          # 0.6B 的 thinker，也是 4B 变体 talker 的初始化来源
fetch "${VOXTRAL_REPO}" "${VOX_DIR}" __voxtral__
${need_4b} && fetch "${THINKER_4B_REPO}" "${QWEN35_DIR}"

COSY_ARGS=()
if [[ -n "${COSYVOICE_DIR:-}" ]]; then
    COSY_ARGS=(--cosyvoice_dir "${COSYVOICE_DIR}")
elif [[ -n "${COSYVOICE_REPO_ID}" ]]; then
    COSYVOICE_DIR="${PRE}/$(basename "${COSYVOICE_REPO_ID}")"
    fetch "${COSYVOICE_REPO_ID}" "${COSYVOICE_DIR}" 'flow.pt' 'hift.pt' 'campplus.onnx' '*.yaml'
    COSY_ARGS=(--cosyvoice_dir "${COSYVOICE_DIR}")
else
    log "未打包声码器（COSYVOICE_REPO_ID 为空）：推理时需另用 --cosyvoice_dir 指定"
fi

# ---------------------------------------------------------------------------
# 2. 导出（export_streaming_si.py 内部会回读逐张量比对；这里再做一次独立加载校验）
# ---------------------------------------------------------------------------
export_one() {
    local variant="$1" dst="${OUT}/base_${1}" args
    if [[ -f "${dst}/.exported" ]]; then log "已导出，跳过: ${dst}"; return; fi
    rm -rf "${dst}"
    case "${variant}" in
        streaming_si)    args=(--thinker_dir "${QWEN3_DIR}") ;;
        streaming_si_4b) args=(--variant streaming_si_4b --thinker_dir "${QWEN35_DIR}" --talker_init_dir "${QWEN3_DIR}") ;;
    esac
    log "导出 ${variant} -> ${dst}"
    "${PYTHON}" "${REPO_ROOT}/custom/streaming_si/scripts/export_streaming_si.py" "${args[@]}" \
        --voxtral_dir "${VOX_DIR}" ${COSY_ARGS[@]+"${COSY_ARGS[@]}"} --dtype "${DTYPE}" --out_dir "${dst}"
    "${PYTHON}" - "${dst}" "${variant}" <<'PY'
import json, os, sys
sys.path.insert(0, os.path.join(os.environ.get('REPO_ROOT', '.'), 'custom'))
import torch
from transformers import AutoTokenizer
from streaming_si.model.modeling_streaming_si import StreamingSIForConditionalGeneration
from streaming_si.model.variants import variant_of
d, want = sys.argv[1:]
m = StreamingSIForConditionalGeneration.from_pretrained(d, dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(d)
assert variant_of(m.config) == want, (variant_of(m.config), want)
assert tok.encode(m.config.audio_special_token, add_special_tokens=False) == list(m.config.audio_ids)
assert m.audio_encoder is not None and os.path.isfile(os.path.join(d, 'voxtral_processor_config.json'))
n = lambda mod: sum(p.numel() for p in mod.parameters()) / 1e6
size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs) / 2**30
info = dict(variant=want, thinker_M=round(n(m.thinker_model), 1), talker_M=round(n(m.talker), 1),
            audio_encoder_M=round(n(m.audio_encoder), 1), total_M=round(n(m), 1), size_GiB=round(size, 2),
            speech_decoder=os.path.isdir(os.path.join(d, 'speech_decoder')), audio_ids=m.config.audio_ids)
print('[verify]', json.dumps(info, ensure_ascii=False), flush=True)
json.dump(info, open(os.path.join(d, 'streaming_si_verify.json'), 'w'), indent=2)
PY
    touch "${dst}/.exported"
}
export REPO_ROOT
for v in ${VARIANTS}; do export_one "${v}"; done

# ---------------------------------------------------------------------------
# 3. 上传（可选）
# ---------------------------------------------------------------------------
if [[ -n "${S3_DEST:-}" ]]; then
    command -v aws >/dev/null || die "上传需要 aws CLI"
    for v in ${VARIANTS}; do
        src="${OUT}/base_${v}" dst="${S3_DEST%/}/base_${v}/"
        log "上传 ${src} -> ${dst}"
        aws s3 sync "${src}/" "${dst}" --region "${AWS_REGION}" --exclude '.exported' --only-show-errors
        n_local=$(find "${src}" -type f ! -name '.exported' | wc -l | tr -d ' ')
        n_s3=$(aws s3 ls "${dst}" --recursive --region "${AWS_REGION}" | wc -l | tr -d ' ')
        [[ "${n_s3}" -ge "${n_local}" ]] || die "上传不完整：本地 ${n_local} 个文件，S3 ${n_s3} 个"
        log "OK ${dst}（${n_s3} 个文件）"
    done
    [[ "${KEEP_DOWNLOADS:-0}" == 1 ]] || { log "清理下载的基座 ${PRE}"; rm -rf "${PRE}"; }
fi

log "完成：$(for v in ${VARIANTS}; do printf '%s ' "${OUT}/base_${v}"; done)"
