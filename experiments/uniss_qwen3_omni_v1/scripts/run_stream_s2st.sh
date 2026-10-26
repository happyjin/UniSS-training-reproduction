#!/usr/bin/env bash
# Streaming (simultaneous) speech-to-speech translation of one recording with the current best system:
# Qwen3-Omni Thinker + ABC adapter (amortized Bayes-optimal commitment) + SDR (speech-queue risk), UniSS Talker T1c,
# Bayesian voice memory with a 30 s freeze. Causal: every read sees only the audio heard so far; outputs are laid on the
# recording's own clock. See docs/deploy/stream_s2st_bayes_deploy.zh-CN.md.
#
#   AUDIO=talk.wav DIRECTION=zh2en OUT=outputs/my_talk bash experiments/uniss_qwen3_omni_v1/scripts/run_stream_s2st.sh
#   DRY_RUN=1 ... prints the commands and starts nothing.
#
# Defaults are the m2 tier (latency about 1.7-1.8 s): en2zh WINDOW=0.5 BIAS=1, zh2en WINDOW=1 BIAS=0.
# Knobs: WINDOW (read window, s), BIAS (wait-marker bias b0; larger waits more),
#        VOICE=bayes|ema|plain, GPU_THINKER / GPU_TALKER (card ids), LIMIT_S (only the first seconds, 0 = all).
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO_ROOT}"
AUDIO="$(realpath "${AUDIO:?AUDIO=<16 kHz or any wav>}")"; DIRECTION="${DIRECTION:?DIRECTION=zh2en|en2zh}"
OUT="$(realpath -m "${OUT:?OUT=<new output dir>}")"
VOICE="${VOICE:-bayes}"; LIMIT_S="${LIMIT_S:-0}"
GPU_THINKER="${GPU_THINKER:-0}"; GPU_TALKER="${GPU_TALKER:-1}"
PYV="${PYV:-/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python}"         # inference (vLLM) venv
PYE="${PYE:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python}"       # segmentation / stitching (conda uniss-eval)
THINKER="${THINKER:-eval_outputs/uniss_qwen3_omni_v1/thinker_op_v3_20261023T213631Z/thinker_merged_i200}"
TALKER="${TALKER:-eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss}"
case "${DIRECTION}" in
  zh2en) RATE=0.339; SHIFT=2; CATCHUP=1.3; WINDOW="${WINDOW:-1}"; BIAS="${BIAS:-0}" ;;
  en2zh) RATE=0.191; SHIFT=0; CATCHUP=1.0; WINDOW="${WINDOW:-0.5}"; BIAS="${BIAS:-1}" ;;
  *) echo "DIRECTION must be zh2en or en2zh" >&2; exit 1 ;;
esac
case "${VOICE}" in
  bayes) VOICE_ARGS="--voice-bayes --voice-freeze-sec 30" ;;
  ema)   VOICE_ARGS="--voice-ema 0.2" ;;
  plain) VOICE_ARGS="" ;;
  *) echo "VOICE must be bayes, ema or plain" >&2; exit 1 ;;
esac
VENV=(env VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=16)
SUF="_swop1b${BIAS}q${RATE}"
SRC16="${OUT}/$(basename "${AUDIO%.*}").wav"   # 16 kHz mono copy of the input; segmentation and the voice memory read it
BLOCKS="${OUT}/thinker/blocks_${DIRECTION}_w${WINDOW}${SUF}.jsonl"

STEP1=("${PYE}" experiments/uniss_qwen3_omni_v1/evaluation/si_longform_demo.py segment --audio "${SRC16}" --direction "${DIRECTION}"
       --out "${OUT}/pieces" --limit-s "${LIMIT_S}")
STEP3=("${VENV[@]}" CUDA_VISIBLE_DEVICES="${GPU_THINKER}" "${PYV}" -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream
       --direction "${DIRECTION}" --items-file "${OUT}/pieces/items_talk.jsonl" --windows "${WINDOW}" --policy selfwait
       --thinker "${THINKER}" --thinker-tag op1 "--wait-biases=${BIAS}" --sw-queue-rate "${RATE}" --out "${OUT}/thinker")
STEP4=("${VENV[@]}" CUDA_VISIBLE_DEVICES="${GPU_TALKER}" "${PYV}" -m experiments.uniss_qwen3_omni_v1.evaluation.si_infer
       --model "${TALKER}" --direction "${DIRECTION}" --window "${WINDOW}" --output "${OUT}/talker" --text-policy file
       --blocks-file "${BLOCKS}" --items-file "${OUT}/pieces/items_talk.jsonl" --speed-policy block --history-ban 8
       --speed-shift "${SHIFT}" --max-model-len 16384 ${VOICE_ARGS})
STEP5=("${PYE}" experiments/uniss_qwen3_omni_v1/evaluation/si_longform_demo.py stitch --items "${OUT}/pieces/items.jsonl"
       --infer "${OUT}/talker" --blocks "${BLOCKS}" --out "${OUT}/result" --catchup-max "${CATCHUP}")

if [ -n "${DRY_RUN:-}" ]; then
  for s in STEP1 STEP3 STEP4 STEP5; do declare -n a=$s; printf '%q ' "${a[@]}"; echo; done
  echo "items_talk.jsonl = items.jsonl + talk_wav (step 2); outputs: ${OUT}/result/{translation,stereo,mix}.wav, transcript.json"
  exit 0
fi
if [ -e "${OUT}" ]; then echo "拒绝覆盖已有输出: ${OUT}" >&2; exit 1; fi
mkdir -p "${OUT}"
log() { echo "[$(TZ=Europe/Prague date +%H:%M:%S)] $*"; }
log "0/5 转成 16 kHz 单声道"
"${PYE}" - "${AUDIO}" "${SRC16}" <<'PY' || { log "音频转换失败（需要 soundfile 能读的格式，如 wav/flac/ogg/mp3）"; exit 1; }
import sys, math
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
x, sr = sf.read(sys.argv[1], dtype="float32", always_2d=True)
x = x.mean(axis=1)
if sr != 16000:
    g = math.gcd(int(sr), 16000)
    x = resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)
sf.write(sys.argv[2], x, 16000)
print(f"{sys.argv[1]}: {sr} Hz -> 16000 Hz, {len(x) / 16000:.1f} s")
PY
log "1/5 按停顿切分"; "${STEP1[@]}" > "${OUT}/step1_segment.log" 2>&1 || { log "切分失败，见 ${OUT}/step1_segment.log"; exit 1; }
log "2/5 生成音色记忆所需的 items_talk.jsonl"
"${PYE}" - "${OUT}/pieces/items.jsonl" "${SRC16}" "${OUT}/pieces/items_talk.jsonl" <<'PY' || { log "items_talk 生成失败"; exit 1; }
import json, sys
src, audio, dst = sys.argv[1:]
with open(dst, "x", encoding="utf-8") as fh:
    for line in open(src, encoding="utf-8"):
        fh.write(json.dumps({**json.loads(line), "talk_wav": audio}, ensure_ascii=False) + "\n")
PY
log "3/5 Thinker 流式提交（GPU ${GPU_THINKER}）"; "${STEP3[@]}" > "${OUT}/step3_thinker.log" 2>&1 || { log "Thinker 失败，见 ${OUT}/step3_thinker.log"; exit 1; }
log "4/5 Talker 逐块发声（GPU ${GPU_TALKER}）"; "${STEP4[@]}" > "${OUT}/step4_talker.log" 2>&1 || { log "Talker 失败，见 ${OUT}/step4_talker.log"; exit 1; }
[ -f "${OUT}/talker/results.jsonl" ] || cat "${OUT}"/talker/part_*.jsonl > "${OUT}/talker/results.jsonl"
log "5/5 按原时间轴拼接"; "${STEP5[@]}" > "${OUT}/step5_stitch.log" 2>&1 || { log "拼接失败，见 ${OUT}/step5_stitch.log"; exit 1; }
log "完成：${OUT}/result/translation.wav（仅译文）、stereo.wav（左原声右译文）、transcript.json"
