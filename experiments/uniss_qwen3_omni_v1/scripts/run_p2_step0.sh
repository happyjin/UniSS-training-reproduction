#!/usr/bin/env bash
# P2 step zero on CVSS-T: the official UniSS speaks Qwen3-Omni's translation
# (P1's, word for word) in the source speaker's voice, one sample a sentence,
# then the same scoring as every other system here.
#
#   generate  vllm_omni_030 venv, one shard a card  (p2_uniss_generate.py)
#   decode    uniss-eval, BiCodec, one shard a card (p2_decode.py)
#   score     run_stage1_metrics.sh + Text-BLEU + SIM-O, one direction at a time
#
# Sampling is UniSS's own (temperature 0.7, top-p 0.8) with the repetition
# penalty raised from 1.1 to 1.3, and the stop guard on: at 1.1 a BiCodec
# stream loops on about 0.8% of sentences and never ends; on 384 diagnostic
# samples 1.3 removed every one without lowering fidelity.
#
#   OUT=eval_outputs/uniss_qwen3_omni_v1/p2_step0_full \
#     bash experiments/uniss_qwen3_omni_v1/scripts/run_p2_step0.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO_ROOT}"
refuse_if_present() {
  if [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; then echo "拒绝覆盖已有输出: $1" >&2; exit 1; fi
}
export PYTHONPATH=. PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/opt/dlami/nvme/jasonleeeli/cache/huggingface}"
PYV="${PYV:-/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python}"
PYE="${PYE:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python}"
T="${T:-${REPO_ROOT}/data/cvss_t/tokenized/cvss_t_zh_en_v1}"
TRANSLATIONS="${TRANSLATIONS:-${REPO_ROOT}/eval_outputs/uniss_qwen3_omni_v1/p1_native_talker_full/p1a}"
OUT="$(realpath -m "${OUT:?}")"; EXPECTED="${EXPECTED:-4897}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; read -r -a G <<<"${GPUS}"; NG=${#G[@]}
REP="${REP:-1.3}"; TEMP="${TEMP:-0.7}"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
PAIRS=(--pair "cmn_eng:${T}/cvss_t_zh_en_test.parquet" --pair "eng_cmn:${T}/cvss_t_en_zh_test.parquet")

refuse_if_present "${OUT}"; mkdir -p "${OUT}/logs"
log "生成（${NG} 卡）"
for ((k = 0; k < NG; k++)); do
  CUDA_VISIBLE_DEVICES="${G[$k]}" VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 \
    "${PYV}" -m experiments.uniss_qwen3_omni_v1.evaluation.p2_uniss_generate "${PAIRS[@]}" \
    --translations "${TRANSLATIONS}" --output "${OUT}/codes" --shard "$k" --num-shards "${NG}" \
    --guard --temperature "${TEMP}" --repetition-penalty "${REP}" > "${OUT}/logs/gen_$k.log" 2>&1 &
done
wait
n=$(cat "${OUT}"/codes/codes_*.jsonl 2>/dev/null | wc -l)
[ "$n" = "$((2 * EXPECTED))" ] || { log "P2_FAILED: 生成 $n 条，应为 $((2 * EXPECTED))"; exit 1; }

log "解码（${NG} 卡）"
for ((k = 0; k < NG; k++)); do
  CUDA_VISIBLE_DEVICES="${G[$k]}" "${PYE}" -m experiments.uniss_qwen3_omni_v1.evaluation.p2_decode \
    --codes "${OUT}/codes" --parquet "cmn_eng:${T}/cvss_t_zh_en_test.parquet" --parquet "eng_cmn:${T}/cvss_t_en_zh_test.parquet" \
    --output "${OUT}" --shard "$k" --num-shards "${NG}" > "${OUT}/logs/dec_$k.log" 2>&1 &
done
wait
"${PYE}" -m experiments.uniss_qwen3_omni_v1.evaluation.p2_decode --codes "${OUT}/codes" \
  --parquet "cmn_eng:${T}/cvss_t_zh_en_test.parquet" --parquet "eng_cmn:${T}/cvss_t_en_zh_test.parquet" \
  --output "${OUT}" --merge | tee -a "${OUT}/logs/merge.log"
for d in cmn_eng eng_cmn; do
  [ "$(wc -l < "${OUT}/$d/results.jsonl")" = "${EXPECTED}" ] || { log "P2_FAILED: $d 结果不完整"; exit 1; }
done

# A runaway utterance pads an A.PCP batch to its own length; 4 keeps it on a card.
export AUTOPCP_BATCH_SIZE="${AUTOPCP_BATCH_SIZE:-4}"
for pair in "cmn_eng cmn->eng" "eng_cmn eng->cmn"; do
  set -- $pair; D="${OUT}/$1"
  log "打分 $2"
  OMP_NUM_THREADS=8 ALLOW_GENERATED_FAILURES=1 EXPECTED_PAIRS="${EXPECTED}" MODES=tts \
    bash experiments/uniss_omni_s2st_v1/scripts/run_stage1_metrics.sh "$D" "$2" > "$D/metrics.log" 2>&1 \
    || { log "P2_FAILED: $2 指标失败，见 $D/metrics.log"; exit 1; }
  "${PYE}" -m evaluation.text_metrics --input "$D/results.jsonl" --output "$D/metrics/text_bleu.json" \
    --hypothesis-field translated_text --reference-field translation_ref > /dev/null 2>&1
  CUDA_VISIBLE_DEVICES="${G[0]}" "${PYE}" -m experiments.uniss_omni_s2st_v1.evaluation.sim_o \
    --input "$D/results.jsonl" --output "$D/metrics/sim_o.json" \
    --model /opt/dlami/nvme/jasonleeeli/evaluation_models/wavlm-base-plus-sv --device cuda:0 > "$D/sim_o.log" 2>&1 \
    || { log "P2_FAILED: $2 SIM-O 失败"; exit 1; }
done
log "P2_DONE ${OUT}"
