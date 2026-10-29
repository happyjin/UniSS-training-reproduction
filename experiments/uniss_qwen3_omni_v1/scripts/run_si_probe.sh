#!/usr/bin/env bash
# Probe a T1 checkpoint: export it, speak RealSI with oracle text block by
# block (evaluation/si_infer.py), score (evaluation/si_metrics.py), and with
# CVSS=1 also run the offline chain on the 500-sentence CVSS-T subsets, to see
# that offline speech has not been lost.
#
#   CKPT=checkpoints/uniss_qwen3_omni_v1/<run>/iter_0000100 OUT=eval_outputs/.../probe_iter100 \
#   GPUS="4 5 6 7" LIMIT=100 WINDOW=3 bash experiments/uniss_qwen3_omni_v1/scripts/run_si_probe.sh
#   SEED=<n> samples the talker with another seed (default: si_infer's), to see how much a score moves with sampling alone;
#   TALKER_T=<t> samples it at another temperature (default 0.7);
#   BEST_OF=<n> samples n continuations a block and keeps the talker's likeliest;
#   SPEED_SHIFT=<k> asks every block k speed steps faster (block speed policy);
#   DUMP_CODES=1 keeps each block's semantic codes in results.jsonl;
#   TALKER_GPU_MEM=<f> sizes each talker engine (si_infer --gpu-mem; 0.15 leaves room for two engines a card);
#   SELECT_K=<k> SCORER_URL=<url> keeps, of k samples a block, the one si_asr_scorer.py finds likeliest (Plan 2).
#   EXPORT_DIR=<an export with uniss/> instead of CKPT uses that model as is (R2 for step zero).
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO_ROOT}"
refuse_if_present() {
  if [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; then echo "拒绝覆盖已有输出: $1" >&2; exit 1; fi
}
export PYTHONPATH=. PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/opt/dlami/nvme/jasonleeeli/cache/huggingface}"
PYV="${PYV:-/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python}"
PYE="${PYE:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python}"
PYM="${PYM:-/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python}"
CKPT="${CKPT:-}"; EXPORT_DIR="${EXPORT_DIR:-}"
[ -n "${CKPT}${EXPORT_DIR}" ] || { echo "需要 CKPT 或 EXPORT_DIR" >&2; exit 1; }
OUT="$(realpath -m "${OUT:?}")"
GPUS="${GPUS:-4 5 6 7}"; read -r -a G <<<"${GPUS}"
LIMIT="${LIMIT:-100}"; WINDOW="${WINDOW:-3}"
log() { echo "[$(TZ=Europe/Prague date +%H:%M:%S)] $*"; }

refuse_if_present "${OUT}"; mkdir -p "${OUT}/logs"
if [ -n "${EXPORT_DIR}" ]; then
  EXPORT="$(realpath "${EXPORT_DIR}")"
else
  EXPORT="${OUT}/export"
  log "导出 ${CKPT}"
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=".:third_party/Megatron-LM" "${PYM}" \
    experiments/uniss_qwen3_omni_v1/training/export_e2.py --checkpoint-dir "${CKPT}" --output "${EXPORT}" \
    > "${OUT}/logs/export.log" 2>&1 || { log "SIP_FAILED: 导出失败"; exit 1; }
fi
log "RealSI 流式（窗 ${WINDOW}s，每方向 ${LIMIT:-全部} 句）"
k=0
for dir in ${DIRS:-zh2en en2zh}; do
  for s in 0 1; do
    CUDA_VISIBLE_DEVICES="${G[$((k % ${#G[@]}))]}" VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 \
      "${PYV}" -m experiments.uniss_qwen3_omni_v1.evaluation.si_infer --model "${EXPORT}/uniss" --direction "${dir}" \
      --window "${WINDOW}" --limit "${LIMIT}" --output "${OUT}/realsi_${dir}" --shard "${s}" --num-shards 2 \
      --text-policy "${TEXT_POLICY:-clause}" --min-words "${MIN_WORDS:-2}" \
      ${BLOCKS_DIR:+--blocks-file "${BLOCKS_DIR}/blocks_${dir}_w${WINDOW}${BLOCKS_SUFFIX:-}.jsonl"} \
      --speed-policy "${SPEED_POLICY:-fixed}" --history-ban "${HISTORY_BAN:-0}" ${SEED:+--seed "${SEED}"} ${MAX_MODEL_LEN:+--max-model-len "${MAX_MODEL_LEN}"} ${TALKER_T:+--temperature "${TALKER_T}"} ${BEST_OF:+--best-of "${BEST_OF}"} ${SPEED_SHIFT:+--speed-shift "${SPEED_SHIFT}"} ${DUMP_CODES:+--dump-codes} ${SELECT_K:+--select-k "${SELECT_K}"} ${SCORER_URL:+--scorer-url "${SCORER_URL}"} ${LOOKAHEAD:+--lookahead} ${VOICE_CTX:+--voice-context-ms "${VOICE_CTX}"} ${VOICE_EMA:+--voice-ema "${VOICE_EMA}"} ${VOICE_BAYES:+--voice-bayes} ${VOICE_LOCK:+--voice-lock "${VOICE_LOCK}"} ${VOICE_FREEZE:+--voice-freeze-sec "${VOICE_FREEZE}"} ${SELECT_VOICE:+--select-voice "${SELECT_VOICE}"} ${TALKER_GPU_MEM:+--gpu-mem "${TALKER_GPU_MEM}"} \
      > "${OUT}/logs/si_${dir}_${s}.log" 2>&1 &
    k=$((k + 1))
    # vLLM sizes its KV cache from the memory free when it profiles: two engines profiling on one card
    # at the same moment can leave the second none (system100 m2), so starts are staggered
    sleep "${STAGGER_S:-20}"
  done
done
wait
for dir in ${DIRS:-zh2en en2zh}; do
  [ "$(ls "${OUT}/realsi_${dir}"/part_*.jsonl 2>/dev/null | wc -l)" -eq 2 ] || { log "SIP_FAILED: ${dir} 推理不完整，见 logs"; exit 1; }
done
log "打分"
k=0
for dir in ${DIRS:-zh2en en2zh}; do
  "${PYE}" -m experiments.uniss_qwen3_omni_v1.evaluation.si_metrics --run "${OUT}/realsi_${dir}" --gpu "${G[$((k % ${#G[@]}))]}" \
    > "${OUT}/logs/metrics_${dir}.log" 2>&1 &
  k=$((k + 1))
done
wait
for dir in ${DIRS:-zh2en en2zh}; do
  [ -f "${OUT}/realsi_${dir}/metrics.json" ] || { log "SIP_FAILED: ${dir} 打分失败"; exit 1; }
  echo "${dir} $(tr -d '\n ' < "${OUT}/realsi_${dir}/metrics.json")"
done
if [ "${EXTRA:-0}" = 1 ]; then
  log "A.PCP / SIM-src / 对齐 LAAL"
  k=0
  for dir in ${DIRS:-zh2en en2zh}; do
    "${PYE}" -m experiments.uniss_qwen3_omni_v1.evaluation.si_extra_metrics --run "${OUT}/realsi_${dir}" --gpu "${G[$((k % ${#G[@]}))]}" \
      > "${OUT}/logs/extra_${dir}.log" 2>&1 &
    k=$((k + 1))
  done
  wait
  for dir in ${DIRS:-zh2en en2zh}; do
    echo "${dir} extra $(tr -d '\n ' < "${OUT}/realsi_${dir}/metrics_extra.json" 2>/dev/null || echo FAILED)"
  done
fi
if [ "${CVSS:-0}" = 1 ]; then
  log "CVSS-T 离线子集（每方向 500 句）"
  # EVAL_GPU_LIST: the metric shards stay on the probe's cards, off the training ones
  EXPORT_DIR="${EXPORT}" OUT="${OUT}/cvss_offline" GPUS="${GPUS}" EVAL_GPU_LIST="$(echo ${GPUS} | tr ' ' ',')" \
  SUBSETS="${REPO_ROOT}/eval_outputs/uniss_qwen3_omni_v1/phase_h_s2st100_20261015T074305Z/decode_sweep_iter74000" \
  LENGTH_POLICY=natural-clip LOOP_NGRAM=8 MIN_CODES_FRAC=0.3 SI_GUARD=1 \
    bash experiments/uniss_qwen3_omni_v1/scripts/run_e2_probe.sh > "${OUT}/logs/cvss_offline.log" 2>&1 \
    || { log "SIP_FAILED: CVSS-T 离线子集失败"; exit 1; }
fi
if [ "${REPORT:-1}" = 1 ]; then   # every finished run refreshes the comparison with published systems
  "${PYE}" -m experiments.uniss_qwen3_omni_v1.evaluation.si_report > "${OUT}/logs/report.log" 2>&1 || log "report failed (see logs/report.log)"
fi
log "SIP_DONE ${OUT}"
