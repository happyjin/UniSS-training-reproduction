#!/usr/bin/env bash
# One end-to-end probe: source audio in, translated speech out, scored.
#
# This is the measurement Phase E is judged on. Every TTS-mode number --
# Phase D's 70.16 among them -- hands the model the reference text and so
# scores the Talker with the translation error removed. Here the model
# hears, translates and speaks, with the hidden states going straight to
# the Talker and no text string between them, which is both the delivered
# form and the only one that can stream.
#
# Staggered at 90 seconds: a shard's load costs 24 to 32 cores for two to
# four minutes, and eight at once on 192 cores had five crawling while
# three ran.
set -uo pipefail

# Never delete. A non-empty output directory is the user's to clear,
# not this script's -- an rm -rf here once destroyed a finished run's
# waveforms because two runs chose the same path.
refuse_if_present() {
  if [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; then
    echo "拒绝覆盖已有输出: $1" >&2
    echo "换一个输出路径，或由你手动清理后重跑。" >&2
    exit 1
  fi
}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO_ROOT}"
export PYTHONPATH=. PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/opt/dlami/nvme/jasonleeeli/cache/huggingface}"
PY3="${PY3:-/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python}"
PYE="${PYE:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python}"
T="${T:-${REPO_ROOT}/data/cvss_t/tokenized/cvss_t_zh_en_v1}"
TALKER="${TALKER:?}"; CKPT="${CKPT:?}"; OUT="${OUT:?}"
LIMIT="${LIMIT:-64}"; GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; STAGGER="${STAGGER:-90}"

refuse_if_present "${OUT}"
mkdir -p "${OUT}"
for pair in "cmn_eng cvss_t_zh_en_test" "eng_cmn cvss_t_en_zh_test"; do
  set -- $pair; NAME=$1; PQ=$2
  D="${OUT}/${NAME}"; mkdir -p "$D/logs"
  shards=0; for _ in $GPUS; do shards=$((shards+1)); done
  i=0
  for g in $GPUS; do
    [ $i -gt 0 ] && sleep "${STAGGER}"
    CUDA_VISIBLE_DEVICES=$g "${PY3}" -m experiments.uniss_qwen3_omni_v1.evaluation.phase_d_e2e \
      --talker-checkpoint "${TALKER}" --phase-d-checkpoint "${CKPT}" \
      --dev-parquet "${T}/${PQ}.parquet" --limit "${LIMIT}" \
      ${SINGLE_PASS:+--single-pass} \
      ${TASK_FORMAT:+--task-format "${TASK_FORMAT}"} ${ORACLE:+--oracle-reply} \
      --shard $i --num-shards $shards --output "$D" > "$D/logs/g_$i.log" 2>&1 &
    i=$((i+1))
  done
  wait
  cat "$D"/e2e_*.jsonl > "$D/results.jsonl"
done
cat "${OUT}"/*/results.jsonl > "${OUT}/results.jsonl"
n=$(wc -l < "${OUT}/results.jsonl")
echo "生成 ${n} 条"
CUDA_VISIBLE_DEVICES=0 "${PYE}" -m evaluation.asr_transcribe \
  --input "${OUT}/results.jsonl" --output "${OUT}/asr.jsonl" \
  --whisper-model "${WHISPER_MODEL:-openai/whisper-large-v3}" \
  --paraformer-model "${PARAFORMER_MODEL:-iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch}" \
  --device cuda:0 --batch-size 16 > "${OUT}/asr.log" 2>&1
"${PYE}" -m evaluation.text_metrics --input "${OUT}/asr.jsonl" \
  --output "${OUT}/asr_bleu.json" --hypothesis-field asr_text \
  --reference-field translation_ref > "${OUT}/bleu.log" 2>&1
"${PYE}" -m evaluation.text_metrics --input "${OUT}/results.jsonl" \
  --output "${OUT}/text_bleu.json" --hypothesis-field translated_text \
  --reference-field translation_ref > /dev/null 2>&1
"${PYE}" -c "
import json
for f,label in (('${OUT}/asr_bleu.json','ASR-BLEU'),('${OUT}/text_bleu.json','Text-BLEU')):
    try:
        d=json.load(open(f))
        for k,v in sorted(d['groups'].items()):
            print(f'  {label} {k}: {v[\"score\"]:.2f} (n={v[\"sample_count\"]})')
    except Exception as e: print(f'  {label}: {e}')"
echo E2E_PROBE_DONE
