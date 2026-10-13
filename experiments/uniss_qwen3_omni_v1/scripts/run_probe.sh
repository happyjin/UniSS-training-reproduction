#!/usr/bin/env bash
# One acoustic probe: generate, vocode, transcribe, score.
#
# ASR and BLEU go through the project's own modules rather than a local
# shortcut, so a probe number is comparable to every other number in the
# project. The first scoring of stage 2's ablation bypassed that and came
# back at half the real value.
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
TALKER="${TALKER:?}"
OUT="${OUT:?}"
LIMIT="${LIMIT:-128}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
EXTRA=()
[ -n "${PHASE_D_CKPT:-}" ] && EXTRA+=(--phase-d-checkpoint "${PHASE_D_CKPT}")
EXTRA+=(--thinker-from "${THINKER_FROM:-qwen3}")

refuse_if_present "${OUT}"
mkdir -p "${OUT}/logs"
shards=0; for _ in $GPUS; do shards=$((shards+1)); done
i=0
for g in $GPUS; do
  # Staggered, because eight processes reading the same 66 GB of weights
  # off one NVMe is 528 GB of I/O and they crawl: measured, seven shards
  # sat at 16% of the load while the eighth had already finished its
  # utterances. The first read warms the page cache and the rest come off
  # it.
  [ $i -gt 0 ] && sleep "${STAGGER:-20}"
  CUDA_VISIBLE_DEVICES=$g "${PY3}" -m experiments.uniss_qwen3_omni_v1.evaluation.acoustic_probe \
    --talker-checkpoint "${TALKER}" --dev-parquet \
    "${T}/cvss_t_zh_en_test.parquet" "${T}/cvss_t_en_zh_test.parquet" \
    --limit "${LIMIT}" --shard $i --num-shards $shards --output "${OUT}" \
    "${EXTRA[@]}" > "${OUT}/logs/g_$i.log" 2>&1 &
  i=$((i+1))
done
wait
cat "${OUT}"/probe_*.jsonl > "${OUT}/results.jsonl"
n=$(wc -l < "${OUT}/results.jsonl")
voiced=$("${PYE}" -c "
import json,sys
rows=[json.loads(l) for l in open('${OUT}/results.jsonl',encoding='utf-8')]
print(sum(1 for r in rows if r['audio_path']))")
echo "生成 ${n} 条，其中 ${voiced} 条有音频"

CUDA_VISIBLE_DEVICES=0 "${PYE}" -m evaluation.asr_transcribe \
  --input "${OUT}/results.jsonl" --output "${OUT}/asr.jsonl" \
  --whisper-model "${WHISPER_MODEL:-openai/whisper-large-v3}" \
  --paraformer-model "${PARAFORMER_MODEL:-iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch}" \
  --device cuda:0 --batch-size 16 > "${OUT}/logs/asr.log" 2>&1
"${PYE}" -m evaluation.text_metrics --input "${OUT}/asr.jsonl" \
  --output "${OUT}/asr_bleu.json" --hypothesis-field asr_text \
  --reference-field translation_ref > "${OUT}/logs/bleu.log" 2>&1
"${PYE}" -c "
import json
d=json.load(open('${OUT}/asr_bleu.json'))
for k,v in sorted(d['groups'].items()):
    print(f'  {k}: ASR-BLEU {v[\"score\"]:.2f}  (n={v[\"sample_count\"]})')"
echo PROBE_DONE
