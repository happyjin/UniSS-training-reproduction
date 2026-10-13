#!/usr/bin/env bash
# Phase D end to end on the full CVSS-T test, in Table 1's four metrics.
#
# Staggered at 90 seconds rather than 12: loading a shard costs 24 to 32
# cores for two to four minutes, and eight of those against 192 cores had
# five processes fighting while three ran. The weights are in page cache by
# then, so the stagger buys CPU, not I/O.
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
T="${T:-${REPO_ROOT}/data/cvss_t/tokenized/cvss_t_zh_en_v1}"
TALKER="${TALKER:?}"; PHASE_D_CKPT="${PHASE_D_CKPT:?}"; OUT="${OUT:?}"
LIMIT="${LIMIT:-0}"; GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; STAGGER="${STAGGER:-90}"

for pair in "cmn_eng cvss_t_zh_en_test cmn->eng" "eng_cmn cvss_t_en_zh_test eng->cmn"; do
  set -- $pair; NAME=$1; PQ=$2; LABEL=$3
  D="${OUT}/${NAME}"; refuse_if_present "$D"; mkdir -p "$D/logs"
  shards=0; for _ in $GPUS; do shards=$((shards+1)); done
  i=0
  for g in $GPUS; do
    [ $i -gt 0 ] && sleep "${STAGGER}"
    CUDA_VISIBLE_DEVICES=$g "${PY3}" -m experiments.uniss_qwen3_omni_v1.evaluation.phase_d_e2e \
      --talker-checkpoint "${TALKER}" --phase-d-checkpoint "${PHASE_D_CKPT}" \
      --dev-parquet "${T}/${PQ}.parquet" --limit "${LIMIT}" \
      --shard $i --num-shards $shards --output "$D" > "$D/logs/g_$i.log" 2>&1 &
    i=$((i+1))
  done
  wait
  cat "$D"/e2e_*.jsonl > "$D/results.jsonl"
  N=$(wc -l < "$D/results.jsonl"); echo "${LABEL}: ${N} 条"
  # The repo harness: multi-GPU ASR shards, A.PCP, UTMOS, and the
  # normalised scorer every other number in this project went through.
  EXPECTED_PAIRS=$N MODES=tts \
    bash experiments/uniss_omni_s2st_v1/scripts/run_stage1_metrics.sh "$D" "$LABEL" 2>&1 | tail -3
  PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
  $PYE -m evaluation.text_metrics --input "$D/results.jsonl" \
    --output "$D/metrics/text_bleu.json" --hypothesis-field translated_text \
    --reference-field translation_ref > /dev/null 2>&1
  CUDA_VISIBLE_DEVICES=0 $PYE -m experiments.uniss_omni_s2st_v1.evaluation.sim_o \
    --input "$D/results.jsonl" --output "$D/metrics/sim_o.json" \
    --model /opt/dlami/nvme/jasonleeeli/evaluation_models/wavlm-base-plus-sv \
    --device cuda:0 2>&1 | tail -2
done
echo PHASE_D_TABLE1_DONE
