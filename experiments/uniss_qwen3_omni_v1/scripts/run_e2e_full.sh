#!/usr/bin/env bash
# One end-to-end checkpoint on the full CVSS-T test sets, 4,897 pairs a
# direction, scored in Table 1's metrics.
#
# The generator and its arguments are run_e2e_probe.sh's, and the scoring
# is run_phase_d_table1.sh's -- run_stage1_metrics.sh for Speech-BLEU, SLC,
# UTMOS and A.PCP, then Text-BLEU and SIM-O -- so the numbers sit beside
# both. What changes is the layout, because at full size the probe's
# layout is mostly waiting:
#
#   Both directions at once, one process per direction on every GPU. The
#   generator decodes one utterance at a time, where the MoE thinker is
#   bound by kernel launches rather than arithmetic, so two processes on a
#   card overlap. Each holds about 66 GB against 141.
#
#   SHARDS_PER_GPU shards per direction per GPU, run back to back. The
#   generator writes its jsonl only at the end, so a shard that dies loses
#   everything it did; smaller shards bound that loss.
#
#   A shard with no output is rerun alone, one process per card, and the
#   run refuses to score if one is still missing -- a partial test set
#   scores as if it were the whole one.
set -uo pipefail

# Never delete. A non-empty output directory is the user's to clear.
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY3="${PY3:-/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python}"
PYE="${PYE:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python}"
T="${T:-${REPO_ROOT}/data/cvss_t/tokenized/cvss_t_zh_en_v1}"
TALKER="${TALKER:?}"; CKPT="${CKPT:?}"; OUT="${OUT:?}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; SHARDS_PER_GPU="${SHARDS_PER_GPU:-2}"
STAGGER="${STAGGER:-40}"; EXPECTED="${EXPECTED:-4897}"
read -r -a G <<<"${GPUS}"; NG=${#G[@]}; N=$((NG * SHARDS_PER_GPU))
DIRS=("cmn_eng cvss_t_zh_en_test cmn->eng" "eng_cmn cvss_t_en_zh_test eng->cmn")
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

refuse_if_present "${OUT}"
for d in "${DIRS[@]}"; do set -- $d; mkdir -p "${OUT}/$1/logs"; done

# The one place the generator is invoked, identical to the probe's call.
gen() {  # name parquet shard gpu
  CUDA_VISIBLE_DEVICES=$4 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
    "${PY3}" -m experiments.uniss_qwen3_omni_v1.evaluation.phase_d_e2e \
    --talker-checkpoint "${TALKER}" --phase-d-checkpoint "${CKPT}" \
    --dev-parquet "${T}/$2.parquet" --limit 0 \
    ${SINGLE_PASS:+--single-pass} \
    --shard "$3" --num-shards "${N}" --output "${OUT}/$1" \
    >> "${OUT}/$1/logs/s_$3.log" 2>&1
}
worker() {  # name parquet gpu-slot -- this card's shards for one direction
  local j
  for ((j = 0; j < SHARDS_PER_GPU; j++)); do gen "$1" "$2" $(($3 + j * NG)) "${G[$3]}"; done
}

log "生成：${#DIRS[@]} 个方向 × ${N} 片，每卡 2 个进程，CKPT=${CKPT}"
for d in "${DIRS[@]}"; do
  set -- $d
  for ((k = 0; k < NG; k++)); do
    worker "$1" "$2" "$k" &
    sleep "${STAGGER}"
  done
done
wait
log "第一轮生成结束"

missing() {  # name -> shard indices with no output
  local s
  for ((s = 0; s < N; s++)); do [ -s "${OUT}/$1/e2e_${s}.jsonl" ] || echo "$s"; done
}
for d in "${DIRS[@]}"; do
  set -- $d
  m=$(missing "$1" | tr '\n' ' ')
  [ -z "${m// /}" ] && continue
  log "补跑 $1 缺失分片: ${m}"
  k=0
  for s in ${m}; do
    gen "$1" "$2" "$s" "${G[$((k % NG))]}" &
    k=$((k + 1))
    [ $((k % NG)) -eq 0 ] && wait
  done
  wait
done

for d in "${DIRS[@]}"; do
  set -- $d
  m=$(missing "$1" | tr '\n' ' ')
  if [ -n "${m// /}" ]; then
    log "FULL_EVAL_FAILED: $1 补跑后仍缺分片 ${m}，不打分"; exit 1
  fi
  cat "${OUT}/$1"/e2e_*.jsonl > "${OUT}/$1/results.jsonl"
  n=$(wc -l < "${OUT}/$1/results.jsonl")
  if [ "${n}" -ne "${EXPECTED}" ]; then
    log "FULL_EVAL_FAILED: $1 有 ${n} 条，应为 ${EXPECTED}，不打分"; exit 1
  fi
  log "$1: ${n} 条"
done

for d in "${DIRS[@]}"; do
  set -- $d; D="${OUT}/$1"
  log "打分 $3"
  EXPECTED_PAIRS="${EXPECTED}" MODES=tts \
    bash experiments/uniss_omni_s2st_v1/scripts/run_stage1_metrics.sh "$D" "$3" \
    > "$D/logs/metrics.log" 2>&1 || { log "FULL_EVAL_FAILED: $3 指标脚本失败，见 $D/logs/metrics.log"; exit 1; }
  "${PYE}" -m evaluation.text_metrics --input "$D/results.jsonl" \
    --output "$D/metrics/text_bleu.json" --hypothesis-field translated_text \
    --reference-field translation_ref > "$D/logs/text_bleu.log" 2>&1
  CUDA_VISIBLE_DEVICES="${G[0]}" "${PYE}" -m experiments.uniss_omni_s2st_v1.evaluation.sim_o \
    --input "$D/results.jsonl" --output "$D/metrics/sim_o.json" \
    --model /opt/dlami/nvme/jasonleeeli/evaluation_models/wavlm-base-plus-sv \
    --device cuda:0 > "$D/logs/sim_o.log" 2>&1
done
log "FULL_EVAL_DONE ${OUT}"
