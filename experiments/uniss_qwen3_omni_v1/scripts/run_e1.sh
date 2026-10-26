#!/usr/bin/env bash
# E1 on CVSS-T: Qwen3-Omni's Thinker and UniSS as one inference chain
# (e1_pipeline.py), one shard a card, then the same scoring as every other
# system here, one direction at a time.
#
#   OUT=eval_outputs/uniss_qwen3_omni_v1/e1_full \
#     bash experiments/uniss_qwen3_omni_v1/scripts/run_e1.sh
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
OUT="$(realpath -m "${OUT:?}")"; EXPECTED="${EXPECTED:-4897}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; read -r -a G <<<"${GPUS}"; NG=${#G[@]}
SUBSETS="${SUBSETS:-}"   # directory holding subset_<dir>.jsonl; unset runs the full test sets
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
ids() { [ -n "${SUBSETS}" ] && echo ":${SUBSETS}/subset_$1.jsonl"; }

refuse_if_present "${OUT}"; mkdir -p "${OUT}/logs"
log "E1 生成（${NG} 卡）"
for ((k = 0; k < NG; k++)); do
  CUDA_VISIBLE_DEVICES="${G[$k]}" VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 \
    "${PYV}" -m experiments.uniss_qwen3_omni_v1.evaluation.e1_pipeline \
    --pair "cmn_eng:${T}/cvss_t_zh_en_test.parquet$(ids cmn_eng)" \
    --pair "eng_cmn:${T}/cvss_t_en_zh_test.parquet$(ids eng_cmn)" \
    --output "${OUT}" --shard "$k" --num-shards "${NG}" > "${OUT}/logs/e1_$k.log" 2>&1 &
done
wait
grep -h "timing" "${OUT}"/logs/e1_*.log | tee "${OUT}/logs/timing.txt"

"${PYE}" - "${OUT}" "${EXPECTED}" <<'PY' || { log "E1_FAILED: 结果不完整"; exit 1; }
import glob, json, sys
out, n = sys.argv[1], int(sys.argv[2]); bad = []
for d in ("cmn_eng", "eng_cmn"):
    recs = sorted((json.loads(l) for p in glob.glob(f"{out}/{d}/part_*.jsonl") for l in open(p)), key=lambda r: r["index"])
    with open(f"{out}/{d}/results.jsonl", "w") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    print(f"{d}: {len(recs)} 句，无音频 {sum(not r['audio_path'] for r in recs)}，未正常结束 {sum(not r['ended'] for r in recs)}")
    if len(recs) != n:
        bad.append(d)
sys.exit(1 if bad else 0)
PY

export AUTOPCP_BATCH_SIZE="${AUTOPCP_BATCH_SIZE:-4}"
for pair in "cmn_eng cmn->eng" "eng_cmn eng->cmn"; do
  set -- $pair; D="${OUT}/$1"
  log "打分 $2"
  OMP_NUM_THREADS=8 ALLOW_GENERATED_FAILURES=1 EXPECTED_PAIRS="${EXPECTED}" MODES=tts \
    bash experiments/uniss_omni_s2st_v1/scripts/run_stage1_metrics.sh "$D" "$2" > "$D/metrics.log" 2>&1 \
    || { log "E1_FAILED: $2 指标失败，见 $D/metrics.log"; exit 1; }
  "${PYE}" -m evaluation.text_metrics --input "$D/results.jsonl" --output "$D/metrics/text_bleu.json" \
    --hypothesis-field translated_text --reference-field translation_ref > /dev/null 2>&1
  CUDA_VISIBLE_DEVICES="${G[0]}" "${PYE}" -m experiments.uniss_omni_s2st_v1.evaluation.sim_o \
    --input "$D/results.jsonl" --output "$D/metrics/sim_o.json" \
    --model /opt/dlami/nvme/jasonleeeli/evaluation_models/wavlm-base-plus-sv --device cuda:0 > "$D/sim_o.log" 2>&1 \
    || { log "E1_FAILED: $2 SIM-O 失败"; exit 1; }
done
log "E1_DONE ${OUT}"
