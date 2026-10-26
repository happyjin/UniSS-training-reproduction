#!/usr/bin/env bash
# Score one E2 checkpoint: export it, run the trained chain (e2_infer.py)
# one shard a card, then the same scoring as every other system here.
# The Thinker is frozen, so its translations are taken from an E1 run
# (REUSE) rather than generated again.
#
#   CKPT=checkpoints/uniss_qwen3_omni_v1/<run>/iter_0001000 \
#   OUT=eval_outputs/uniss_qwen3_omni_v1/<run>/probe_iter1000 \
#   SUBSETS=eval_outputs/.../decode_sweep_iter74000 \
#     bash experiments/uniss_qwen3_omni_v1/scripts/run_e2_probe.sh
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
T="${T:-${REPO_ROOT}/data/cvss_t/tokenized/cvss_t_zh_en_v1}"
CKPT="${CKPT:-}"; EXPORT_DIR="${EXPORT_DIR:-}"   # EXPORT_DIR: an export_e2-shaped directory used as is
[ -n "${CKPT}${EXPORT_DIR}" ] || { echo "需要 CKPT（iter 目录或 untrained）或 EXPORT_DIR" >&2; exit 1; }
OUT="$(realpath -m "${OUT:?}")"
REUSE="${REUSE:-${REPO_ROOT}/eval_outputs/uniss_qwen3_omni_v1/e1_full}"
SUBSETS="${SUBSETS:-}"; LENGTH_RATIO="${LENGTH_RATIO:-1.0}"
# The frozen Thinker's states hold for every checkpoint: computed once, read after.
STATES_CACHE="${STATES_CACHE:-${REPO_ROOT}/eval_outputs/uniss_qwen3_omni_v1/e2_thinker_states}"
EXPECTED="${EXPECTED:-$([ -n "${SUBSETS}" ] && echo 500 || echo 4897)}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; read -r -a G <<<"${GPUS}"; NG=${#G[@]}
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
ids() { [ -n "${SUBSETS}" ] && echo ":${SUBSETS}/subset_$1.jsonl"; }

refuse_if_present "${OUT}"; mkdir -p "${OUT}/logs"
EXPORT=untrained
if [ -n "${EXPORT_DIR}" ]; then
  EXPORT="$(realpath "${EXPORT_DIR}")"; log "用现成的导出 ${EXPORT}"
elif [ "${CKPT}" != untrained ]; then
  EXPORT="${OUT}/export"
  log "导出 ${CKPT}"
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=".:third_party/Megatron-LM" "${PYM}" \
    experiments/uniss_qwen3_omni_v1/training/export_e2.py --checkpoint-dir "${CKPT}" --output "${EXPORT}" \
    > "${OUT}/logs/export.log" 2>&1 || { log "E2P_FAILED: 导出失败"; exit 1; }
  tail -4 "${OUT}/logs/export.log"
fi
log "推理（${NG} 卡）"
for ((k = 0; k < NG; k++)); do
  CUDA_VISIBLE_DEVICES="${G[$k]}" VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 \
    "${PYV}" -m experiments.uniss_qwen3_omni_v1.evaluation.e2_infer --export "${EXPORT}" \
    --pair "cmn_eng:${T}/cvss_t_zh_en_test.parquet$(ids cmn_eng)" \
    --pair "eng_cmn:${T}/cvss_t_en_zh_test.parquet$(ids eng_cmn)" \
    --reuse "${REUSE}" --length-ratio "${LENGTH_RATIO}" --states-cache "${STATES_CACHE}" ${DROP:+--drop "${DROP}"} \
    --length-policy "${LENGTH_POLICY:-source}" --clip "${CLIP:-0.85,1.15}" --loop-ngram "${LOOP_NGRAM:-0}" \
    --min-codes-frac "${MIN_CODES_FRAC:-0}" ${EXTRA_STOP:+--extra-stop "${EXTRA_STOP}"} ${SI_GUARD:+--si-guard} \
    --output "${OUT}" --shard "$k" --num-shards "${NG}" > "${OUT}/logs/infer_$k.log" 2>&1 &
done
wait
grep -h "timing" "${OUT}"/logs/infer_*.log | head -2
"${PYE}" - "${OUT}" "${EXPECTED}" <<'PY' || { log "E2P_FAILED: 结果不完整"; exit 1; }
import glob, json, sys
out, n = sys.argv[1], int(sys.argv[2]); bad = []
for d in ("cmn_eng", "eng_cmn"):
    recs = sorted((json.loads(l) for p in glob.glob(f"{out}/{d}/part_*.jsonl") for l in open(p)), key=lambda r: r["index"])
    with open(f"{out}/{d}/results.jsonl", "w") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    print(f"{d}: {len(recs)} 句，无音频 {sum(not r['audio_path'] for r in recs)}，未正常结束 {sum(not r['ended'] for r in recs)}")
    bad += [d] if len(recs) != n else []
sys.exit(1 if bad else 0)
PY
export AUTOPCP_BATCH_SIZE="${AUTOPCP_BATCH_SIZE:-4}"
for pair in "cmn_eng cmn->eng" "eng_cmn eng->cmn"; do
  set -- $pair; D="${OUT}/$1"
  log "打分 $2"
  OMP_NUM_THREADS=8 ALLOW_GENERATED_FAILURES=1 EXPECTED_PAIRS="${EXPECTED}" MODES=tts \
    bash experiments/uniss_omni_s2st_v1/scripts/run_stage1_metrics.sh "$D" "$2" > "$D/metrics.log" 2>&1 \
    || { log "E2P_FAILED: $2 指标失败，见 $D/metrics.log"; exit 1; }
  "${PYE}" -m evaluation.text_metrics --input "$D/results.jsonl" --output "$D/metrics/text_bleu.json" \
    --hypothesis-field translated_text --reference-field translation_ref > /dev/null 2>&1
  CUDA_VISIBLE_DEVICES="${G[0]}" "${PYE}" -m experiments.uniss_omni_s2st_v1.evaluation.sim_o \
    --input "$D/results.jsonl" --output "$D/metrics/sim_o.json" \
    --model /opt/dlami/nvme/jasonleeeli/evaluation_models/wavlm-base-plus-sv --device cuda:0 > "$D/sim_o.log" 2>&1 \
    || { log "E2P_FAILED: $2 SIM-O 失败"; exit 1; }
done
log "E2P_DONE ${OUT}"
