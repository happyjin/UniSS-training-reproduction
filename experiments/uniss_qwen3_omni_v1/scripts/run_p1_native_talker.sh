#!/usr/bin/env bash
# P1: Qwen3-Omni's own talker (p1a) and the same speech put back in the
# source voice through BiCodec (p1b), scored with the full-set harness so
# every number sits beside phase_h and the cascade on the same sentences.
# Inference only: nothing is trained.
#
# SUBSETS names the decoding-sweep subset (500 a direction); without it the
# whole CVSS-T test set is run, 4,897 a direction. Workers take utterances
# from a shared claim directory, so none idles while another finishes, and
# BATCH utterances go through one generate call. The whole model is ~73 GB
# against 141, so it stays one process a card.
set -uo pipefail
refuse_if_present() {
  if [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; then
    echo "拒绝覆盖已有输出: $1" >&2; exit 1
  fi
}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO_ROOT}"
export PYTHONPATH=. PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
export HF_HOME="${HF_HOME:-/opt/dlami/nvme/jasonleeeli/cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY3="${PY3:-/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python}"
PYE="${PYE:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python}"
T="${T:-${REPO_ROOT}/data/cvss_t/tokenized/cvss_t_zh_en_v1}"
SUBSETS="${SUBSETS:-}"   # directory holding subset_cmn_eng.jsonl and subset_eng_cmn.jsonl
# Absolute, because the records carry their audio paths and the metric
# harness resolves a relative one against the results file, not here.
OUT="$(realpath -m "${OUT:?}")"; EXPECTED="${EXPECTED:-$([ -n "${SUBSETS}" ] && echo 500 || echo 4897)}"
BATCH="${BATCH:-1}"; SEED="${SEED:-}"; TALKER_MAX="${TALKER_MAX:-4096}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; STAGGER="${STAGGER:-40}"
read -r -a G <<<"${GPUS}"; NG=${#G[@]}
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

refuse_if_present "${OUT}"; mkdir -p "${OUT}/logs"
subset() { [ -n "${SUBSETS}" ] && echo ":${SUBSETS}/subset_$1.jsonl"; }
gen() {  # worker gpu round
  CUDA_VISIBLE_DEVICES=$2 OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" \
    "${PY3}" -m experiments.uniss_qwen3_omni_v1.evaluation.p1_native_talker \
    --pair "cmn_eng:${T}/cvss_t_zh_en_test.parquet$(subset cmn_eng)" \
    --pair "eng_cmn:${T}/cvss_t_en_zh_test.parquet$(subset eng_cmn)" \
    --claims "${OUT}/claims/round$3" --batch-size "${BATCH}" ${SEED:+--seed "${SEED}"} \
    --talker-max-new-tokens "${TALKER_MAX}" \
    --shard "$1" --num-shards "${NG}" --output "${OUT}" >> "${OUT}/logs/gen_$1.log" 2>&1
}
# The second round has fresh claims, so anything a dead worker had claimed
# but not written is redone; what was written is skipped.
for round in 1 2; do
  log "生成第 ${round} 轮"
  for ((k = 0; k < NG; k++)); do gen "$k" "${G[$k]}" "${round}" & [ "${round}" = 1 ] && sleep "${STAGGER}"; done
  wait
done

"${PYE}" - "${OUT}" "${EXPECTED}" <<'PY' || { log "P1_FAILED: 生成不完整"; exit 1; }
import glob, json, sys
out, n = sys.argv[1], int(sys.argv[2]); bad = []
for system in ("p1a", "p1b"):
    for d in ("cmn_eng", "eng_cmn"):
        rows = {}
        for p in sorted(glob.glob(f"{out}/{system}/{d}/p1_*.jsonl")):
            for line in open(p):
                try:
                    r = json.loads(line); rows[r["id"]] = r   # a redone utterance replaces its first try
                except json.JSONDecodeError:
                    pass
        if len(rows) != n:
            bad.append(f"{system}/{d}: {len(rows)}")
        with open(f"{out}/{system}/{d}/results.jsonl", "w") as fh:
            for r in sorted(rows.values(), key=lambda r: r["index"]):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fails = sum(1 for r in rows.values() if not r.get("audio_path"))
        print(f"{system}/{d}: {len(rows)} 条，无音频 {fails}")
if bad:
    print("不完整:", bad); sys.exit(1)
PY
log "生成完成"

# The four scorings write under their own directories, so they run at once,
# each sharding its ASR, UTMOS and A.PCP over every card and its SIM-O on one.
score() {  # system dir direction sim-gpu
  local D="${OUT}/$1/$2"
  OMP_NUM_THREADS="${SCORE_THREADS:-8}" ALLOW_GENERATED_FAILURES=1 EXPECTED_PAIRS="${EXPECTED}" MODES=tts \
    bash experiments/uniss_omni_s2st_v1/scripts/run_stage1_metrics.sh "$D" "$3" \
    > "$D/metrics.log" 2>&1 || { log "$1 $3 指标失败，见 $D/metrics.log"; return 1; }
  "${PYE}" -m evaluation.text_metrics --input "$D/results.jsonl" \
    --output "$D/metrics/text_bleu.json" --hypothesis-field translated_text \
    --reference-field translation_ref > /dev/null 2>&1
  CUDA_VISIBLE_DEVICES="$4" "${PYE}" -m experiments.uniss_omni_s2st_v1.evaluation.sim_o \
    --input "$D/results.jsonl" --output "$D/metrics/sim_o.json" \
    --model /opt/dlami/nvme/jasonleeeli/evaluation_models/wavlm-base-plus-sv \
    --device cuda:0 > "$D/sim_o.log" 2>&1 || { log "$1 $3 SIM-O 失败，见 $D/sim_o.log"; return 1; }
}
log "打分（4 组同时）"
pids=(); j=0
for system in p1a p1b; do
  for pair in "cmn_eng cmn->eng" "eng_cmn eng->cmn"; do
    set -- $pair; score "${system}" "$1" "$2" "${G[$((j % NG))]}" & pids+=($!); j=$((j + 1))
  done
done
failed=0; for p in "${pids[@]}"; do wait "$p" || failed=1; done
[ "${failed}" = 0 ] || { log "P1_FAILED: 打分有失败"; exit 1; }
log "P1_DONE ${OUT}"
