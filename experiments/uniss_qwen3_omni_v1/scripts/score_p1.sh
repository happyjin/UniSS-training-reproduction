#!/usr/bin/env bash
# Scores a P1 output directory with exactly the harness
# run_p1_native_talker.sh uses, so every P1 run sits beside the others line
# for line. After p1_vllm_omni.py it first makes P1-b by BiCodec swap;
# SKIP_SWAP=1 scores a directory whose p1b is already there, such as a
# run_p1_native_talker.sh run whose scoring has to be redone.
#
#   OUT=eval_outputs/uniss_qwen3_omni_v1/p1_vllm_omni_full \
#     bash experiments/uniss_qwen3_omni_v1/scripts/score_p1.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO_ROOT}"
export PYTHONPATH=. PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/opt/dlami/nvme/jasonleeeli/cache/huggingface}"
PY3="${PY3:-/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python}"
PYE="${PYE:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python}"
T="${T:-${REPO_ROOT}/data/cvss_t/tokenized/cvss_t_zh_en_v1}"
OUT="$(realpath -m "${OUT:?}")"; EXPECTED="${EXPECTED:-4897}"
# A few thinker outputs loop to the token cap and their speech runs for
# minutes; A.PCP pads a batch to its longest item, and at 32 such a batch
# wants more than a card. Batch size changes the speed, not the scores.
export AUTOPCP_BATCH_SIZE="${AUTOPCP_BATCH_SIZE:-4}"
SCORE_JOBS="${SCORE_JOBS:-1}"   # scorings at once; each already shards over every card
ONLY="${ONLY:-}"                # e.g. "p1a/cmn_eng p1b/eng_cmn" to redo just those
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; read -r -a G <<<"${GPUS}"; NG=${#G[@]}
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

if [ -z "${SKIP_SWAP:-}" ]; then
  for d in cmn_eng eng_cmn; do
    [ -e "${OUT}/p1b/${d}" ] && { echo "拒绝覆盖已有输出: ${OUT}/p1b/${d}" >&2; exit 1; }
  done
  log "P1-b 换音色"
  pids=(); j=0
  for pair in "cmn_eng cvss_t_zh_en_test" "eng_cmn cvss_t_en_zh_test"; do
    set -- $pair
    CUDA_VISIBLE_DEVICES="${G[$((j % NG))]}" "${PY3}" -m experiments.uniss_qwen3_omni_v1.evaluation.p1b_swap \
      --p1a "${OUT}/p1a/$1" --parquet "${T}/$2.parquet" --output "${OUT}/p1b/$1" \
      > "${OUT}/swap_$1.log" 2>&1 & pids+=($!); j=$((j + 1))
  done
  failed=0; for p in "${pids[@]}"; do wait "$p" || failed=1; done
  [ "${failed}" = 0 ] || { log "P1V_FAILED: 换音色失败"; exit 1; }
fi

"${PYE}" - "${OUT}" "${EXPECTED}" <<'PY' || { log "P1V_FAILED: 结果不完整"; exit 1; }
import glob, json, os, sys
out, n = sys.argv[1], int(sys.argv[2]); bad = []
for system in ("p1a", "p1b"):
    for d in ("cmn_eng", "eng_cmn"):
        rows = {}
        for p in sorted(glob.glob(f"{out}/{system}/{d}/p1_*.jsonl")):
            for line in open(p):
                try:
                    r = json.loads(line); rows[r["id"]] = r
                except json.JSONDecodeError:
                    pass
        for r in rows.values():  # a relative path would be read against results.jsonl's directory
            if r.get("audio_path"):
                r["audio_path"] = os.path.abspath(r["audio_path"])
        if len(rows) != n:
            bad.append(f"{system}/{d}: {len(rows)}")
        with open(f"{out}/{system}/{d}/results.jsonl", "w") as fh:
            for r in sorted(rows.values(), key=lambda r: r["index"]):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{system}/{d}: {len(rows)} 条，无音频 {sum(1 for r in rows.values() if not r.get('audio_path'))}")
if bad:
    print("不完整:", bad); sys.exit(1)
PY

# The same four scorings as run_p1_native_talker.sh, at once.
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
log "打分（同时 ${SCORE_JOBS} 组）"
pids=(); j=0; failed=0
for system in p1a p1b; do
  for pair in "cmn_eng cmn->eng" "eng_cmn eng->cmn"; do
    set -- $pair
    [ -n "${ONLY}" ] && [[ " ${ONLY} " != *" ${system}/$1 "* ]] && continue
    score "${system}" "$1" "$2" "${G[$((j % NG))]}" & pids+=($!); j=$((j + 1))
    if [ "${#pids[@]}" -ge "${SCORE_JOBS}" ]; then wait "${pids[0]}" || failed=1; pids=("${pids[@]:1}"); fi
  done
done
for p in "${pids[@]}"; do wait "$p" || failed=1; done
[ "${failed}" = 0 ] || { log "P1V_FAILED: 打分有失败"; exit 1; }
log "P1V_DONE ${OUT}"
