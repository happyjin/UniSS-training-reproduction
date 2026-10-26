#!/usr/bin/env bash
# Talker decoding sweep on a random CVSS-T subset, scored for ASR-BLEU,
# fidelity to the model's own translation, and UTMOS.
#
# phase_d_sweep runs the thinker once per utterance and the talker once
# per setting. Every setting's rows carry the setting as `mode`, so one ASR
# pass and one UTMOS pass cover all of them -- eight GPUs, models loaded
# once -- and text_metrics returns one BLEU per setting and direction.
set -uo pipefail

refuse_if_present() {
  if [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; then
    echo "拒绝覆盖已有输出: $1" >&2; exit 1
  fi
}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO_ROOT}"
export PYTHONPATH=. PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/opt/dlami/nvme/jasonleeeli/cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/opt/dlami/nvme/jasonleeeli/cache/modelscope}"
export TORCH_HOME="${TORCH_HOME:-/opt/dlami/nvme/jasonleeeli/cache/torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY3="${PY3:-/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python}"
PYE="${PYE:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python}"
T="${T:-${REPO_ROOT}/data/cvss_t/tokenized/cvss_t_zh_en_v1}"
TALKER="${TALKER:?}"; CKPT="${CKPT:?}"; OUT="${OUT:?}"; CONFIGS="${CONFIGS:?}"
N_PER_DIR="${N_PER_DIR:-500}"; GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; STAGGER="${STAGGER:-40}"
read -r -a G <<<"${GPUS}"; NG=${#G[@]}
DIRS=("cmn_eng cvss_t_zh_en_test" "eng_cmn cvss_t_en_zh_test")
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# RESUME=1 continues an interrupted sweep in place: the subsets it drew
# are kept, and phase_d_sweep skips what each setting already holds. That
# is also how settings are dropped mid-run -- relaunch with fewer.
if [ "${RESUME:-0}" = "1" ]; then
  for d in "${DIRS[@]}"; do set -- $d; [ -s "${OUT}/subset_$1.jsonl" ] || { echo "RESUME=1 but ${OUT}/subset_$1.jsonl is missing" >&2; exit 1; }; done
  log "续跑 ${OUT}"
else
refuse_if_present "${OUT}"; mkdir -p "${OUT}/logs"

# A random subset, not the first N by id: the id-sorted head is a biased
# slice of the test set (+3.30 BLEU on the first 64).
for d in "${DIRS[@]}"; do
  set -- $d
  "${PYE}" - "${T}/$2.parquet" "${N_PER_DIR}" "${OUT}/subset_$1.jsonl" <<'PY'
import json, random, sys
import pyarrow.parquet as pq
ids = sorted(pq.read_table(sys.argv[1], columns=["id"]).column("id").to_pylist())
pick = random.Random(20261019).sample(ids, int(sys.argv[2]))
open(sys.argv[3], "w").write("".join(json.dumps({"id": i}) + "\n" for i in sorted(pick)))
PY
done
fi
log "子集：每方向 ${N_PER_DIR} 句；配置：${CONFIGS}"

gen() {  # name parquet shard gpu
  CUDA_VISIBLE_DEVICES=$4 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
    "${PY3}" -m experiments.uniss_qwen3_omni_v1.evaluation.phase_d_sweep \
    --talker-checkpoint "${TALKER}" --phase-d-checkpoint "${CKPT}" \
    --dev-parquet "${T}/$2.parquet" --subset "${OUT}/subset_$1.jsonl" \
    --configs "${CONFIGS}" --shard "$3" --num-shards "${NG}" \
    --output "${OUT}/$1" >> "${OUT}/logs/gen_$1_$3.log" 2>&1
}
# Two rounds: the second resumes any shard the first left unfinished and
# returns at once for the rest, before loading a model.
for round in 1 2; do
  log "生成第 ${round} 轮"
  for d in "${DIRS[@]}"; do
    set -- $d
    for ((k = 0; k < NG; k++)); do
      gen "$1" "$2" "$k" "${G[$k]}" &
      [ "${round}" = 1 ] && sleep "${STAGGER}"
    done
  done
  wait
done

# Every setting must hold the whole subset in both directions.
"${PYE}" - "${OUT}" "${CONFIGS}" "${N_PER_DIR}" <<'PY' || { log "SWEEP_FAILED: 生成不完整"; exit 1; }
import glob, json, sys
out, spec, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
bad = []
with open(f"{out}/all.jsonl", "w") as fh:
    for d in ("cmn_eng", "eng_cmn"):
        for name in (c.split(":")[0] for c in spec.split(",")):
            rows = {}
            for p in sorted(glob.glob(f"{out}/{d}/{name}/e2e_*.jsonl")):
                for line in open(p):
                    try:
                        r = json.loads(line); rows[r["id"]] = r
                    except json.JSONDecodeError:
                        pass
            if len(rows) != n:
                bad.append(f"{d}/{name}: {len(rows)}")
            for r in rows.values():
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
if bad:
    print("不完整:", bad); sys.exit(1)
PY
log "生成完成：$(wc -l < "${OUT}/all.jsonl") 条"

# ASR and UTMOS side by side on every card; each is a few GB.
mkdir -p "${OUT}/asr_parts" "${OUT}/utmos_parts"
for ((k = 0; k < NG; k++)); do
  CUDA_VISIBLE_DEVICES="${G[$k]}" "${PYE}" -m evaluation.asr_transcribe \
    --input "${OUT}/all.jsonl" --output "${OUT}/asr_parts/part_${k}.jsonl" \
    --whisper-model "${WHISPER_MODEL:-openai/whisper-large-v3}" \
    --paraformer-model "${PARAFORMER_MODEL:-iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch}" \
    --device cuda:0 --batch-size 16 --num-shards "${NG}" --shard-index "$k" \
    > "${OUT}/logs/asr_${k}.log" 2>&1 &
  mkdir -p "${OUT}/utmos_parts/part_${k}"
  CUDA_VISIBLE_DEVICES="${G[$k]}" "${PYE}" -m evaluation.utmos_metrics \
    --input "${OUT}/all.jsonl" --output-dir "${OUT}/utmos_parts/part_${k}" \
    --device cuda:0 --num-shards "${NG}" --shard-index "$k" \
    > "${OUT}/logs/utmos_${k}.log" 2>&1 &
done
wait
cat "${OUT}"/asr_parts/part_*.jsonl > "${OUT}/asr_all.jsonl"
cat "${OUT}"/utmos_parts/part_*/per_sample*.jsonl > "${OUT}/utmos_all.jsonl" 2>/dev/null
log "ASR $(wc -l < "${OUT}/asr_all.jsonl") 条，UTMOS $(wc -l < "${OUT}/utmos_all.jsonl") 条"
"${PYE}" -m evaluation.text_metrics --input "${OUT}/asr_all.jsonl" \
  --output "${OUT}/asr_bleu.json" --hypothesis-field asr_text \
  --reference-field translation_ref > "${OUT}/logs/asr_bleu.log" 2>&1
"${PYE}" -m evaluation.text_metrics --input "${OUT}/all.jsonl" \
  --output "${OUT}/text_bleu.json" --hypothesis-field translated_text \
  --reference-field translation_ref > "${OUT}/logs/text_bleu.log" 2>&1
log "SWEEP_DONE ${OUT}"
