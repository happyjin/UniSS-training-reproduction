#!/usr/bin/env bash
# Shard UniST test across 8 GPUs for the 3B Phase2 HF export.
# Scores Text-BLEU and SLC. Speech-BLEU, UTMOS, and AutoPCP need the
# separate eval environment and metric models, which are not on this machine.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/opt/dlami/nvme/neuhao/conda_envs/uniss-train}"
PYTHON="${ENV_ROOT}/bin/python"
MODEL="${MODEL:-${REPO_ROOT}/checkpoints/exported_hf/qwen3b_phase2_unist198_iter_0010645_hf}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/experiments/evaluation/uniss_full198_phase2_phase3/manifests/unist_test_all.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/eval_outputs/qwen3b_phase2_unist198_iter_0010645_unist_test_all}"
SPEECH_TOKENIZER="${SPEECH_TOKENIZER:-${REPO_ROOT}/pretrained_models/UniSS}"
NUM_GPUS="${NUM_GPUS:-8}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -e "${OUTPUT_ROOT}/COMPLETE" ]]; then
  echo "Already complete: ${OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}/shards" "${OUTPUT_ROOT}/logs"

"${PYTHON}" - "${MANIFEST}" "${OUTPUT_ROOT}/shards" "${NUM_GPUS}" <<'PY'
import sys
from pathlib import Path
manifest, shard_root, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
lines = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
handles = []
for index in range(n):
    path = shard_root / f"manifest_{index}.jsonl"
    handles.append(path.open("w", encoding="utf-8"))
for index, line in enumerate(lines):
    handles[index % n].write(line + "\n")
for handle in handles:
    handle.close()
print(f"split {len(lines)} rows into {n} shards")
PY

pids=()
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  shard_manifest="${OUTPUT_ROOT}/shards/manifest_${gpu}.jsonl"
  shard_output="${OUTPUT_ROOT}/shards/gpu_${gpu}"
  if [[ -e "${shard_output}" ]]; then
    echo "Refusing to overwrite ${shard_output}" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${REPO_ROOT}/training/generate_unist_eval_audio.py" \
    --manifest "${shard_manifest}" \
    --model "${MODEL}" \
    --speech-tokenizer "${SPEECH_TOKENIZER}" \
    --output-dir "${shard_output}" \
    --mode quality performance \
    --limit-records 0 \
    --max-new-tokens 1500 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k -1 \
    --repetition-penalty 1.1 \
    --seed 20260726 \
    --dtype bfloat16 \
    --device cuda:0 \
    --local-files-only \
    --save-source-audio \
    > "${OUTPUT_ROOT}/logs/gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
  echo "gpu ${gpu} pid ${pids[-1]}"
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    echo "worker ${pid} failed" >&2
    fail=1
  fi
done
if [[ "${fail}" != "0" ]]; then
  echo "generation failed; shard logs are in ${OUTPUT_ROOT}/logs" >&2
  exit 1
fi

cat "${OUTPUT_ROOT}"/shards/gpu_*/results.jsonl > "${OUTPUT_ROOT}/results.jsonl"
mkdir -p "${OUTPUT_ROOT}/metrics"
"${PYTHON}" -m evaluation.text_metrics \
  --input "${OUTPUT_ROOT}/results.jsonl" \
  --output "${OUTPUT_ROOT}/metrics/text_bleu.json"
"${PYTHON}" -m evaluation.slc_metrics \
  --input "${OUTPUT_ROOT}/results.jsonl" \
  --output-dir "${OUTPUT_ROOT}/metrics"
date -u +%FT%TZ > "${OUTPUT_ROOT}/COMPLETE"
echo "COMPLETE ${OUTPUT_ROOT}"
