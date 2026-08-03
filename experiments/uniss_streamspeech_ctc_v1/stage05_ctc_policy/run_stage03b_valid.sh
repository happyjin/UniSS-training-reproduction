#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
PYTHON=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python
SOURCE="$ROOT/data/processed/simul_uniss_subsecond_v1/pilot_15shard/stage_a_source/stage_a_source_manifest.jsonl"
PART_DIR="$ROOT/reports/uniss_streamspeech_ctc_v1/stage05_stage03b_valid_256_v1_parts"
OUTPUT_JSON="$ROOT/reports/uniss_streamspeech_ctc_v1/stage05_stage03b_valid_256_v1.json"
OUTPUT_MD="$ROOT/reports/uniss_streamspeech_ctc_v1/stage05_stage03b_valid_256_v1.md"

export PYTORCH_KERNEL_CACHE_PATH=/opt/dlami/nvme/jasonleeeli/cache/torch_kernel
mkdir -p "$PYTORCH_KERNEL_CACHE_PATH"
test ! -e "$PART_DIR"
test ! -e "$OUTPUT_JSON"
test ! -e "$OUTPUT_MD"
mkdir -p "$PART_DIR"

pids=()
for rank in 0 1 2 3 4 5 6 7; do
  start=$((rank * 32))
  CUDA_VISIBLE_DEVICES="$rank" "$PYTHON" \
    "$ROOT/experiments/uniss_streamspeech_ctc_v1/stage05_ctc_policy/evaluate_real_policy.py" \
    --dataset-index "$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage02_ctc_probe/dataset_index.json" \
    --source-manifest "$SOURCE" \
    --source-offsets "${SOURCE}.offsets.bin" \
    --tokenizer-dir "$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage01_data/tokenizers" \
    --checkpoint "$ROOT/checkpoints/uniss_streamspeech_ctc_v1/stage03b_ar_s2tt_b16_v3/best.pt" \
    --split valid \
    --start-index "$start" \
    --max-samples 32 \
    --confirmations 2 \
    --lagging-k 0 \
    --output-json "$PART_DIR/part_${rank}.json" \
    --output-md "$PART_DIR/part_${rank}.md" \
    >"$PART_DIR/part_${rank}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PYTHON" "$ROOT/experiments/uniss_streamspeech_ctc_v1/stage05_ctc_policy/merge_policy_reports.py" \
  --parts "$PART_DIR"/part_?.json \
  --output-json "$OUTPUT_JSON" \
  --output-md "$OUTPUT_MD"
