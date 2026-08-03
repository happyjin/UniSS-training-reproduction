#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
PYTHON=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python
SOURCE="$ROOT/data/processed/simul_uniss_subsecond_v1/pilot_15shard/stage_a_source/stage_a_source_manifest.jsonl"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_KERNEL_CACHE_PATH=/opt/dlami/nvme/jasonleeeli/cache/torch_kernel

"$PYTHON" "$ROOT/experiments/uniss_streamspeech_ctc_v1/stage05_ctc_policy/evaluate_real_policy.py" \
  --dataset-index "$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage02_ctc_probe/dataset_index.json" \
  --source-manifest "$SOURCE" \
  --source-offsets "${SOURCE}.offsets.bin" \
  --tokenizer-dir "$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage01_data/tokenizers" \
  --checkpoint "$ROOT/checkpoints/uniss_streamspeech_ctc_v1/stage03b_ar_s2tt_b16_v3/best.pt" \
  --split valid \
  --max-samples 256 \
  --confirmations 2 \
  --lagging-k 0 \
  --output-json "$ROOT/reports/uniss_streamspeech_ctc_v1/stage05_stage03b_valid_256.json" \
  --output-md "$ROOT/reports/uniss_streamspeech_ctc_v1/stage05_stage03b_valid_256.md"

