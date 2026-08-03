#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
PYTHON=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python
STAGE="$ROOT/experiments/uniss_streamspeech_ctc_v1/stage04_b2_discrete_bridge"
SOURCE="$ROOT/data/processed/simul_uniss_subsecond_v1/pilot_15shard/stage_a_source/stage_a_source_manifest.jsonl"
RUN_NAME=${RUN_NAME:-stage04_b2_text_probe32_v1}
BRIDGE_CHECKPOINT=${BRIDGE_CHECKPOINT:-$ROOT/checkpoints/uniss_streamspeech_ctc_v1/stage04_b2_phase3_endpoint_v1/best.pt}
PART_DIR="$ROOT/reports/uniss_streamspeech_ctc_v1/${RUN_NAME}_parts"
OUTPUT_JSON="$ROOT/reports/uniss_streamspeech_ctc_v1/${RUN_NAME}.json"
OUTPUT_MD="$ROOT/reports/uniss_streamspeech_ctc_v1/${RUN_NAME}.md"

export PYTORCH_KERNEL_CACHE_PATH=/opt/dlami/nvme/jasonleeeli/cache/torch_kernel
mkdir -p "$PYTORCH_KERNEL_CACHE_PATH"
test ! -e "$PART_DIR"
test ! -e "$OUTPUT_JSON"
test ! -e "$OUTPUT_MD"
mkdir -p "$PART_DIR"

pids=()
for rank in 0 1 2 3 4 5 6 7; do
  direction=$((rank / 4))
  direction_offset=$(((rank % 4) * 4))
  CUDA_VISIBLE_DEVICES="$rank" "$PYTHON" "$STAGE/evaluate_text_bleu.py" \
    --dataset-index "$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage02_ctc_probe/dataset_index.json" \
    --source-manifest "$SOURCE" \
    --source-offsets "${SOURCE}.offsets.bin" \
    --ctc-tokenizer-dir "$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage01_data/tokenizers" \
    --endpoint-checkpoint "$ROOT/checkpoints/uniss_streamspeech_ctc_v1/stage03b_ar_s2tt_b16_v3/best.pt" \
    --historical-stage-b-checkpoint "$ROOT/checkpoints/simul_uniss_subsecond_v3/stage_b_v3_balanced_hidden_15shard_v1/candidates/step_008000.pt" \
    --bridge-checkpoint "$BRIDGE_CHECKPOINT" \
    --codebook-model "$ROOT/pretrained_models/UniSS/glm4_tokenizer" \
    --phase3-model "$ROOT/checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf" \
    --direction-id "$direction" \
    --direction-offset "$direction_offset" \
    --max-samples 4 \
    --output-json "$PART_DIR/part_${rank}.json" \
    >"$PART_DIR/part_${rank}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PYTHON" "$STAGE/merge_text_bleu.py" \
  --parts "$PART_DIR"/part_?.json \
  --output-json "$OUTPUT_JSON" \
  --output-md "$OUTPUT_MD"
