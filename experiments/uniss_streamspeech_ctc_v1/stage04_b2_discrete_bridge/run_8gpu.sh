#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
PYTHON=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python
STAGE="$ROOT/experiments/uniss_streamspeech_ctc_v1/stage04_b2_discrete_bridge"
SOURCE="$ROOT/data/processed/simul_uniss_subsecond_v1/pilot_15shard/stage_a_source/stage_a_source_manifest.jsonl"
LOG="$ROOT/logs/uniss_streamspeech_ctc_v1/stage04_b2_phase3_endpoint_v1.log"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

mkdir -p "$(dirname "$LOG")"

"$PYTHON" -m torch.distributed.run --nproc_per_node=8 --master_port=29650 \
  "$STAGE/train_b2.py" \
  --dataset-index "$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage02_ctc_probe/dataset_index.json" \
  --source-manifest "$SOURCE" \
  --source-offsets "${SOURCE}.offsets.bin" \
  --ctc-tokenizer-dir "$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage01_data/tokenizers" \
  --endpoint-checkpoint "$ROOT/checkpoints/uniss_streamspeech_ctc_v1/stage03b_ar_s2tt_b16_v3/best.pt" \
  --historical-stage-b-checkpoint "$ROOT/checkpoints/simul_uniss_subsecond_v3/stage_b_v3_balanced_hidden_15shard_v1/candidates/step_008000.pt" \
  --codebook-model "$ROOT/pretrained_models/UniSS/glm4_tokenizer" \
  --phase3-model "$ROOT/checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf" \
  --length-index "$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage03_multitask_encoder/train_lengths.u32" \
  --output-dir "$ROOT/checkpoints/uniss_streamspeech_ctc_v1/stage04_b2_phase3_endpoint_v1" \
  --tensorboard-dir "$ROOT/runs/uniss_streamspeech_ctc_v1/stage04_b2_phase3_endpoint_v1" \
  --batch-size 16 \
  --num-workers 4 \
  --max-steps 2000 \
  --eval-every 200 \
  2>&1 | tee "$LOG"
