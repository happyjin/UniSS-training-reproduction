#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
PYTHON=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python
STAGE="$ROOT/experiments/uniss_streamspeech_ctc_v1/stage03_multitask_encoder"
DATA="$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage02_ctc_probe/dataset_index.json"
SOURCE="$ROOT/data/processed/simul_uniss_subsecond_v1/pilot_15shard/stage_a_source/stage_a_source_manifest.jsonl"
TOKENIZERS="$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage01_data/tokenizers"
INITIAL="$ROOT/checkpoints/simul_uniss_subsecond_v3/stage_b_v3_balanced_hidden_15shard_v1/candidates/step_008000.pt"
RESUME="$ROOT/checkpoints/uniss_streamspeech_ctc_v1/stage03_multitask_encoder_v1/latest_resume.pt"
LENGTHS="$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage03_multitask_encoder/train_lengths.u32"
OUTPUT="$ROOT/checkpoints/uniss_streamspeech_ctc_v1/stage03_multitask_encoder_bucket_v2"
TENSORBOARD="$ROOT/runs/uniss_streamspeech_ctc_v1/stage03_multitask_encoder_bucket_v2"

test -f "$DATA"
test -f "$INITIAL"
test -f "$RESUME"
test -f "$LENGTHS"
mkdir -p "$OUTPUT" "$TENSORBOARD"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

"$PYTHON" -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_port=29631 \
  "$STAGE/train_encoder.py" \
  --dataset-index "$DATA" \
  --source-manifest "$SOURCE" \
  --source-offsets "${SOURCE}.offsets.bin" \
  --tokenizer-dir "$TOKENIZERS" \
  --initialize-from "$INITIAL" \
  --resume "$RESUME" \
  --length-index "$LENGTHS" \
  --output-dir "$OUTPUT" \
  --tensorboard-dir "$TENSORBOARD" \
  --batch-size 16 \
  --num-workers 4 \
  --max-steps 10000 \
  --eval-every 1000
