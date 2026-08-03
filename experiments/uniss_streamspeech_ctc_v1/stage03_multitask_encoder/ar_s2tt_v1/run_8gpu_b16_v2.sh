#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
PYTHON=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python
STAGE="$ROOT/experiments/uniss_streamspeech_ctc_v1/stage03_multitask_encoder/ar_s2tt_v1"
SOURCE="$ROOT/data/processed/simul_uniss_subsecond_v1/pilot_15shard/stage_a_source/stage_a_source_manifest.jsonl"
INITIAL="$ROOT/checkpoints/uniss_streamspeech_ctc_v1/stage03_multitask_encoder_bucket_b32_v3/best.pt"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

"$PYTHON" -m torch.distributed.run --nproc_per_node=8 --master_port=29641 \
  "$STAGE/train.py" \
  --dataset-index "$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage02_ctc_probe/dataset_index.json" \
  --source-manifest "$SOURCE" \
  --source-offsets "${SOURCE}.offsets.bin" \
  --tokenizer-dir "$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage01_data/tokenizers" \
  --initialize-from "$INITIAL" \
  --length-index "$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage03_multitask_encoder/train_lengths.u32" \
  --output-dir "$ROOT/checkpoints/uniss_streamspeech_ctc_v1/stage03b_ar_s2tt_b16_v2" \
  --tensorboard-dir "$ROOT/runs/uniss_streamspeech_ctc_v1/stage03b_ar_s2tt_b16_v2" \
  --batch-size 16 \
  --max-steps 5000 \
  --eval-every 500
