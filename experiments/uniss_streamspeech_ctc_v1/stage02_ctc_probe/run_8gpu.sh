#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
PYTHON=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python
STAGE="$ROOT/experiments/uniss_streamspeech_ctc_v1/stage02_ctc_probe"
DATA="$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage02_ctc_probe/dataset_index.json"
TOKENIZERS="$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage01_data/tokenizers"
OUTPUT="$ROOT/checkpoints/uniss_streamspeech_ctc_v1/stage02_ctc_probe_v1"
TENSORBOARD="$ROOT/runs/uniss_streamspeech_ctc_v1/stage02_ctc_probe_v1"

test -f "$DATA"
test -f "$TOKENIZERS/tokenizers.json"
mkdir -p "$OUTPUT" "$TENSORBOARD"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

"$PYTHON" -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_port=29620 \
  "$STAGE/train_probe.py" \
  --dataset-index "$DATA" \
  --tokenizer-dir "$TOKENIZERS" \
  --output-dir "$OUTPUT" \
  --tensorboard-dir "$TENSORBOARD" \
  --batch-size 128 \
  --num-workers 4 \
  --max-steps 3000 \
  --eval-every 250 \
  --eval-batches 40

