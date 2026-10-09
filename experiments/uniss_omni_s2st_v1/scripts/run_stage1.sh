#!/usr/bin/env bash
# Stage 1: Talker warmup on the BiCodec codebook, 8 GPUs, DDP.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}" PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/opt/dlami/nvme/jasonleeeli/cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python
DEV=/opt/dlami/nvme/jasonleeeli/CVSS/tokenized/cvss_t_zh_en_dev_v1
SUBSET=/opt/dlami/nvme/jasonleeeli/CVSS/manifests/cvss_t_zh_en_dev_v1/cvss_t_zh_en_dev_subset1000.jsonl
OUTPUT="${OUTPUT:-${REPO_ROOT}/checkpoints/uniss_omni_s2st_v1/stage1}"

SHARDS=(train-00001.parquet train-00004.parquet train-00013.parquet train-00030.parquet
        train-00040.parquet train-00067.parquet train-00070.parquet train-00124.parquet
        train-00142.parquet train-00164.parquet)

"${PY}" -m torch.distributed.run --nproc_per_node=8 --master_port="${MASTER_PORT:-29531}" \
  -m experiments.uniss_omni_s2st_v1.training.stage1_talker_warmup \
  --shards "${SHARDS[@]}" \
  --dev-parquet "${DEV}/cvss_t_zh_en_dev.parquet" "${DEV}/cvss_t_en_zh_dev.parquet" \
  --dev-subset "${SUBSET}" \
  --rows-per-shard "${ROWS_PER_SHARD:-0}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --accum "${ACCUM:-2}" \
  --lr "${LR:-2e-4}" \
  --warmup "${WARMUP:-200}" \
  --steps "${STEPS:-15000}" \
  --eval-every "${EVAL_EVERY:-250}" \
  --save-every "${SAVE_EVERY:-1000}" \
  --dev-batches "${DEV_BATCHES:-16}" \
  --output "${OUTPUT}"
echo "STAGE1 LAUNCHER DONE"
