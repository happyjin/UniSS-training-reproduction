#!/usr/bin/env bash
# Does raising the code-row scale help, given the conditioning vector is 36x it?
#
# A controlled comparison, not a replacement run: identical config, identical
# global batch, identical seed and dev set -- only --embed-scale differs. The
# embed_scale=1.0 arm is already on disk as the main Stage 1 curve, so this
# runs the other arm to the same step and the two are read off together.
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
SCALE="${SCALE:-8.0}"
STEPS="${STEPS:-2500}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/checkpoints/uniss_omni_s2st_v1/probe_embed_scale_${SCALE}}"

SHARDS=(train-00001.parquet train-00004.parquet train-00013.parquet train-00030.parquet
        train-00040.parquet train-00067.parquet train-00070.parquet train-00124.parquet
        train-00142.parquet train-00164.parquet)

"${PY}" -m torch.distributed.run --nproc_per_node=8 --master_port="${MASTER_PORT:-29533}" \
  -m experiments.uniss_omni_s2st_v1.training.stage1_talker_warmup \
  --shards "${SHARDS[@]}" \
  --dev-parquet "${DEV}/cvss_t_zh_en_dev.parquet" "${DEV}/cvss_t_en_zh_dev.parquet" \
  --dev-subset "${SUBSET}" \
  --batch-size 8 --accum 2 --lr 2e-4 --warmup 200 \
  --steps "${STEPS}" --eval-every 250 --save-every 100000 --dev-batches 16 \
  --embed-scale "${SCALE}" \
  --output "${OUTPUT}"
echo "EMBED SCALE PROBE DONE scale=${SCALE}"
