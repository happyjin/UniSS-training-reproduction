#!/bin/bash
# Detached future-candidate sampling, one vLLM shard per card. Usage: launch_candidates.sh <shard> <num_shards> <card>
set -euo pipefail
SH="${1:?shard}"; NS="${2:?num_shards}"; CARD="${3:?card}"
ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS/experiments/llm_vits_route_v1
V=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030
mkdir -p "$ROOT/data/future_cands" "$ROOT/runs/logs"
cd "$ROOT"
PATH=$V/bin:$PATH CUDA_VISIBLE_DEVICES=$CARD VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 \
  setsid nohup $V/bin/python scripts/sample_future_candidates.py --manifest data/manifest.jsonl \
  --model /opt/dlami/nvme/jasonleeeli/hf_models/Qwen3-1.7B --out data/future_cands/shard${SH}of${NS}.jsonl \
  --temperature 0.8 --top-p 0.95 --shard "$SH" --num-shards "$NS" \
  > runs/logs/cands_shard${SH}.log 2>&1 < /dev/null & disown
echo "shard $SH/$NS on card $CARD"
