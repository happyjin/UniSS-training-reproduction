#!/usr/bin/env bash
# One Hy-SimulMT-30B-A3B vLLM server on one card, as SimulInfer 0.5.18 deploy.py model_command (registration
# paper/baselines_run_plan_hysimulmt_v1.md §1); only the resource flags differ (max-num-seqs, gpu-memory-utilization,
# max-model-len). Usage: GPU=<card> PORT=<port> [UTIL=0.55] bash serve_hy.sh  (run it under setsid nohup)
set -euo pipefail
GPU="${GPU:?}"; PORT="${PORT:?}"; UTIL="${UTIL:-0.55}"
V=/opt/dlami/nvme/jasonleeeli/venvs/hysimulmt
MODEL=/opt/dlami/nvme/jasonleeeli/models/Hy-SimulMT-30B-A3B_simulinfer0518
export PATH="$V/bin:$PATH" CUDA_VISIBLE_DEVICES="$GPU" HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface \
  VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_hysimulmt OMP_NUM_THREADS=8 PYTHONNOUSERSITE=1
exec "$V/bin/python" -m vllm.entrypoints.cli.main serve "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --served-model-name Hy-SimulMT-30B-A3B --dtype bfloat16 --tensor-parallel-size 1 \
  --max-num-seqs 64 --gpu-memory-utilization "$UTIL" --max-model-len 16384 \
  --enable-prefix-caching --enable-chunked-prefill --limit-mm-per-prompt '{"audio":1024}' \
  --chat-template-content-format openai --logits-processors simulinfer.logits_plugins:NoChineseTargetProcessor
