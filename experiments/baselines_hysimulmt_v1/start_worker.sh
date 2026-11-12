#!/usr/bin/env bash
# One talker or ASR worker (talker_worker_hy.py) on one card; run under setsid nohup.
# Usage: MODE=talker|asr GPU=<card> bash experiments/baselines_hysimulmt_v1/start_worker.sh
set -euo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
MODE="${MODE:?}"; GPU="${GPU:?}"
export CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 \
  HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface MODELSCOPE_CACHE=/opt/dlami/nvme/jasonleeeli/cache/modelscope
T=eval_outputs/baselines_hysimulmt_v1/tasks.jsonl
if [ "$MODE" = talker ]; then
  export PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030
  exec /opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python experiments/baselines_hysimulmt_v1/talker_worker_hy.py talker $T --gpu-mem 0.15
else
  exec /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python experiments/baselines_hysimulmt_v1/talker_worker_hy.py asr $T
fi
