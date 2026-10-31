#!/usr/bin/env bash
# Start talker_worker.py workers, detached, one card each, 75 s apart (vLLM engines must not profile together).
#   bash reports/qabc_v10/start_workers.sh <tasks.jsonl> <gpu_mem> <card> [<card> ...]     (talker workers)
#   bash reports/qabc_v10/start_workers.sh <tasks.jsonl> asr <card>                         (one ASR worker)
set -u
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
T=$1; MODE=$2; shift 2
ENV="PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8"
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
first=1
for c in "$@"; do
  [ $first = 1 ] || sleep 75
  first=0
  if [ "$MODE" = asr ]; then
    L=logs/qabc_v10/asr_worker_c${c}_$(date -u +%H%M%S).log
    env $ENV CUDA_VISIBLE_DEVICES=$c setsid nohup $PYV -u reports/qabc_v10/talker_worker.py asr "$T" > "$L" 2>&1 < /dev/null &
  else
    L=logs/qabc_v10/talker_worker_c${c}_$(date -u +%H%M%S).log
    env $ENV CUDA_VISIBLE_DEVICES=$c setsid nohup $PYV -u reports/qabc_v10/talker_worker.py talker "$T" --gpu-mem "$MODE" > "$L" 2>&1 < /dev/null &
  fi
  echo "[$(date -u '+%F %T') UTC] Q(v10): talker_worker $MODE on card $c (pid $!), log $L" | tee -a logs/gpu_coord/events.log
done
