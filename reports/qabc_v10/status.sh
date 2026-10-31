#!/usr/bin/env bash
# One-line v10 status: complete RealSI/dev speech runs, live workers, failures, the last hot-card line and MIST messages
# to Q(v10) of the last 15 minutes, per-card memory.
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
O=eval_outputs/uniss_qwen3_omni_v1/qabc_v10
echo "$(date -u +%H:%M) speech $(ls -d $O/speech/*/*/realsi_*/asr_speech.jsonl 2>/dev/null | wc -l) dev $(ls -d $O/devspeech/*/asr_speech.jsonl 2>/dev/null | wc -l)" \
  "workers $(ps -eo args | grep -c '[t]alker_worker.py talker') asr $(ps -eo args | grep -c '[t]alker_worker.py asr')" \
  "failed $(ls $O/claims | grep -c '^failed_') FAILED-lines $(cat logs/qabc_v10/talker_worker_c*.log logs/qabc_v10/asr_worker_c*.log | grep -c FAILED)"
echo "mem $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '{printf "%s:%d ", $1, $2/1000}')"
since=$(date -u -d '-15 min' '+%F %T')
awk -v s="[$since" '$0 >= s' logs/gpu_coord/events.log | grep "MIST(v6) -> Q\|MIST(v6) watchdog" | tail -n 3
grep "HOT card" logs/qabc_v10/mem_hot.log | awk -v s="[$since" '$0 >= s' | tail -n 2
