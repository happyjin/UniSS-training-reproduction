#!/usr/bin/env bash
# B6: every 60 s, the compute processes on each card (ours = gstar decoders), until the memq_4 launcher is done (CPU only)
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
O=eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/b6/cotenancy.jsonl
declare -A IDX; while IFS=, read i u; do IDX[${u// /}]=${i// /}; done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
until grep -q MEMQ_DONE logs/multispk/v8/memq_4.log; do
  ts=$(date -u +%FT%TZ)
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits | while IFS=, read u p m; do
    u=${u// /}; p=${p// /}; full=$(ps -o args= -p $p 2>/dev/null); a=$(echo "$full" | cut -c1-100 | tr -d '"\\')
    case "$full" in *gstar_si_stream_online*) ours=true;; *) ours=false;; esac
    echo "{\"t\":\"$ts\",\"gpu\":${IDX[$u]:-null},\"pid\":$p,\"mib\":${m// /},\"ours\":$ours,\"cmd\":\"$a\"}"
  done >> $O
  sleep 60
done
