#!/usr/bin/env bash
# Paper v7 Table 1 (offline S2ST on CVSS-T, SimulS2ST-Omni's columns), cards 0-3 only (4-7 belong to other work):
#   1. the Talker (T1c@1500) speaks the s0 adapter's whole-sentence CVSS-T text (cvss_off_s0, b0 = 0), both directions,
#      exactly as cvss_base was made in s2_jobs.json, then speech_only + ASR (paraformer-zh / whisper-large-v3);
#   2. A.PCP and SIM-O (base-plus) for cvss_base and cvss_s0, both directions (cvss_table1_metrics.py);
#   3. the GPU holder back on cards 0-3.
# Log: logs/si_v4/cvss_table1.log
set -u
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
LOG=logs/si_v4/cvss_table1.log
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
V4=eval_outputs/uniss_qwen3_omni_v1/si_v4
TALKER=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
VPY=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
EPY=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
VENV="PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=."

log "start; holder off on 0-3"
bash scripts/gpu_queue/holder_off.sh 0,1,2,3 >> $LOG 2>&1
for i in $(seq 1 60); do
  busy=0; for c in 0 1 2 3; do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c)" -lt 2000 ] || busy=1; done
  [ $busy = 0 ] && break; sleep 5
done
[ $busy = 0 ] || { log "cards 0-3 not free; abort"; exit 1; }

speech() {   # direction gpu0 gpu1
  local d=$1 g0=$2 g1=$3 out=$V4/speech/cvss_s0/cvss_$1
  for s in 0 1; do
    local g=$g0; [ $s = 1 ] && g=$g1
    env CUDA_VISIBLE_DEVICES=$g PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 \
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. \
      $VPY -m experiments.uniss_qwen3_omni_v1.evaluation.si_infer --model $TALKER --direction $d --window 60 --output $out \
      --text-policy file --blocks-file $V4/cvss_off_s0/blocks_${d}_w60_sws0b0.jsonl --items-file data/cvss_t/si_items_v1/items_$d.jsonl \
      --speed-policy block --history-ban 8 --gpu-mem 0.25 --shard $s --num-shards 2 > $V4/speech/cvss_s0/cvss_${d}_shard$s.log 2>&1 &
    [ $s = 0 ] && sleep 60
  done
  wait
  [ "$(ls $out/part_*.jsonl 2>/dev/null | wc -l)" -eq 2 ] || { log "$d: speech shards missing"; return 1; }
  $EPY reports/uniss_si_v4/speech_only.py $out > /dev/null || { log "$d: speech_only failed"; return 1; }
  CUDA_VISIBLE_DEVICES=$g0 PYTHONPATH=. $EPY -m evaluation.asr_transcribe --input $out/results_speechonly.jsonl \
    --output $out/asr_speech.jsonl --device cuda:0 --batch-size 16 >> $V4/speech/cvss_s0/cvss_${d}_asr.log 2>&1 \
    || { log "$d: ASR failed"; return 1; }
  log "$d: speech + ASR done"
}

if [ -e $V4/speech/cvss_s0 ]; then log "$V4/speech/cvss_s0 exists; not overwriting"; else
  mkdir -p $V4/speech/cvss_s0
  log "speech en2zh on 0,1; zh2en on 2,3 (120 s later)"
  speech en2zh 0 1 & p1=$!
  sleep 120
  speech zh2en 2 3 & p2=$!
  wait $p1; wait $p2
fi

log "A.PCP + SIM-O"
g=0
for m in base s0; do for d in en2zh zh2en; do
  run=$V4/speech/cvss_$m/cvss_$d
  if [ -f $run/asr_speech.jsonl ]; then
    PYTHONPATH=. $EPY reports/uniss_si_v4/cvss_table1_metrics.py $run $V4/cvss_table1/${m}_$d $g >> $V4/cvss_table1_${m}_$d.log 2>&1 &
    g=$((g+1))
  else log "missing $run/asr_speech.jsonl"; fi
done; done
wait
mkdir -p $V4/cvss_table1
log "metrics: $(grep -l CVSS_TABLE1_METRICS_DONE $V4/cvss_table1_*.log 2>/dev/null | wc -l)/4 done"

bash scripts/gpu_queue/holder_on.sh 0,1,2,3 >> $LOG 2>&1
log "CVSS_TABLE1_DONE"
