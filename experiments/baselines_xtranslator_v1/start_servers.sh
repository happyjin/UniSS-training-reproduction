#!/usr/bin/env bash
# Start X-Translator's backend services on one card (resource flags only; decoding settings are the released ones).
#   TAG=c3a CARD=3 MT=1 MT_PORT=18100 NASR=4 ASR_BASE=18001 NTTS=4 TTS_BASE=18201 [ASR_CHUNK=0.6] bash start_servers.sh
# vLLM engines start >= 60 s apart (memory profiling) with fixed absolute KV caps (--kv-cache-memory-bytes). Each service: own process group (setsid), PID in <dir>/*.pid.
set -uo pipefail
: "${TAG:?}" "${CARD:?}"
R=/opt/dlami/nvme/jasonleeeli/projects/UniSS
V=/opt/dlami/nvme/jasonleeeli/venvs/xtranslator
M=/opt/dlami/nvme/jasonleeeli/models/xtranslator
E=$V/opt/espeak-ng
XV=/opt/dlami/nvme/jasonleeeli/research_sources/baselines_20261112/X-Voice
EXP=$R/experiments/baselines_xtranslator_v1
D=$R/eval_outputs/baselines_xtranslator_v1/servers/$TAG
if [ -e "$D" ]; then echo "refusing: $D exists"; exit 1; fi
mkdir -p "$D"
export PATH=$V/bin:$PATH CUDA_VISIBLE_DEVICES=$CARD
export HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface MODELSCOPE_CACHE=/opt/dlami/nvme/jasonleeeli/cache/modelscope
export SSL_CERT_FILE=$($V/bin/python -m certifi)
export PHONEMIZER_ESPEAK_LIBRARY=$E/lib/libespeak-ng.so PHONEMIZER_ESPEAK_DATA_PATH=$E/share/espeak-ng-data
echo "$(date -u +%FT%TZ) start TAG=$TAG CARD=$CARD MT=${MT:-0} NASR=${NASR:-0} NTTS=${NTTS:-0} ASR_CHUNK=${ASR_CHUNK:-0.6}" >> "$D/start.log"
if [ "${MT:-0}" = 1 ]; then
  setsid nohup "$V/bin/python" -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port "$MT_PORT" \
    --trust-remote-code --model "$M/LMT-60-8B" --tensor-parallel-size 1 --dtype bfloat16 --served-model-name lmt \
    --gpu-memory-utilization "${MT_MEM:-0.30}" --kv-cache-memory-bytes "${MT_KV:-3G}" --max-model-len 1024 --max-num-seqs 128 \
    > "$D/lmt_$MT_PORT.log" 2>&1 < /dev/null &
  echo $! > "$D/lmt_$MT_PORT.pid"
  sleep 60
fi
for i in $(seq 0 $(( ${NASR:-0} - 1 ))); do
  p=$((ASR_BASE + i))
  setsid nohup "$V/bin/python" "$EXP/serve_qwen3asr.py" --model "$M/Qwen3-ASR-1.7B" \
    --gpu-memory-utilization "${ASR_MEM:-0.12}" --kv-cache-memory-bytes "${ASR_KV:-1536M}" --port "$p" --chunk-size-sec "${ASR_CHUNK:-0.6}" \
    > "$D/asr_$p.log" 2>&1 < /dev/null &
  echo $! > "$D/asr_$p.pid"
  sleep 60
done
for i in $(seq 0 $(( ${NTTS:-0} - 1 ))); do
  p=$((TTS_BASE + i))
  env -C "$XV" XVOICE_ROOT="$XV" setsid nohup "$V/bin/python" "$EXP/serve_xvoice.py" --host 127.0.0.1 --port "$p" \
    --stage 2 --no-auto-detect-lang --ckpt-file "$M/X-Voice/XVoice_Base_Stage2/model_70000.safetensors" \
    --srp-ckpt-file "$M/X-Voice/SpeedPredictor/model_28000.safetensors" --device cuda \
    > "$D/xvoice_$p.log" 2>&1 < /dev/null &
  echo $! > "$D/xvoice_$p.pid"
  sleep 20
done
echo "$(date -u +%FT%TZ) SERVERS_LAUNCHED" >> "$D/start.log"
