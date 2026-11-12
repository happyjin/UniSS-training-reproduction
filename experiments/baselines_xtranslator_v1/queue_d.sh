#!/usr/bin/env bash
# Card 6 continuation: after queue_b (xt-a1.6, chunk-1.6 ASR server) exits, swap the ASR server on port 18011 for a
# released-default one (chunk 0.6, same KV cap), then run xt-c5120:zh2en with 1 shard on the card-6 pair.
set -uo pipefail
R=/opt/dlami/nvme/jasonleeeli/projects/UniSS
EXP=$R/experiments/baselines_xtranslator_v1
S=$R/eval_outputs/baselines_xtranslator_v1/servers
: "${QB_PID:?}"
while kill -0 "$QB_PID" 2>/dev/null; do sleep 10; done
grep -q QUEUE_DONE "$R/eval_outputs/baselines_xtranslator_v1/logs/queue_b.log" || { echo "queue_b did not finish"; exit 1; }
p=$(cat "$S/c6a/asr_18011.pid")
c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
case "$c" in *serve_qwen3asr.py*"--port 18011"*"--chunk-size-sec 1.6"*) kill -TERM -- "-$p" ;; *) echo "unexpected ASR pid $p: $c"; exit 1 ;; esac
for i in $(seq 1 60); do ps -o pid= -g "$p" | grep -q . || break; sleep 1; done
ps -o pid= -g "$p" | grep -q . && { echo "old ASR group $p still alive"; exit 1; }
echo "$(date -u +%FT%TZ) stopped chunk-1.6 ASR group $p"
TAG=c6b CARD=6 MT=0 NASR=1 ASR_BASE=18011 ASR_CHUNK=0.6 ASR_KV=1536M NTTS=0 bash "$EXP/start_servers.sh"
for i in $(seq 1 60); do grep -q "Uvicorn running" "$S/c6b/asr_18011.log" 2>/dev/null && break; grep -q -E "Traceback|initialization failed" "$S/c6b/asr_18011.log" 2>/dev/null && { echo "ASR restart failed"; exit 1; }; sleep 5; done
grep -q "Uvicorn running" "$S/c6b/asr_18011.log" || { echo "ASR restart timeout"; exit 1; }
echo "$(date -u +%FT%TZ) chunk-0.6 ASR up on 18011"
QUEUE="xt-c5120:zh2en" CARD=6 NSH=1 ASR_BASE=18011 TTS_BASE=18211 MT_PORT=18100 bash "$EXP/run_queue2.sh"
