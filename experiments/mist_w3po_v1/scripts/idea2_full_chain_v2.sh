#!/usr/bin/env bash
# v2 (2026-11-19 01:50 Prague): identical CPU part; the GPU part (refs on card 4, idea2_gpu_pre.sh) first waits for
# GPU_GATE (default logs/mist_w3po_v1/voicecond/idea2/GAME_TALKER_DONE, written when the esports Talker runs on cards
# 4-7 have finished or failed) so the preprocessing does not compete with vLLM; then also v~ features (feats2).
# Idea 2: complete the 100 h open synthetic data (M1 steps 2-3) after G2.0 failed on the pilot, so that any follow-up
# (contraction-aware model / learned version E2.3) has the data ready. No fitting, no E2.2, no test-set runs.
#   1. wait for LibriSpeech train-clean-360 to be unpacked (idea2_extract_sources_v2.sh) and index it
#   2. build the remaining 720 streams (--set rest) and their clean reference prompts (CPU)
#   3. u* for the new speakers on card 4 (its placeholder tree SIGTERMed by PID first)
#   4. idea2_gpu_pre.sh WHICH=all LS=2 on cards 4-7 (diarizer, HDemucs, window encodes, features)
#   5. the pre-registered E2.0 re-check on the full train/dev data with L, M kind and r fixed from the pilot (CPU)
# Marker data/open_voicecond_v1/idea2/FULL_CHAIN_DONE (or FULL_CHAIN_FAILED_<step>).
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea2_full_chain_v2.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond/idea2/full_chain_v2.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
I2=data/open_voicecond_v1/idea2
V=/opt/dlami/nvme/jasonleeeli/venvs
PY=$V/vllm_omni_030/bin/python
export HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface
export PATH=$V/vllm_omni_030/bin:$PATH
E20P=eval_outputs/mist_w3po_v1/voicecond_idea2/e20_pilot
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED at $1"; [ -n "${DRY_RUN:-}" ] || date -u +%FT%TZ >> $I2/FULL_CHAIN_FAILED_$1; exit 1; }
run() { if [ -n "${DRY_RUN:-}" ]; then log "[dry] $*"; else log "run: $*"; eval "$@"; fi; }

log "start"
if [ -z "${DRY_RUN:-}" ]; then
  until [ -e $I2/src/ls_train-clean-360.UNPACKED ]; do
    [ -e $I2/src/ls_train-clean-360.FAILED_v2 ] && fail unpack360
    grep -q 'FAIL.*train-clean-360' logs/mist_w3po_v1/voicecond/download.log && fail download360
    sleep 30
  done
fi
[ -e $I2/index/speakers_ls_train-clean-360.jsonl ] || run "$PY -I experiments/mist_w3po_v1/voicecond/idea2_index_sources.py speakers ls_train-clean-360" || fail index360
run "$PY -I experiments/mist_w3po_v1/voicecond/idea2_build_streams.py streams --set rest --workers 48" || fail build
[ -n "${DRY_RUN:-}" ] || [ -e $I2/streams/BUILD_REST_DONE ] || fail build_marker
run "$PY -I experiments/mist_w3po_v1/voicecond/idea2_build_streams.py refs --workers 48" || fail refs_cpu
GPU_GATE=${GPU_GATE:-logs/mist_w3po_v1/voicecond/idea2/GAME_TALKER_DONE}
if [ -z "${DRY_RUN:-}" ]; then
  log "waiting for $GPU_GATE before the GPU part"
  until [ -e $GPU_GATE ]; do sleep 30; done
fi
sup=$(ps -eo pid,ppid,args | awk '$2 == 1 && $0 ~ /train_mist_workload\.py/ && $0 ~ /--devices 4 / {print $1}')
for s in $sup; do
  kids=$(ps -eo pid,ppid | awk -v s="$s" '$2 == s {print $1}')
  log "card 4: SIGTERM placeholder supervisor $s children: $kids"
  [ -n "${DRY_RUN:-}" ] || kill -TERM $kids $s 2>/dev/null
done
run "CUDA_VISIBLE_DEVICES=4 $PY -I experiments/mist_w3po_v1/voicecond/idea2_windows.py refs --device cuda:0" || fail refs_gpu
run "WHICH=all LS=2 bash experiments/mist_w3po_v1/scripts/idea2_gpu_pre.sh" || fail pre
[ -n "${DRY_RUN:-}" ] || [ -e $I2/pre/PRE_ALL_DONE ] || fail pre_marker
run "$PY -I experiments/mist_w3po_v1/voicecond/idea2_feats2.py --which all" || fail feats2
run "$PY -I experiments/mist_w3po_v1/voicecond/idea2_e20.py --which all --L 2 --fixed $E20P" || fail e20_all
[ -n "${DRY_RUN:-}" ] && exit 0
date -u +%FT%TZ >> $I2/FULL_CHAIN_DONE
log FULL_CHAIN_DONE
