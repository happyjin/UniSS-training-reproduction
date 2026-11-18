#!/usr/bin/env bash
# OpenBroadcast-SI v1: Nemotron-3-Diarization lowlat posteriors (dump_nemotron_posteriors.py, venvs/nemo_diar, preset
# lowlat -> <data>/diar/nemotron_lowlat/<m>.{post.pt,rttm}) for the given benchmark sets, sequentially on ONE card.
# The script resumes per session and never rewrites one. Marker logs/mist_w3po_v1/voicecond_bench/DIAR_<tag>_{DONE,FAILED}.
#   [DRY_RUN=1] bash scripts/voicecond_bench_diar.sh <card> <tag> <cond>/<dir> ...
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond_bench/diar_<tag>.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
card=$1; tag=$2; shift 2
VL=logs/mist_w3po_v1/voicecond_bench
NP=$J/venvs/nemo_diar/bin/python
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail=0
for s in "$@"; do
  D=data/open_voicecond_v1/bench/$s
  case $s in */zh2en) A=mix;; */en2zh) A=ihm_mix;; *) log "bad set $s"; exit 1;; esac
  [ -e $D/meetings.json ] || { log "missing $D/meetings.json"; fail=1; continue; }
  if [ -n "${DRY_RUN:-}" ]; then log "[dry] card $card: $NP experiments/uniss_qwen3_omni_v1/data_clean/dump_nemotron_posteriors.py --preset lowlat --data $D --audio $A --out-name nemotron_lowlat"; continue; fi
  log "start $s on card $card"
  CUDA_VISIBLE_DEVICES=$card HF_HOME=$J/cache/huggingface TMPDIR=$J/work/mist_w3po_v1/tmp $NP \
    experiments/uniss_qwen3_omni_v1/data_clean/dump_nemotron_posteriors.py --preset lowlat --data $D --audio $A \
    --out-name nemotron_lowlat || { log "FAIL $s"; fail=1; continue; }
  n=$(ls $D/diar/nemotron_lowlat/*.post.pt 2>/dev/null | wc -l); m=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" $D/meetings.json)
  [ "$n" = "$m" ] && log "done $s ($n sessions)" || { log "FAIL $s: $n of $m posteriors"; fail=1; }
done
[ -n "${DRY_RUN:-}" ] && exit 0
if [ $fail = 0 ]; then date -u +%FT%TZ > $VL/DIAR_${tag}_DONE; log "DIAR_${tag}_DONE"; else date -u +%FT%TZ > $VL/DIAR_${tag}_FAILED; log "DIAR_${tag}_FAILED"; fi
