#!/usr/bin/env bash
# Idea 2 prereg v3 §B: frozen shrink+Vfb on OpenBroadcast-SI v1 (ID-2 / OOD-1 / clean, both directions, both TTS seeds)
# through the benchmark's documented hook (reports/mist_w3po_v1/voicecond_bench.zh-CN.md §4-5). Two workers (one per
# direction); per (cond, dir): wait for the benchmark agent's baseline marker VB_<cond>_<dir>_DONE, build the token table
# + sidecar (idea2_stream_tokens.py, full-train fit, card 4 stacked), run scripts/voicecond_bench_orch.py
# --voice-override <sidecar> (--speech-from obsi_b_<cond>_<dir>, --exclude 0 1 2 3, run obsi_i2_<cond>_<dir>__t1c), then
# the single eval command bench_eval. Markers logs/mist_w3po_v1/voicecond/idea2/bench/I2_<cond>_<dir>_{DONE,FAILED}.
#   [ORDER_EN="clean id2 ood1"] [ORDER_ZH="ood1 id2 clean"] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea2_bench_run.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond/idea2/bench_run.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
export PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface
VPY=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
B2=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/thinker_merged
FIT=eval_outputs/mist_w3po_v1/voicecond_idea2/shrink_fit_all
TOK=eval_outputs/mist_w3po_v1/voicecond_idea2/tokens
VB=logs/mist_w3po_v1/voicecond_bench
GL=logs/mist_w3po_v1/voicecond/idea2/bench
ORDER_EN=${ORDER_EN:-"clean id2 ood1"}
ORDER_ZH=${ORDER_ZH:-"ood1 id2 clean"}
mkdir -p $GL

one() {   # <cond> <dir>
  local c=$1 d=$2 N=obsi_i2_$1_$2__t1c R
  R=$GL/$N
  [ -e $GL/I2_${c}_${d}_DONE ] && { log "$N done already"; return 0; }
  if [ -z "${DRY_RUN:-}" ]; then
    until [ -e $VB/VB_${c}_${d}_DONE ]; do sleep 60; done
  fi
  if [ ! -e $TOK/obsi_${c}_${d}.sidecar.json ]; then
    if [ -n "${DRY_RUN:-}" ]; then log "[dry] tokens obsi_${c}_${d}"
    else CUDA_VISIBLE_DEVICES=4 $VPY -I experiments/mist_w3po_v1/voicecond/idea2_stream_tokens.py \
        --data data/open_voicecond_v1/bench/$c/$d --dir $d --run obsi_b_${c}_${d}__t1c \
        --windows eval_outputs/mist_w3po_v1/data/windows_obsi_${c}_${d}_c2g06_v1/windows_test_online_${d}.jsonl \
        --fit $FIT --tag obsi_${c}_${d} > $GL/tokens_${c}_${d}.log 2>&1 || { log "tokens failed $c $d"; return 1; }
    fi
  fi
  local CMD="$PY scripts/voicecond_bench_orch.py --cond $c --dir $d --voice-override $TOK/obsi_${c}_${d}.sidecar.json \
    --base $B2 --run $R --name $N --ckpt - --speech-from obsi_b_${c}_${d} --talker t1c --window 1 --cap135 --exclude 0 1 2 3"
  if [ -n "${DRY_RUN:-}" ]; then log "[dry] $CMD"; return 0; fi
  if [ ! -e $R ]; then
    setsid nohup $CMD < /dev/null > $R.launch.log 2>&1 9>&- &
    log "$N orchestrator pid $!"
  fi
  wait_done $R; [ -z "$(orch_failed $R)" ] || { log "$N failed"; date -u +%FT%TZ >> $GL/I2_${c}_${d}_FAILED; return 1; }
  PYTHONPATH=. $PY -m experiments.mist_w3po_v1.voicecond.bench_eval --cond $c --dir $d --name $N > $GL/eval_${c}_${d}.log 2>&1 \
    || { log "bench_eval failed $N"; date -u +%FT%TZ >> $GL/I2_${c}_${d}_FAILED; return 1; }
  date -u +%FT%TZ >> $GL/I2_${c}_${d}_DONE
  log "$N DONE"
}
worker() {   # <dir> <conds...>
  local d=$1; shift
  for c in "$@"; do one $c $d || true; done
}
worker en2zh $ORDER_EN &
sleep 90
worker zh2en $ORDER_ZH &
wait
log BENCH_RUN_EXIT
