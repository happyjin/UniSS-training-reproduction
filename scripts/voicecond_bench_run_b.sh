#!/usr/bin/env bash
# OpenBroadcast-SI v1 (reports/mist_w3po_v1/voicecond_bench.zh-CN.md): baseline B -- exactly game_run.sh's B
# (joint_v8h6k merged s06000 + identity; text decode window 1, --cap135; then the frozen T1c talker, two TTS seeds) --
# through scripts/voicecond_bench_orch.py on every condition (id2, ood1, clean) x direction (zh2en, en2zh), on cards
# 0-3 only (--exclude 4 5 6 7). Per chain: text orchestrator -> T1c orchestrator -> bench_eval (all metrics, both
# seeds). Run dirs logs/mist_w3po_v1/voicecond_bench/<cond>_<dir>{,_t1c}; names obsi_b_<cond>_<dir>[__t1c[_tts2]];
# markers logs/mist_w3po_v1/voicecond_bench/VB_<cond>_<dir>_{DONE,FAILED}. Refuses to overwrite a run dir.
#   [DRY_RUN=1] [CHAINS="id2:zh2en ..."] bash scripts/voicecond_bench_run_b.sh
# Run detached: setsid nohup bash scripts/voicecond_bench_run_b.sh > logs/mist_w3po_v1/voicecond_bench/run_b.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
export PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH
VL=logs/mist_w3po_v1/voicecond_bench
VO="$PY scripts/voicecond_bench_orch.py"
B2=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/thinker_merged
IDN=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/identity
CHAINS=${CHAINS:-"id2:zh2en id2:en2zh ood1:zh2en ood1:en2zh clean:zh2en clean:en2zh"}
mkdir -p $VL
run_orch() {   # <run dir> <command...>
  local R=$1; shift
  if [ -n "${DRY_RUN:-}" ]; then echo "$(basename $R): $*"; "$@" --dry | tail -1; return 0; fi
  [ -e $R ] && { log "$R exists (not overwritten)"; return 1; }
  setsid nohup "$@" < /dev/null > $R.launch.log 2>&1 9>&- &
  log "$(basename $R) orchestrator pid $!"
  wait_done $R; [ -z "$(orch_failed $R)" ] || { log "$(basename $R) failed ($(tail -1 $R/orch.log | cut -c1-200))"; return 1; }
}
chain() {   # <cond> <dir>
  local c=$1 d=$2 n=obsi_b_$1_$2
  if run_orch $VL/${c}_$d $VO --cond $c --dir $d --base $B2 --run $VL/${c}_$d --name $n --ckpt $IDN --talker none \
       --window 1 --cap135 --exclude 4 5 6 7 \
     && run_orch $VL/${c}_${d}_t1c $VO --cond $c --dir $d --base $B2 --run $VL/${c}_${d}_t1c --name ${n}__t1c --ckpt - \
          --speech-from $n --talker t1c --window 1 --cap135 --exclude 4 5 6 7; then
    [ -n "${DRY_RUN:-}" ] && return 0
    $PY -m experiments.mist_w3po_v1.voicecond.bench_eval --cond $c --dir $d --name ${n}__t1c > $VL/eval_${c}_$d.log 2>&1 \
      && { date -u +%FT%TZ > $VL/VB_${c}_${d}_DONE; log "VB_${c}_${d}_DONE"; } \
      || { date -u +%FT%TZ > $VL/VB_${c}_${d}_FAILED; log "VB_${c}_${d}_FAILED (eval)"; }
  else
    [ -n "${DRY_RUN:-}" ] || { date -u +%FT%TZ > $VL/VB_${c}_${d}_FAILED; log "VB_${c}_${d}_FAILED"; }
  fi
}
for x in $CHAINS; do
  c=${x%%:*}; d=${x##*:}
  for f in data/open_voicecond_v1/bench/$c/$d/refs_windows.jsonl \
           $DD/windows_obsi_${c}_${d}_c2g06_v1/windows_test_online_$d.jsonl \
           $DD/windows_obsi_${c}_${d}_c2g06_chunk_v1/windows_dev_online.jsonl $B2/sd_proj.pt $IDN; do
    [ -e $f ] || { log "missing $f"; exit 1; }
  done
done
if [ -n "${DRY_RUN:-}" ]; then for x in $CHAINS; do chain ${x%%:*} ${x##*:}; done; exit 0; fi
P=""
for x in $CHAINS; do
  chain ${x%%:*} ${x##*:} &
  P="$P $!"
  sleep 120
done
for p in $P; do wait $p; done
log "VB_RUN_EXIT"
