#!/usr/bin/env bash
# Robust wave 2, diagnosis S2 (reports/mist_w3po_v1/robust_wave2_diag_prereg.zh-CN.md): baseline B on id2-dev with
# REFERENCE-turn diarizer posteriors, everything else B's (d0_runs.sh chain_b: joint_v8h6k merged s06000 + identity,
# window 1, --talker none, --cap135 -> T1c orchestrator with --speech-from, two TTS seeds -> eval).
#   build  s2_oracle_build post + windows per direction (CPU) -> data/open_voicecond_v1/bench_dev/id2/<d>/diar/oracle_lowlat,
#          eval_outputs/mist_w3po_v1/data/windows_obsidev_id2or_<d>_c2g06{,_chunk}_v1
#   runs   scripts/voicecond_bench_devor_orch.py (dev orchestrator, oracle windows, talker voice prompts from the oracle
#          posteriors): obsidev_bor_<d> (text), obsidev_bor_<d>__t1c{,_tts2}
#   eval   bench_devor_eval (bench_eval UNCHANGED, oracle windows), at most 3 bench_eval at once together with D0 (D0's
#          three slot locks, opened in append mode, fd closed for the child) and only while the 1-min load < 130
#   summary  s2_summary (waits for B's id2 / clean metrics.json of D0, never touches those runs)
# Every orchestrator: SI_KV_CACHE_GIB=7.73 ORCH_TALK_PAIR_GAP_S=0 ORCH_LEDGER_VLLM_GAP_S=10 --exclude none (stacks on any
# card with real free memory, <= 135 GiB per card), started under D0's launch lock (fd 8, append mode; closed for the
# child, 9 closed too), 60 s apart. A placeholder watcher (voicecond/s2_noplaceholder.sh) runs while this script lives
# and is stopped by PID at the end. Refuses existing run dirs; existing outputs are skipped.
# Logs / markers: logs/mist_w3po_v1/wave2/s2/ (S2_<d>_{DONE,FAILED}, S2_RUNS_DONE / S2_RUNS_FAILED).
#   [DRY_RUN=1] [DIRS="zh2en en2zh"] bash experiments/mist_w3po_v1/scripts/s2_runs.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/wave2/s2/runs.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
. experiments/mist_w3po_v1/scripts/d0_gpu_lib.sh
J=/opt/dlami/nvme/jasonleeeli
export PATH=$J/venvs/vllm_omni_030/bin:$PATH HF_HOME=$J/cache/huggingface
export SI_KV_CACHE_GIB=7.73 ORCH_TALK_PAIR_GAP_S=0 ORCH_LEDGER_VLLM_GAP_S=10
L=logs/mist_w3po_v1/wave2/s2
R0=$L/runs
D0L=logs/mist_w3po_v1/wave1/d0
EV=eval_outputs/mist_w3po_v1/voicecond_bench
B2=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/thinker_merged
IDN=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/identity
VO="$PY scripts/voicecond_bench_devor_orch.py"
DIRS=${DIRS:-"zh2en en2zh"}
mkdir -p $R0
build() {   # <dir>
  local d=$1
  if [ -n "${DRY_RUN:-}" ]; then d0_log "[dry] s2_oracle_build post/windows --dir $d"; return 0; fi
  [ -e data/open_voicecond_v1/bench_dev/id2/$d/diar/oracle_lowlat/stats.json ] || \
    PYTHONPATH=. $PY -m experiments.mist_w3po_v1.voicecond.s2_oracle_build post --dir $d > $L/build_post_$d.log 2>&1 \
    || { d0_log "build post $d failed: $(tail -1 $L/build_post_$d.log)"; return 1; }
  [ -e $DD/windows_obsidev_id2or_${d}_c2g06_chunk_v1/windows_dev_online.jsonl ] || \
    PYTHONPATH=. OMP_NUM_THREADS=8 $PY -m experiments.mist_w3po_v1.voicecond.s2_oracle_build windows --dir $d > $L/build_win_$d.log 2>&1 \
    || { d0_log "build windows $d failed: $(tail -1 $L/build_win_$d.log)"; return 1; }
  d0_log "build $d: $(tail -1 $L/build_post_$d.log | cut -c1-200); $(grep -m1 'C2 windows' $L/build_win_$d.log)"
}
orch() {   # <run dir> <orchestrator args...>: start under the launch lock, wait for DONE; rc 0 = no failed job
  local R=$1
  shift
  if [ -n "${DRY_RUN:-}" ]; then
    d0_log "[dry] $(basename $R): $VO $* --run $R"
    $VO "$@" --run $R.dry --dry 2>&1 | tail -3
    return 0
  fi
  if [ ! -e $R ]; then
    (
      flock 8
      [ -e $R ] || { setsid nohup $VO "$@" --run $R < /dev/null > $R.launch.log 2>&1 8>&- 9>&- &
                     echo $! > $R.pid; d0_log "$(basename $R) orchestrator pid $!"; sleep 60; }
    ) 8>> $D0L/launch.lock
  else
    d0_log "$(basename $R) exists: only waited for"
  fi
  until [ -e $R/events/DONE ]; do
    if [ -s $R.pid ] && ! kill -0 $(cat $R.pid) 2>/dev/null; then
      sleep 5; [ -e $R/events/DONE ] && break
      d0_log "$(basename $R) orchestrator exited without DONE: $(tail -2 $R.launch.log | tr '\n' ' ' | cut -c1-300)"
      return 1
    fi
    sleep 30
  done
  [ -z "$(orch_failed $R)" ] || { d0_log "$(basename $R) failed ($(tail -1 $R/orch.log | cut -c1-200))"; return 1; }
}
ev() {   # <dir> <name>: bench_devor_eval, <= 3 bench_eval at once together with D0, load < 130
  local d=$1 n=$2 s fd rc
  [ -e $EV/$n/metrics.json ] && { d0_log "$n metrics.json exists"; return 0; }
  if [ -n "${DRY_RUN:-}" ]; then d0_log "[dry] bench_devor_eval --cond id2 --dir $d --name $n"; return 0; fi
  while :; do
    for s in 0 1 2; do
      exec {fd}>> $D0L/eval_slot_$s.lock
      if flock -n $fd; then
        while [ "$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)" -ge 130 ]; do sleep 60; done
        d0_log "eval $n start (slot $s, load $(cut -d' ' -f1 /proc/loadavg))"
        PYTHONPATH=. $PY -m experiments.mist_w3po_v1.voicecond.bench_devor_eval --cond id2 --dir $d --name $n \
          > $L/eval_$n.log 2>&1 {fd}>&- 8>&- 9>&-
        rc=$?
        flock -u $fd; exec {fd}>&-
        d0_log "eval $n rc=$rc: $(tail -1 $L/eval_$n.log | cut -c1-400)"
        return $rc
      fi
      exec {fd}>&-
    done
    sleep 60
  done
}
chain() {   # <dir>
  local d=$1 n=obsidev_bor_$1
  build $d || { date > $L/S2_${d}_FAILED; return 1; }
  orch $R0/$n --cond id2 --dir $d --base $B2 --name $n --ckpt $IDN --talker none --window 1 --cap135 --exclude none \
    || { date > $L/S2_${d}_FAILED; return 1; }
  orch $R0/${n}__t1c --cond id2 --dir $d --base $B2 --name ${n}__t1c --ckpt - --speech-from $n --talker t1c --window 1 \
    --cap135 --exclude none || { date > $L/S2_${d}_FAILED; return 1; }
  ev $d ${n}__t1c && date > $L/S2_${d}_DONE || { date > $L/S2_${d}_FAILED; return 1; }
}
for d in $DIRS; do
  for f in data/open_voicecond_v1/bench_dev/id2/$d/refs_windows.jsonl data/open_voicecond_v1/bench_dev/id2/$d/meetings.json \
           $B2/sd_proj.pt $IDN; do
    [ -e $f ] || { d0_log "missing $f"; [ -n "${DRY_RUN:-}" ] || exit 1; }
  done
done
if [ -n "${DRY_RUN:-}" ]; then
  for d in $DIRS; do chain $d; done
  exit 0
fi
echo $$ > $L/s2_runs.pid
setsid nohup bash experiments/mist_w3po_v1/voicecond/s2_noplaceholder.sh > $L/noplaceholder.log 2>&1 < /dev/null 8>&- 9>&- &
WP=$!
echo $WP > $L/noplaceholder.pid
d0_log "S2 runs start: $DIRS (placeholder watcher pid $WP)"
P=""
for d in $DIRS; do chain $d & P="$P $!"; sleep 5; done
for p in $P; do wait $p || true; done
ok=1
for d in $DIRS; do [ -e $L/S2_${d}_DONE ] || ok=0; done
if kill -0 $WP 2>/dev/null && tr '\0' ' ' < /proc/$WP/cmdline | grep -q s2_noplaceholder.sh; then
  kill -TERM $WP && d0_log "placeholder watcher $WP stopped"
fi
[ $ok = 1 ] && date > $L/S2_RUNS_DONE || date > $L/S2_RUNS_FAILED
d0_log "S2 runs exit (ok=$ok)"
if [ $ok = 1 ]; then
  PYTHONPATH=. $PY -m experiments.mist_w3po_v1.voicecond.s2_summary --wait > $L/summary.log 2>&1
  d0_log "summary rc=$?: $(tail -1 $L/summary.log | cut -c1-400)"
fi
