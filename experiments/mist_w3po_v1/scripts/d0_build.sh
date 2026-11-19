#!/usr/bin/env bash
# D0 (reports/mist_w3po_v1/robust_wave1_prereg.zh-CN.md): build the OpenBroadcast-SI DEV twin with
# experiments/mist_w3po_v1/voicecond/bench_dev_build.py (CoVoST 2 validation speech, non-held-out backgrounds; id2 + clean
# x zh2en + en2zh; meeting ids db[ze] / dc[ze]; root data/open_voicecond_v1/bench_dev/):
#   1. src zh2en | src en2zh | bgprep (in parallel, CPU)   2. plan (both)   3. mix clean, mix bg (both)
#   4. Nemotron lowlat posteriors (venvs/nemo_diar, dump_nemotron_posteriors.py --preset lowlat, the four sets in parallel
#      on ONE card picked by d0_gpu_lib.d0_gpu_one; a placeholder there is stopped by PID only if needed)
#   5. windows (C2 g0.6 + chunk protocol) for the four sets -> eval_outputs/mist_w3po_v1/data/windows_obsidev_*
#   6. check (leak check) + summary
# Every step refuses to overwrite its outputs; a step whose output exists is skipped. Logs + markers in
# logs/mist_w3po_v1/wave1/d0/ (D0_BUILD_DONE | D0_BUILD_FAILED).
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/d0_build.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/wave1/d0/build.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
. experiments/mist_w3po_v1/scripts/d0_gpu_lib.sh
J=/opt/dlami/nvme/jasonleeeli
L=logs/mist_w3po_v1/wave1/d0
BDV=data/open_voicecond_v1/bench_dev
NP=$J/venvs/nemo_diar/bin/python
SETS="id2:zh2en id2:en2zh clean:zh2en clean:en2zh"
mkdir -p $L
fail() { d0_log "FAIL $*"; date > $L/D0_BUILD_FAILED; exit 1; }
step() {   # <tag> <skip-if-exists path> <bench_dev_build args...>
  local t=$1 out=$2
  shift 2
  if [ -n "${DRY_RUN:-}" ]; then d0_log "[dry] $t: bench_dev_build $* (skip if $out exists: $([ -e $out ] && echo yes || echo no))"; return 0; fi
  [ -e $out ] && { d0_log "$t: $out exists, skipped"; return 0; }
  PYTHONPATH=. OMP_NUM_THREADS=4 $PY -m experiments.mist_w3po_v1.voicecond.bench_dev_build "$@" > $L/build_$t.log 2>&1 \
    || { d0_log "$t failed: $(tail -2 $L/build_$t.log | tr '\n' ' ' | cut -c1-300)"; return 1; }
  d0_log "$t: $(tail -1 $L/build_$t.log | cut -c1-300)"
}
[ -e $L/D0_BUILD_DONE ] && { d0_log "D0_BUILD_DONE exists"; exit 0; }
d0_log "D0 build start"
step src_zh2en $BDV/src/zh2en/pool.jsonl src --dir zh2en & p1=$!
step src_en2zh $BDV/src/en2zh/pool.jsonl src --dir en2zh & p2=$!
step bgprep $BDV/bgsrc/pools.json bgprep & p3=$!
wait $p1 || fail src_zh2en; wait $p2 || fail src_en2zh; wait $p3 || fail bgprep
for d in zh2en en2zh; do step plan_$d $BDV/src/$d/plan.json plan --dir $d || fail plan_$d; done
for d in zh2en en2zh; do
  step mixclean_$d $BDV/clean/$d/meetings.json mix --dir $d --phase clean || fail mixclean_$d
  step mixbg_$d $BDV/id2/$d/meetings.json mix --dir $d --phase bg || fail mixbg_$d
done
complete() {   # <data dir>: one post.pt per meeting
  local n m
  m=$($PY -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" $1/meetings.json)
  n=$(ls $1/diar/nemotron_lowlat/*.post.pt 2>/dev/null | wc -l)
  [ "$n" = "$m" ]
}
diar_all() {   # the four sets in parallel on the current CUDA_VISIBLE_DEVICES
  local x c d A pids="" rc=0
  for x in $SETS; do
    c=${x%%:*}; d=${x##*:}; A=$([ $d = zh2en ] && echo mix || echo ihm_mix)
    complete $BDV/$c/$d && continue
    HF_HOME=$J/cache/huggingface TMPDIR=$J/work/mist_w3po_v1/tmp $NP experiments/uniss_qwen3_omni_v1/data_clean/dump_nemotron_posteriors.py \
      --preset lowlat --data $BDV/$c/$d --audio $A --out-name nemotron_lowlat > $L/diar_${c}_$d.log 2>&1 &
    pids="$pids $!"
  done
  for x in $pids; do wait $x || rc=1; done
  return $rc
}
if [ -n "${DRY_RUN:-}" ]; then
  d0_log "[dry] diar: d0_gpu_one 30000 $L/diar.DONE diar_all ($SETS; $NP dump_nemotron_posteriors.py --preset lowlat)"
  d0_log "[dry] pick now: $(d0_pick_card 30000)"
else
  all=1; for x in $SETS; do complete $BDV/${x%%:*}/${x##*:} || all=0; done
  if [ $all = 0 ]; then
    [ -e $L/diar.DONE ] && fail "diar.DONE exists but posteriors incomplete ($(cat $L/diar.DONE))"
    d0_gpu_one 30000 $L/diar.DONE diar_all || fail "diar ($(cat $L/diar.DONE 2>/dev/null))"
  fi
  for x in $SETS; do complete $BDV/${x%%:*}/${x##*:} || fail "posteriors incomplete for $x"; done
  d0_log "posteriors complete"
fi
for x in $SETS; do
  c=${x%%:*}; d=${x##*:}
  step win_${c}_$d $DD/windows_obsidev_${c}_${d}_c2g06_chunk_v1/windows_dev_online.jsonl windows --cond $c --dir $d || fail win_${c}_$d
done
step check $BDV/meta/leak_check.json check || fail check
step summary $BDV/summary.json summary || fail summary
[ -n "${DRY_RUN:-}" ] && exit 0
date > $L/D0_BUILD_DONE
d0_log "D0_BUILD_DONE"
