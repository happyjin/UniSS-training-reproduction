#!/usr/bin/env bash
# Idea 1 M3 (reports/mist_w3po_v1/voicecond_idea1_prereg.zh-CN.md v4.2): Talker-only runs of the frozen encoders Z / C on
# OpenBroadcast-SI v1 through the benchmark hook (scripts/voicecond_bench_orch.py --voice-override <sidecar>, B's text via
# --speech-from obsi_b_<cond>_<dir>, --cap135 --exclude 4 5 6 7), order id2, ood1, clean x zh2en, en2zh x Z, C, one
# orchestrator start per 60 s; then bench_eval (CPU) of each run as soon as its orchestrator is DONE.
# Run names obsi_i1{z,c}_<cond>_<dir>__t1c (+ _tts2); orchestrator dirs logs/mist_w3po_v1/voicecond_idea1/m3/<name>.
# Marker logs/mist_w3po_v1/voicecond_idea1/m3/M3_RUNS_DONE. Refuses existing run dirs (the orchestrator refuses too).
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea1_m3_run.sh     (DRY_RUN: the orchestrators' own --dry listing)
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond_idea1/m3_run.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
export PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH
B2=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/thinker_merged
L=logs/mist_w3po_v1/voicecond_idea1/m3
M3=eval_outputs/mist_w3po_v1/voicecond_idea1/m3
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
mkdir -p $L
if [ -z "${DRY_RUN:-}" ]; then
  until [ -e $M3/tokens.DONE ]; do sleep 20; done
  grep -q rc=0 $M3/tokens.DONE || { log "tokens failed"; exit 1; }
fi
RUNS=""
for cond in id2 ood1 clean; do for d in zh2en en2zh; do for m in z c; do
  N=obsi_i1${m}_${cond}_${d}__t1c; R=$L/$N; S=$M3/sidecar_${m}_${cond}_${d}.json
  RUNS="$RUNS $cond:$d:$N"
  if [ -n "${DRY_RUN:-}" ]; then
    $PY scripts/voicecond_bench_orch.py --cond $cond --dir $d --voice-override $S --base $B2 --run $R.dry --name $N --ckpt - \
      --speech-from obsi_b_${cond}_${d} --talker t1c --window 1 --cap135 --exclude 4 5 6 7 --dry 2>&1 | grep -c 'bench_talker_override --voice-override' | sed "s/^/$N talk jobs with override: /"
    continue
  fi
  [ -e $R ] && { log "$R exists (not relaunched)"; continue; }
  setsid nohup $PY scripts/voicecond_bench_orch.py --cond $cond --dir $d --voice-override $S --base $B2 --run $R --name $N --ckpt - \
    --speech-from obsi_b_${cond}_${d} --talker t1c --window 1 --cap135 --exclude 4 5 6 7 < /dev/null > $R.launch.log 2>&1 9>&- &
  log "launched $N"
  sleep 60
done; done; done
[ -n "${DRY_RUN:-}" ] && exit 0
for x in $RUNS; do
  IFS=: read cond d N <<< "$x"
  until [ -e $L/$N/events/DONE ]; do sleep 30; done
  tail -n 3 $L/$N/orch.log | grep -q "DONE; failed/skipped \[\]" || log "WARN $N orchestrator reports failures: $(tail -n 1 $L/$N/orch.log | cut -c1-200)"
  if [ ! -e eval_outputs/mist_w3po_v1/voicecond_bench/$N/metrics.json ]; then
    PYTHONPATH=. $PY -m experiments.mist_w3po_v1.voicecond.bench_eval --cond $cond --dir $d --name $N > $L/eval_$N.log 2>&1 \
      && log "evaluated $N" || log "FAIL bench_eval $N"
  fi
done
date > $L/M3_RUNS_DONE
log "M3 runs done"
