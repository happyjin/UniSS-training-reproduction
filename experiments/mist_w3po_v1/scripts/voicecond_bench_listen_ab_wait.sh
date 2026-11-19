#!/usr/bin/env bash
# OpenBroadcast-SI v1 listening A/B: wait until the arms' benchmark runs are over -- six Idea 2 markers
# logs/mist_w3po_v1/voicecond/idea2/bench/I2_*_{DONE,FAILED} and logs/mist_w3po_v1/voicecond_idea1/m3/M3_RUNS_{DONE,FAILED}
# -- then build voicecond_bench_listen_ab.py into demo_packages/voicecond_bench_listen_ab_v1 (refuses an existing
# folder) and write LISTEN_AB_DONE there (or logs/mist_w3po_v1/voicecond_bench/LISTEN_AB_FAILED).
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/voicecond_bench_listen_ab_wait.sh
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond_bench/listen_ab.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
I2=logs/mist_w3po_v1/voicecond/idea2/bench
M3=logs/mist_w3po_v1/voicecond_idea1/m3
OUT=demo_packages/voicecond_bench_listen_ab_v1
VL=logs/mist_w3po_v1/voicecond_bench
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
n_i2() { ls $I2/I2_*_DONE $I2/I2_*_FAILED 2>/dev/null | wc -l; }
m3_over() { [ -e $M3/M3_RUNS_DONE ] || [ -e $M3/M3_RUNS_FAILED ]; }
[ -e $OUT ] && { log "拒绝覆盖已有输出: $OUT"; exit 1; }
if [ -n "${DRY_RUN:-}" ]; then log "[dry] now: I2 markers $(n_i2)/6, M3 over: $(m3_over && echo yes || echo no); then $PY experiments/mist_w3po_v1/scripts/voicecond_bench_listen_ab.py --out $OUT"; exit 0; fi
log "waiting: I2 markers (6) and M3_RUNS_DONE"
until [ "$(n_i2)" -ge 6 ] && m3_over; do sleep 60; done
log "markers: $(ls $I2/I2_* $M3/M3_RUNS_* 2>/dev/null | tr '\n' ' ')"
if OMP_NUM_THREADS=4 $PY experiments/mist_w3po_v1/scripts/voicecond_bench_listen_ab.py --out $OUT; then
  date -u +%FT%TZ > $OUT/LISTEN_AB_DONE; log "LISTEN_AB_DONE $OUT"
else
  date -u +%FT%TZ > $VL/LISTEN_AB_FAILED; log "LISTEN_AB_FAILED"
fi
