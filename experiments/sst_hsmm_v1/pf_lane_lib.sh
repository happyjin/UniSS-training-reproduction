# portfolio 2026-11-27: lane helper (sourced). run <TAG> <env assignments...>: one pf_pipe2.sh pipeline in the foreground of the lane.
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
run() { local t=$1; shift; log "lane $$: start $t"; env -u WAIT "$@" TAG=$t bash experiments/sst_hsmm_v1/pf_pipe2.sh > logs/sst_hsmm_v1/$t.log 2>&1 < /dev/null 8>&- 9>&-; log "lane $$: $t rc=$? $(ls eval_outputs/sst_hsmm_v1/$t 2>/dev/null | grep -E '^(DONE|FAILED)$' | tr '\n' ' ')"; }
D0=eval_outputs/sst_hsmm_v1
B0W=$D0/pfb0/dev/RUNS_DONE
