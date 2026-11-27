# portfolio 2026-11-27: lane helper v2 (sourced). Work-stealing: run <TAG> <env...> claims the method atomically (mkdir) and skips it
# when already claimed or its output exists, so several lanes can list the same methods; the pipeline log is never truncated (noclobber).
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
D0=eval_outputs/sst_hsmm_v1
B0W=$D0/pfb0/dev/RUNS_DONE
CL=logs/sst_hsmm_v1/pfclaims
mkdir -p $CL
run() {
  local t=$1; shift
  [ -e $D0/$t ] || [ -e logs/sst_hsmm_v1/$t.log ] && { log "lane $$: skip $t (exists)"; return; }
  mkdir $CL/$t 2>/dev/null || { log "lane $$: skip $t (claimed)"; return; }
  log "lane $$: start $t"
  ( set -C; env -u WAIT "$@" TAG=$t bash experiments/sst_hsmm_v1/pf_pipe2.sh > logs/sst_hsmm_v1/$t.log 2>&1 < /dev/null 8>&- 9>&- )
  log "lane $$: $t rc=$? $(ls $D0/$t 2>/dev/null | grep -E '^(DONE|FAILED)$' | tr '\n' ' ')"
}
A1() { run pfa1 WAIT=$B0W ROLL=1 FIT=hsmm_anchor XF="--bias-redraft 1.0"; }
A2() { run pfa2 WAIT=$B0W ROLL=1 FIT=hsmm_anchor XF="--mid-mbr 6 --mbr-util bid"; }
D1() { run pfd1 WAIT=$B0W ROLL=1 FIT=hsmm_anchor XF="--mid-mbr 6 --mbr-util support"; }
D2() { run pfd2 WAIT=$B0W ROLL=1 FIT=hsmm_anchor XF="--mid-mbr 6 --mbr-util support --sentcut"; }
C2() { run pfc2 WAIT=$B0W ROLL=1 FIT=hsmm_map ROLL_ITEMS=$D0/cvdev_a/data ROLL_VIEWS=eval_outputs/mist_w3po_v1/wave15/views FITX="--items-dir $D0/cvdev_a/data --views eval_outputs/mist_w3po_v1/wave15/views" CVDEV_DATA=$D0/cvdev_b/data; }
