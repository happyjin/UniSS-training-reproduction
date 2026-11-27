#!/usr/bin/env bash
# overnight 2026-11-27 final (copy of pv4_final.sh): waits for every file in WAITS (space list, relative to D0) -> if any new test (NEWS) exists: COMET dump (CPU) + wmt22-comet-da (1 pool card) for the
# systems of the final table -> pv4_table.py -> reports/sst_hsmm_v1/pv4_tables.zh-CN.md; then touches pv4/FINAL_DONE.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
D0=eval_outputs/sst_hsmm_v1
R=$D0/pv8/comet
LOGD=logs/sst_hsmm_v1/pv8_final
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
source experiments/sst_hsmm_v1/lib_pool.sh
SYSL="p3hk p3hk2 p3x p3d3s05 p3p2fxff p3pfd1 p3pfe1 p3pfh3 p3pfe1d1"
if [ -z "$DRY" ]; then
  for x in $R $LOGD; do [ -e $x ] && { echo "exists: $x" >&2; exit 1; }; done
  mkdir -p $LOGD
  for w in ${WAITS:-}; do until [ -e $D0/$w ]; do sleep 60; done; done
  NEW=""; for s in ${NEWS:?}; do [ -e $D0/$s/summary.json ] && NEW="$NEW $s"; done
  [ -z "$NEW" ] && { mkdir -p $D0/pv8; touch $D0/pv8/FINAL_DONE; log "pv8 final: no new test"; exit 0; }
  SYSL="$SYSL$NEW"
  mkdir -p $R
  PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.pf_comet_dump --out $R/dump.jsonl $SYSL > $LOGD/dump.log 2>&1 || die "dump"
  start_watcher
fi
p=$(launch 0 comet env HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface CUDA_VISIBLE_DEVICES=0 /opt/dlami/nvme/jasonleeeli/venvs/comet/bin/python \
  experiments/sst_hsmm_v1/pf_comet_score.py --dump $R/dump.jsonl --out $R/comet.json)
waitp "$p"
[ -n "$DRY" ] && exit 0
touch $STOP
[ -s $R/comet.json ] || die "comet"
python3 experiments/sst_hsmm_v1/pv8_table.py reports/sst_hsmm_v1/${TABLE:-pv8_tables}.zh-CN.md > $LOGD/table.log 2>&1 || die "table"
touch $D0/pv8/FINAL_DONE
log "pv8 final done"
