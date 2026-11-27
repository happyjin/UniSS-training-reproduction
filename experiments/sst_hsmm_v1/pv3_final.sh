#!/usr/bin/env bash
# plan v3 (change 56): after lanes 9-11 are done -> COMET dump (CPU) + wmt22-comet-da (1 pool card) for every system of the final table
# -> pv3_table.py -> reports/sst_hsmm_v1/pv3_20261127_tables.zh-CN.md, then touches pv3/FINAL_DONE. Refuses existing outputs.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
D0=eval_outputs/sst_hsmm_v1
R=$D0/pv3/comet
LOGD=logs/sst_hsmm_v1/pv3_final
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
source experiments/sst_hsmm_v1/lib_pool.sh
SYSL="p3hk p3hk2 p3x p3d3s05 p3p2fxff p3pfd1 p3pfe1 p3pfh1 p3pfh3 p3pfh4 p3pfe1d1"
if [ -z "$DRY" ]; then
  for x in $R $LOGD; do [ -e $x ] && { echo "exists: $x" >&2; exit 1; }; done
  mkdir -p $LOGD
  until [ -e $D0/PV3_LANE9_DONE ] && [ -e $D0/PV3_LANE10_DONE ] && [ -e $D0/PV3_LANE11_DONE ]; do sleep 60; done
  for s in $SYSL; do [ -e $D0/$s ] || die "missing test root $s"; done
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
python3 experiments/sst_hsmm_v1/pv3_table.py reports/sst_hsmm_v1/pv3_20261127_tables.zh-CN.md > $LOGD/table.log 2>&1 || die "table"
touch $D0/pv3/FINAL_DONE
log "pv3 final done"
