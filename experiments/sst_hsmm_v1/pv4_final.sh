#!/usr/bin/env bash
# change 57: after both chains (pv4_chain.sh b1 / b2) -> if any new test exists: COMET dump (CPU) + wmt22-comet-da (1 pool card) for the
# systems of the final table -> pv4_table.py -> reports/sst_hsmm_v1/pv4_tables.zh-CN.md; then touches pv4/FINAL_DONE.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
D0=eval_outputs/sst_hsmm_v1
R=$D0/pv4/comet
LOGD=logs/sst_hsmm_v1/pv4_final
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
source experiments/sst_hsmm_v1/lib_pool.sh
SYSL="p3hk p3hk2 p3x p3d3s05 p3p2fxff p3pfd1 p3pfe1 p3pfh3 p3pfe1d1"
if [ -z "$DRY" ]; then
  for x in $R $LOGD; do [ -e $x ] && { echo "exists: $x" >&2; exit 1; }; done
  mkdir -p $LOGD
  until [ -e $D0/pv4/CHAIN_b1_DONE ] && [ -e $D0/pv4/CHAIN_b2_DONE ]; do sleep 60; done
  NEW=""; for s in p3pfn1 p3pff1; do [ -e $D0/$s/summary.json ] && NEW="$NEW $s"; done
  [ -z "$NEW" ] && { touch $D0/pv4/FINAL_DONE; log "pv4 final: no new test (G0 / training)"; exit 0; }
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
python3 experiments/sst_hsmm_v1/pv4_table.py reports/sst_hsmm_v1/pv4_tables.zh-CN.md > $LOGD/table.log 2>&1 || die "table"
touch $D0/pv4/FINAL_DONE
log "pv4 final done"
