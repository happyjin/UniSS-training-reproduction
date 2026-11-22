#!/usr/bin/env bash
# sst_hsmm_v1 P1a (reports/sst_hsmm_v1/prereg.zh-CN.md 3.2): torchrun 8 cards -> eval_outputs/sst_hsmm_v1/p1/train, then the
# merge into a vLLM-servable Thinker (merge.py, base = wave17 merged_enc_wise05) -> eval_outputs/sst_hsmm_v1/p1/merged.
#   setsid nohup bash experiments/sst_hsmm_v1/p1_orch.sh > logs/sst_hsmm_v1/p1_orch.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
R=eval_outputs/sst_hsmm_v1/p1
LOGD=logs/sst_hsmm_v1/p1
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
if [ -z "$DRY" ]; then
  [ -e $R/train/adapters_final.pt ] && { echo "exists: $R/train" >&2; exit 1; }
  [ -s eval_outputs/sst_hsmm_v1/p0/summary.json ] || { echo "no p0 summary" >&2; exit 1; }
  mkdir -p $LOGD $R
  git rev-parse HEAD > $R/commit.txt
  start_watcher
fi
p=$(launch 0,1,2,3,4,5,6,7 train "${QENV[@]}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_P2P_DISABLE=0 \
    /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -m torch.distributed.run --nproc-per-node 8 \
    --master-port 29611 -m experiments.sst_hsmm_v1.train --out $R/train)
waitp "$p"
[ -n "$DRY" ] && exit 0
[ -s $R/train/adapters_final.pt ] || die "train"
log "train done"
PYTHONPATH=. $PYQ experiments/sst_hsmm_v1/merge.py --adapters $R/train/adapters_final.pt --out $R/merged > $LOGD/merge.log 2>&1 || die "merge"
grep -q '"adapted"' $LOGD/merge.log || die "merge check"
touch $STOP
touch $R/DONE
log "p1 done"
