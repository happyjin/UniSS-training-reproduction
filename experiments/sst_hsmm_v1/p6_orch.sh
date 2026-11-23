#!/usr/bin/env bash
# P6 (prereg change 11): copy of p5b_orch.sh with train_risk4 (multiplier-conditioned Gate2), output p6 (control.json made by p6_control beforehand), port 29619. Header below refers to the original.
# sst_hsmm_v1 P5' (prereg change 9): copy of p5b_orch.sh with train_risk3 (relative budgets, p5/budget.json reused), output p5b, port 29618; no wait for A.
# Then a 3-step torchrun smoke of train_risk3 (p5/smoke, must log 3 steps with samples > 0), the full
# 1000-step training (p5/train), merge into a vLLM Thinker (p5/merged), p5/DONE. Cards 0-7; placeholders stopped by PID; watcher noplaceholder.sh. Refuses outputs.
#   setsid nohup bash experiments/sst_hsmm_v1/p6_orch.sh > logs/sst_hsmm_v1/p6_orch.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
R=eval_outputs/sst_hsmm_v1/p6
LOGD=logs/sst_hsmm_v1/p6
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
TR=(/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -m torch.distributed.run --nproc-per-node 8 --master-port 29619)
if [ -z "$DRY" ]; then
  [ -e $R/train ] || [ -e $R/smoke ] && { echo "exists: $R/train or smoke" >&2; exit 1; }
  [ -s $R/control.json ] || { echo "missing $R/control.json" >&2; exit 1; }
  mkdir -p $LOGD $R
  git rev-parse HEAD > $R/commit.txt
  log "orch $$: P6 (multiplier-conditioned policy, change 11)"
  start_watcher
fi
p=$(launch 0,1,2,3,4,5,6,7 smoke "${QENV[@]}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "${TR[@]}" -m experiments.sst_hsmm_v1.train_risk4 \
    --out $R/smoke --max-steps-run 3)
waitp "$p"
if [ -z "$DRY" ]; then
  n=$($PYQ -c "import json; L=[json.loads(l) for l in open('$R/smoke/metrics_rank0.jsonl')]; print(sum(1 for x in L if x['samples'] > 0))" 2>/dev/null)
  [ "$n" = 3 ] || die "smoke ($n steps with samples)"
  log "smoke ok: $(tail -n 1 $R/smoke/metrics_rank0.jsonl)"
fi
p=$(launch 0,1,2,3,4,5,6,7 train "${QENV[@]}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "${TR[@]}" -m experiments.sst_hsmm_v1.train_risk4 \
    --out $R/train)
waitp "$p"
[ -n "$DRY" ] && exit 0
[ -s $R/train/adapters_final.pt ] || die "train"
log "train done"
PYTHONPATH=. $PYQ experiments/sst_hsmm_v1/merge.py --adapters $R/train/adapters_final.pt --out $R/merged > $LOGD/merge.log 2>&1 || die "merge"
grep -q '"adapted"' $LOGD/merge.log || die "merge check"
touch $STOP
touch $R/DONE
log "p6 done: $(tail -n 1 $LOGD/merge.log)"
