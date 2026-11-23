#!/usr/bin/env bash
# sst_hsmm_v1 P5 (prereg change 8): waits for option A (p2d/DONE; stops on p2d/FAILED), budgets (calib_budget ->
# p5/budget.json), then a 3-step torchrun smoke of train_risk2 (p5/smoke, must log 3 steps with samples > 0), the full
# 1000-step training (p5/train), merge into a vLLM Thinker (p5/merged), p5/DONE. Cards 0-7; placeholders stopped by PID; watcher noplaceholder.sh. Refuses outputs.
#   setsid nohup bash experiments/sst_hsmm_v1/p5_orch.sh > logs/sst_hsmm_v1/p5_orch.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
R=eval_outputs/sst_hsmm_v1/p5
A=eval_outputs/sst_hsmm_v1/p2d
LOGD=logs/sst_hsmm_v1/p5
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
TR=(/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -m torch.distributed.run --nproc-per-node 8 --master-port 29617)
if [ -z "$DRY" ]; then
  [ -e $R/train ] || [ -e $R/smoke ] && { echo "exists: $R/train or smoke" >&2; exit 1; }
  mkdir -p $LOGD
  log "orch $$: waiting for $A/DONE"
  until [ -e $A/DONE ]; do [ -e $A/FAILED ] && { log "A FAILED: no P5"; exit 1; }; sleep 30; done
  mkdir -p $R
  git rev-parse HEAD > $R/commit.txt
  PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.calib_budget --out $R/budget.json > $LOGD/calib.log 2>&1 || die "calib"
  log "budget: $(grep -E '^(en2zh|zh2en) ' $LOGD/calib.log | tr '\n' ' ')"
  start_watcher
fi
p=$(launch 0,1,2,3,4,5,6,7 smoke "${QENV[@]}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "${TR[@]}" -m experiments.sst_hsmm_v1.train_risk2 \
    --out $R/smoke --max-steps-run 3)
waitp "$p"
if [ -z "$DRY" ]; then
  n=$($PYQ -c "import json; L=[json.loads(l) for l in open('$R/smoke/metrics_rank0.jsonl')]; print(sum(1 for x in L if x['samples'] > 0))" 2>/dev/null)
  [ "$n" = 3 ] || die "smoke ($n steps with samples)"
  log "smoke ok: $(tail -n 1 $R/smoke/metrics_rank0.jsonl)"
fi
p=$(launch 0,1,2,3,4,5,6,7 train "${QENV[@]}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "${TR[@]}" -m experiments.sst_hsmm_v1.train_risk2 \
    --out $R/train)
waitp "$p"
[ -n "$DRY" ] && exit 0
[ -s $R/train/adapters_final.pt ] || die "train"
log "train done"
PYTHONPATH=. $PYQ experiments/sst_hsmm_v1/merge.py --adapters $R/train/adapters_final.pt --out $R/merged > $LOGD/merge.log 2>&1 || die "merge"
grep -q '"adapted"' $LOGD/merge.log || die "merge check"
touch $STOP
touch $R/DONE
log "p5 done: $(tail -n 1 $LOGD/merge.log)"
