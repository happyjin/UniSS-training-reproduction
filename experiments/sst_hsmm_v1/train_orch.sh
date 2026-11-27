#!/usr/bin/env bash
# change 49+: generic LoRA-drafter training (copy of d3_orch.sh stage 1): optional WAIT file -> 8-card 3-step smoke -> full train
# (torchrun, module $MOD) -> CE_a divergence check (last-100 mean <= 1.5 x first-20) -> merge.py ($R/merged) and wave16_make_lora
# alpha 1 ($R/peft/a1.0000) -> $R/TRAIN_DONE. env: TAG, MOD, PORT, optional WAIT. Refuses existing outputs.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
: "${TAG:?}" "${MOD:?}" "${PORT:?}"
R=eval_outputs/sst_hsmm_v1/$TAG
LOGD=logs/sst_hsmm_v1/${TAG}_train
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
TR=($PYQ -m torch.distributed.run --nproc-per-node 8 --master-port $PORT)
if [ -z "$DRY" ]; then
  for x in $R $LOGD; do [ -e $x ] && { echo "exists: $x (refuse to reuse; pick a new TAG)" >&2; exit 1; }; done
  mkdir -p $LOGD $R
  if [ -n "${WAIT:-}" ]; then log "$TAG train $$: waiting for $WAIT"; until [ -e "$WAIT" ]; do sleep 30; done; sleep 30; fi
  git rev-parse HEAD > $R/commit_train.txt
  start_watcher
fi
p=$(launch 0,1,2,3,4,5,6,7 smoke "${QENV[@]}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "${TR[@]}" -m $MOD --out $R/smoke --max-steps-run 3)
waitp "$p"
if [ -z "$DRY" ]; then
  n=$($PYQ -c "import json; L=[json.loads(l) for l in open('$R/smoke/metrics_rank0.jsonl')]; print(sum(1 for x in L if x['n_a'] > 0 and x['n_b'] > 0))" 2>/dev/null)
  [ "$n" = 3 ] || die "smoke ($n steps with samples)"
  log "smoke ok: $(tail -n 1 $R/smoke/metrics_rank0.jsonl | cut -c1-500)"
fi
p=$(launch 0,1,2,3,4,5,6,7 train "${QENV[@]}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "${TR[@]}" -m $MOD --out $R/train)
waitp "$p"
[ -n "$DRY" ] && exit 0
[ -s $R/train/adapters_final.pt ] || die "train"
$PYQ - $R/train/metrics_rank0.jsonl <<'PY' || { log "CE_a diverged; $TAG stops"; touch $R/DIVERGED $STOP; exit 0; }
import json, sys
L = [json.loads(l) for l in open(sys.argv[1])]
ce = [(x["en2zh_ce_a"] + x["zh2en_ce_a"]) / 2 for x in L]
a, b = sum(ce[:20]) / 20, sum(ce[-100:]) / 100
print("CE_a first-20", round(a, 4), "last-100", round(b, 4))
sys.exit(0 if b <= 1.5 * a else 1)
PY
touch $STOP
PYTHONPATH=. $PYQ experiments/sst_hsmm_v1/merge.py --adapters $R/train/adapters_final.pt --out $R/merged > $LOGD/merge.log 2>&1 || die "merge"
grep -q '"adapted"' $LOGD/merge.log || die "merge check"
PYTHONPATH=. $PYQ experiments/mist_w3po_v1/voicecond/wave16_make_lora.py --adapters $R/train/adapters_final.pt --alphas 1 --out $R/peft > $LOGD/peft.log 2>&1 || die "peft"
touch $R/TRAIN_DONE
log "$TAG train done: $(tail -n 1 $LOGD/merge.log)"
