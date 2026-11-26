#!/usr/bin/env bash
# prereg change 46 (D3), stages 0-1: d3_pre (card 0) -> stop on NO-GO -> wait for p3hk2 (DONE / FAILED) -> 3-step 8-card smoke of
# train_d3 (d3/smoke) -> 1000-step train_d3 (d3/train) -> divergence check (CE_a last-100 mean <= 1.5 x first-20 mean)
# -> merge.py (d3/merged) -> d3/TRAIN_DONE. Placeholders stopped by PID (lib.sh). Refuses existing outputs.
#   setsid nohup bash experiments/sst_hsmm_v1/d3_orch.sh > logs/sst_hsmm_v1/d3_orch.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
R=eval_outputs/sst_hsmm_v1/d3
LOGD=logs/sst_hsmm_v1/d3
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
TR=(/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -m torch.distributed.run --nproc-per-node 8 --master-port 29646)
if [ -z "$DRY" ]; then
  for x in $R/pre $R/smoke $R/train $R/merged; do [ -e $x ] && { echo "exists: $x (refuse to overwrite)" >&2; exit 1; }; done
  [ -s $R/src_words.jsonl ] || { echo "missing $R/src_words.jsonl (train_d3 --prep)" >&2; exit 1; }
  mkdir -p $LOGD
  git rev-parse HEAD > $R/commit.txt
  log "orch $$: D3 (change 46) stages 0-1"
  start_watcher
fi
p=$(launch 0 pre "${QENV[@]}" CUDA_VISIBLE_DEVICES=0 $PYQ -m experiments.sst_hsmm_v1.d3_pre --out $R/pre)
waitp "$p"
if [ -z "$DRY" ]; then
  [ -s $R/pre/result.json ] || die "pre"
  go=$($PYQ -c "import json; print(json.load(open('$R/pre/result.json'))['GO'])")
  log "stage 0: $($PYQ -c "import json; r=json.load(open('$R/pre/result.json')); print({d: round(r[d]['clean']['hit5'], 4) for d in ('en2zh', 'zh2en')}, r['mean_clean_hit5'])") GO=$go"
  [ "$go" = True ] || { log "stage 0 NO-GO: D3 stops"; touch $R/NOGO $STOP; exit 0; }
  log "waiting for p3hk2"
  until [ -e eval_outputs/sst_hsmm_v1/p3hk2/DONE ] || [ -e eval_outputs/sst_hsmm_v1/p3hk2/FAILED ]; do sleep 30; done
  sleep 30
fi
p=$(launch 0,1,2,3,4,5,6,7 smoke "${QENV[@]}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "${TR[@]}" -m experiments.sst_hsmm_v1.train_d3 \
    --out $R/smoke --max-steps-run 3)
waitp "$p"
if [ -z "$DRY" ]; then
  n=$($PYQ -c "import json; L=[json.loads(l) for l in open('$R/smoke/metrics_rank0.jsonl')]; print(sum(1 for x in L if x['n_a'] > 0 and x['n_b'] > 0))" 2>/dev/null)
  [ "$n" = 3 ] || die "smoke ($n steps with samples)"
  log "smoke ok: $(tail -n 1 $R/smoke/metrics_rank0.jsonl | cut -c1-600)"
fi
p=$(launch 0,1,2,3,4,5,6,7 train "${QENV[@]}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "${TR[@]}" -m experiments.sst_hsmm_v1.train_d3 --out $R/train)
waitp "$p"
[ -n "$DRY" ] && exit 0
[ -s $R/train/adapters_final.pt ] || die "train"
$PYQ - $R/train/metrics_rank0.jsonl <<'PY' || { log "stage 1: CE_a diverged; D3 stops"; touch $R/DIVERGED $STOP; exit 0; }
import json, sys
L = [json.loads(l) for l in open(sys.argv[1])]
ce = [(x["en2zh_ce_a"] + x["zh2en_ce_a"]) / 2 for x in L]
a, b = sum(ce[:20]) / 20, sum(ce[-100:]) / 100
print("CE_a first-20", round(a, 4), "last-100", round(b, 4))
sys.exit(0 if b <= 1.5 * a else 1)
PY
log "train done"
touch $STOP
PYTHONPATH=. $PYQ experiments/sst_hsmm_v1/merge.py --adapters $R/train/adapters_final.pt --out $R/merged > $LOGD/merge.log 2>&1 || die "merge"
grep -q '"adapted"' $LOGD/merge.log || die "merge check"
touch $R/TRAIN_DONE
log "d3 stages 0-1 done: $(tail -n 1 $LOGD/merge.log)"
