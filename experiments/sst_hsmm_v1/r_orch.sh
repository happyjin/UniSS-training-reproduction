#!/usr/bin/env bash
# prereg change 22 (CR-LA): items (p4 s < 600) -> wave17 LA2 rollouts (8 jobs) -> commit events -> prefix-forced full-audio
# continuations (8 jobs) -> regret.jsonl -> 3-step smoke -> 1000-step train_crla -> merge -> r/DONE. Cards 0-7; placeholders
# stopped by PID (lib.sh). Refuses outputs.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
R=eval_outputs/sst_hsmm_v1/r
LOGD=logs/sst_hsmm_v1/r
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
W17=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
TR=(/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -m torch.distributed.run --nproc-per-node 8 --master-port 29625)
if [ -z "$DRY" ]; then
  [ -e $R ] && { echo "exists: $R (refuse to overwrite)" >&2; exit 1; }
  mkdir -p $LOGD $R
  git rev-parse HEAD > $R/commit.txt
  log "orch $$: CR-LA (change 22)"
  PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.crla_regret items --root $R > $LOGD/items.log 2>&1 || die "items"
  start_watcher
fi
ps_=(); g=0
for d in en2zh zh2en; do for k in 0 1 2 3; do
  ps_+=("$(launch $g roll_${d}_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream \
    --thinker $W17 --direction $d --windows 1 --policy la2 --gpu-mem 0.85 \
    --items-file $R/data/items_${d}_$k.jsonl --out $R/roll_${d}_$k)")
  g=$((g + 1)); [ -n "$DRY" ] || sleep 20
done; done
waitp "${ps_[@]}"
if [ -z "$DRY" ]; then
  for d in en2zh zh2en; do for k in 0 1 2 3; do [ -s $R/roll_${d}_$k/blocks_${d}_w1.jsonl ] || die "roll $d $k"; done; done
  PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.crla_regret events --root $R > $LOGD/events.log 2>&1 || die "events"
  log "events: $(tail -n 1 $LOGD/events.log)"
fi
ps_=()
for k in 0 1 2 3 4 5 6 7; do
  ps_+=("$(launch $k pfx_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$k $PYV -m experiments.sst_hsmm_v1.prefix_forced \
    --events $R/events.jsonl --thinker $W17 --shard $k --num-shards 8 --out $R/forced_$k.jsonl)")
  [ -n "$DRY" ] || sleep 20
done
waitp "${ps_[@]}"
if [ -z "$DRY" ]; then
  for k in 0 1 2 3 4 5 6 7; do [ -s $R/forced_$k.jsonl ] || die "pfx $k"; done
  PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.crla_regret regret --root $R > $LOGD/regret.log 2>&1 || die "regret"
  log "regret: $(tail -n 1 $LOGD/regret.log)"
fi
p=$(launch 0,1,2,3,4,5,6,7 smoke "${QENV[@]}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "${TR[@]}" -m experiments.sst_hsmm_v1.train_crla \
    --regret $R/regret.jsonl --out $R/smoke --max-steps-run 3)
waitp "$p"
if [ -z "$DRY" ]; then
  n=$($PYQ -c "import json; L=[json.loads(l) for l in open('$R/smoke/metrics_rank0.jsonl')]; print(sum(1 for x in L if x['samples'] > 0 and x['events'] > 0))" 2>/dev/null)
  [ "$n" = 3 ] || die "smoke ($n steps with samples)"
  log "smoke ok: $(tail -n 1 $R/smoke/metrics_rank0.jsonl | cut -c1-500)"
fi
p=$(launch 0,1,2,3,4,5,6,7 train "${QENV[@]}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "${TR[@]}" -m experiments.sst_hsmm_v1.train_crla \
    --regret $R/regret.jsonl --out $R/train)
waitp "$p"
[ -n "$DRY" ] && exit 0
[ -s $R/train/adapters_final.pt ] || die "train"
log "train done"
touch $STOP
PYTHONPATH=. $PYQ experiments/sst_hsmm_v1/merge.py --adapters $R/train/adapters_final.pt --out $R/merged > $LOGD/merge.log 2>&1 || die "merge"
grep -q '"adapted"' $LOGD/merge.log || die "merge check"
touch $R/DONE
log "r done: $(tail -n 1 $LOGD/merge.log)"
