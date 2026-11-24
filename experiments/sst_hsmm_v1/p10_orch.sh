#!/usr/bin/env bash
# P10 (prereg change 18): frozen translator (wave17 merged_enc_wise05) + trained write head.
#  1. gen_own_targets: the frozen model's own full-audio translations of the 16000 p4 training mixtures, 8 shards on cards 0-7
#  2. 3-step torchrun smoke of train_head (p10/smoke, must log 3 steps with samples > 0), 3. 1000-step train_head (p10/train)
#  4. merged -> wave17 merged_enc_wise05 (symlink, nothing merged), control.json = p7p/control.json (same controller), p10/DONE.
# Placeholders stopped by PID per card (lib.sh launch); watcher noplaceholder.sh. Refuses existing outputs.
#   setsid nohup bash experiments/sst_hsmm_v1/p10_orch.sh > logs/sst_hsmm_v1/p10_orch.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
R=eval_outputs/sst_hsmm_v1/p10
LOGD=logs/sst_hsmm_v1/p10
WAVE17=/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
TR=(/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -m torch.distributed.run --nproc-per-node 8 --master-port 29622)
if [ -z "$DRY" ]; then
  [ -e $R ] && { echo "exists: $R (refuse to overwrite)" >&2; exit 1; }
  mkdir -p $LOGD $R/targets
  git rev-parse HEAD > $R/commit.txt
  log "orch $$: P10 (frozen translator + write head, change 18)"
  start_watcher
fi
ps_=()
for k in 0 1 2 3 4 5 6 7; do
  ps_+=("$(launch $k gen_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$k $PYV -m experiments.sst_hsmm_v1.gen_own_targets \
    --thinker $WAVE17 --shard $k --num-shards 8 --out $R/targets/part_$k.jsonl)")
  [ -n "$DRY" ] || sleep 20
done
waitp "${ps_[@]}"
if [ -z "$DRY" ]; then
  for k in 0 1 2 3 4 5 6 7; do [ -s $R/targets/part_$k.jsonl ] || die "targets $k"; done
  cat $R/targets/part_{0,1,2,3,4,5,6,7}.jsonl > $R/own_targets.jsonl
  n=$(wc -l < $R/own_targets.jsonl); e=$(grep -c '"translation": ""' $R/own_targets.jsonl)
  [ "$n" = 16000 ] || die "own targets $n lines"
  log "own targets $n (empty $e)"
fi
p=$(launch 0,1,2,3,4,5,6,7 smoke "${QENV[@]}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "${TR[@]}" -m experiments.sst_hsmm_v1.train_head \
    --targets $R/own_targets.jsonl --out $R/smoke --max-steps-run 3)
waitp "$p"
if [ -z "$DRY" ]; then
  n=$($PYQ -c "import json; L=[json.loads(l) for l in open('$R/smoke/metrics_rank0.jsonl')]; print(sum(1 for x in L if x['samples'] > 0))" 2>/dev/null)
  [ "$n" = 3 ] || die "smoke ($n steps with samples)"
  log "smoke ok: $(tail -n 1 $R/smoke/metrics_rank0.jsonl | cut -c1-400)"
fi
p=$(launch 0,1,2,3,4,5,6,7 train "${QENV[@]}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "${TR[@]}" -m experiments.sst_hsmm_v1.train_head \
    --targets $R/own_targets.jsonl --out $R/train)
waitp "$p"
[ -n "$DRY" ] && exit 0
[ -s $R/train/gate_final.pt ] && [ -s $R/train/head_final.pt ] || die "train"
log "train done"
ln -s $WAVE17 $R/merged || die "link merged"
[ -s $R/merged/config.json ] || die "merged link check"
cp -n eval_outputs/sst_hsmm_v1/p7p/control.json $R/control.json && [ -s $R/control.json ] || die "control"
touch $STOP
touch $R/DONE
log "p10 done: $(tail -n 1 $R/train/metrics_rank0.jsonl | cut -c1-400)"
