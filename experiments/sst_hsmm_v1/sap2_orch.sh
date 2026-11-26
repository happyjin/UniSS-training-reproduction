#!/usr/bin/env bash
# prereg change 37-4 SAP2 (copy of t_orch.sh): data2 (CPU) -> risk_prep2 -> p5 (CPU) -> 3-step torchrun smoke of train_sap2 (t2/smoke)
# -> 4000-step train_sap2 (t2/train) -> wave16_make_lora alpha 1 -> v2/lora/a1.0000 -> t2/DONE. Cards 0-7. Refuses outputs.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
D=eval_outputs/sst_hsmm_v1
R=$D/t2
LOGD=logs/sst_hsmm_v1/t2
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
TR=(/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -m torch.distributed.run --nproc-per-node 8 --master-port 29627)
if [ -z "$DRY" ]; then
  [ -e $R ] && { echo "exists: $R (refuse to overwrite)" >&2; exit 1; }
  [ -e $D/v2 ] && { echo "exists: $D/v2" >&2; exit 1; }
  mkdir -p $LOGD $R
  git rev-parse HEAD > $R/commit.txt
  log "orch $$: SAP2 (change 37-4)"
  [ -s $D/data2/train_zh2en.jsonl ] || { PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.data2 --procs 96 > $LOGD/data2.log 2>&1 || die "data2"; }
  log "data2: $(tail -n 2 $LOGD/data2.log | tr '\n' ' ' | cut -c1-300)"
  [ -s $D/p5/samples.jsonl ] || { OMP_NUM_THREADS=1 PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.risk_prep2 --out $D/p5 --procs 96 > $LOGD/p5.log 2>&1 || die "p5"; }
  log "p5: $(tail -n 2 $LOGD/p5.log | tr '\n' ' ' | cut -c1-300)"
  start_watcher
fi
p=$(launch 0,1,2,3,4,5,6,7 smoke "${QENV[@]}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "${TR[@]}" -m experiments.sst_hsmm_v1.train_sap2 \
    --out $R/smoke --max-steps-run 3)
waitp "$p"
if [ -z "$DRY" ]; then
  n=$($PYQ -c "import json; L=[json.loads(l) for l in open('$R/smoke/metrics_rank0.jsonl')]; print(sum(1 for x in L if x['samples'] > 0))" 2>/dev/null)
  [ "$n" = 3 ] || die "smoke ($n steps with samples)"
  log "smoke ok: $(tail -n 1 $R/smoke/metrics_rank0.jsonl | cut -c1-300)"
fi
p=$(launch 0,1,2,3,4,5,6,7 train "${QENV[@]}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "${TR[@]}" -m experiments.sst_hsmm_v1.train_sap2 --out $R/train)
waitp "$p"
[ -n "$DRY" ] && exit 0
[ -s $R/train/adapters_final.pt ] || die "train"
touch $STOP
log "train done"
$PYQ experiments/mist_w3po_v1/voicecond/wave16_make_lora.py --adapters $R/train/adapters_final.pt --alphas 1 --out $D/v2/lora > $LOGD/make_lora.log 2>&1 || die "make_lora"
[ -s $D/v2/lora/a1.0000/adapter_config.json ] || die "lora dir"
touch $R/DONE
log "t2 done"
