#!/usr/bin/env bash
# change 65 driver (reversed-pair en2zh adapter): rev_prep decode (card pool, 1 card) -> build (CPU) -> check >= 7900 en2zh slots ->
# train_orch4.sh TAG=revtr MOD=train_p2ft_rev -> after TRAIN_DONE, in parallel: G0 report (pv4_g0.sh M=v1, OFFMOD dev_offline_tag;
# reported, not gating) and the HCG v2 streaming chain (pvx_chain.sh M=v1 PT=pfv1, hsc24 with the revtr adapter -> test p3pfv1).
# (rev_prep select already ran: eval_outputs/sst_hsmm_v1/rev/rows.jsonl, slots.jsonl.)
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
D0=eval_outputs/sst_hsmm_v1
LOGD=logs/sst_hsmm_v1/pv11_rev
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
source experiments/sst_hsmm_v1/lib_pool.sh
fin() { mkdir -p $D0/pvx; touch $D0/pvx/CHAIN_v1_DONE; log "rev chain stops: $*"; exit 0; }
if [ -z "$DRY" ]; then
  [ -e $LOGD ] && { echo "exists: $LOGD" >&2; exit 1; }
  [ -e $D0/rev/slots.jsonl ] || { echo "no rev/slots.jsonl" >&2; exit 1; }
  [ -e $D0/rev/clean ] && { echo "exists: rev/clean" >&2; exit 1; }
  mkdir -p $LOGD
  start_watcher
fi
p=$(launch 0 decode "${QENV[@]}" CUDA_VISIBLE_DEVICES=0 $PYQ -m experiments.sst_hsmm_v1.rev_prep decode)
waitp "$p"
[ -n "$DRY" ] && exit 0
[ -s $D0/rev/decode_stats.json ] || fin "decode failed"
log "decode: $(cat $D0/rev/decode_stats.json | tr -d '\n ')"
env PYTHONPATH=. OMP_NUM_THREADS=1 $PYQ -m experiments.sst_hsmm_v1.rev_prep build --procs 48 > $LOGD/build.log 2>&1 || fin "build failed"
n=$($PYQ -c "import json; print(json.load(open('$D0/rev/build_stats.json'))['en2zh_slots'])")
log "build: $(cat $D0/rev/build_stats.json | tr -d '\n ')"
[ "$n" -ge 7900 ] || fin "only $n en2zh slots"
touch $STOP
TAG=revtr MOD=experiments.sst_hsmm_v1.train_p2ft_rev PORT=29683 bash experiments/sst_hsmm_v1/train_orch4.sh > logs/sst_hsmm_v1/revtr_orch.log 2>&1 < /dev/null 8>&- 9>&-
[ -e $D0/revtr/TRAIN_DONE ] || fin "training did not finish ($(ls $D0/revtr | tr '\n' ' '))"
M=v1 TR=revtr OFFMOD=dev_offline_tag setsid nohup bash experiments/sst_hsmm_v1/pv4_g0.sh > logs/sst_hsmm_v1/pv4_g0_v1_orch.log 2>&1 < /dev/null 8>&- 9>&- &
M=v1 PT=pfv1 ON=experiments.sst_hsmm_v1.hsc24_si_thinker XF="--draft-lora $D0/revtr/peft/a1.0000 --draft-tag --mid-mbr 4 --mbr-util hcg" \
  bash experiments/sst_hsmm_v1/pvx_chain.sh
