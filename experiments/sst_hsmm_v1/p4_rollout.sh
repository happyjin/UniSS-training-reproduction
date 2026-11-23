#!/usr/bin/env bash
# sst_hsmm_v1 P4 step 1 (prereg change 7): roll out the change-5 P1a (p1/merged + gate_final, threshold rule) on the P4
# training samples (risk_prep items, 4 shards per direction) on cards 0-7 -> p4/rollout_<d>_<k>/blocks_<d>_w1.jsonl,
# then p4/rollout/DONE. Placeholders stopped by PID (lib.sh launch); watcher noplaceholder.sh. Refuses existing outputs.
#   setsid nohup bash experiments/sst_hsmm_v1/p4_rollout.sh > logs/sst_hsmm_v1/p4_rollout.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
P1=eval_outputs/sst_hsmm_v1/p1
R=eval_outputs/sst_hsmm_v1/p4
LOGD=logs/sst_hsmm_v1/p4_rollout
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
declare -A TOKMS=([en2zh]=345.6 [zh2en]=332.5)
if [ -z "$DRY" ]; then
  [ -e $R/rollout ] && { echo "exists: $R/rollout" >&2; exit 1; }
  [ -s $R/samples.jsonl ] || { echo "no samples" >&2; exit 1; }
  mkdir -p $LOGD $R/rollout
  git rev-parse HEAD > $R/rollout/commit.txt
  start_watcher
fi
ps_=(); g=0
for d in en2zh zh2en; do for k in 0 1 2 3; do
  ps_+=("$(launch $g roll_${d}_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g SST_RULE=threshold SST_GATE=$P1/train/gate_final.pt SST_TOKMS=${TOKMS[$d]} \
    $PYV -m experiments.sst_hsmm_v1.e2e_si_thinker --thinker $P1/merged --direction $d \
    --items-file $R/items_${d}_$k.jsonl --out $R/rollout_${d}_$k)")
  g=$((g + 1)); [ -n "$DRY" ] || sleep 30
done; done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
touch $STOP
for d in en2zh zh2en; do for k in 0 1 2 3; do [ -s $R/rollout_${d}_$k/blocks_${d}_w1.jsonl ] || die "rollout $d $k"; done; done
touch $R/rollout/DONE
log "rollout done"
