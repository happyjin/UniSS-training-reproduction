#!/usr/bin/env bash
# change 57 G0 gate for one LoRA drafter: whole-audio wave2_S1 dev decodes (4 shards x 2 directions, card pool) of $D0/$TR/merged with
# module $OFFMOD (dev_offline_tag = TAG_C prompt as E1/B1 final read; dev_offline_plain = plain prompt as B2) -> pv4_g0_score ->
# $D0/pv4/g0_$M.json and $D0/pv4/G0_${M}_PASS or G0_${M}_FAIL. env: M, TR, OFFMOD. Refuses existing outputs.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
: "${M:?}" "${TR:?}" "${OFFMOD:?}"
D0=eval_outputs/sst_hsmm_v1
R=$D0/pv4/g0_$M
LOGD=logs/sst_hsmm_v1/pv4_g0_$M
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
source experiments/sst_hsmm_v1/lib_pool.sh
if [ -z "$DRY" ]; then
  for x in $R $LOGD $D0/pv4/g0_$M.json; do [ -e $x ] && { echo "exists: $x" >&2; exit 1; }; done
  mkdir -p $LOGD $R
  start_watcher
fi
ps_=(); g=0
for d in en2zh zh2en; do for k in 0 1 2 3; do
  ps_+=("$(launch $g off_${d}_$k "${VENV[@]}" OFF_MODEL=$D0/$TR/merged CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.sst_hsmm_v1.$OFFMOD gen \
    --direction $d --shard $k --num-shards 4 --gpu-mem 0.6 --out $R/${d}_$k.jsonl)")
  g=$(( (g + 1) % 8 )); [ -n "$DRY" ] || sleep 20
done; done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
touch $STOP
PYTHONPATH=. $PYQ experiments/sst_hsmm_v1/pv4_g0_score.py --out $D0/pv4/g0_$M.json --model "$R/{d}_*.jsonl" > $LOGD/score.log 2>&1 || die "score"
if python3 -c "import json,sys; sys.exit(0 if json.load(open('$D0/pv4/g0_$M.json'))['pass'] else 1)"; then touch $D0/pv4/G0_${M}_PASS; else touch $D0/pv4/G0_${M}_FAIL; fi
log "G0 $M: $(cat $LOGD/score.log | tail -1)"
