#!/usr/bin/env bash
# change 68 chain: no-harm offline precheck (ctx_offline gen on cvdev + edev x 2 directions, card pool; score) -> on GO the streaming
# dev + test pipeline TAG pfx1 (pf_pipe2, ROLL=0, FIT=none, h14, ONLINE=hsc30, XF="--ctx 3") -> $D0/ctx/CHAIN_DONE.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
D0=eval_outputs/sst_hsmm_v1
R=$D0/ctx
LOGD=logs/sst_hsmm_v1/ctx_pre
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
source experiments/sst_hsmm_v1/lib_pool.sh
ON=experiments.sst_hsmm_v1.hsc30_si_thinker
fin() { touch $R/CHAIN_DONE; log "ctx chain done: $*"; exit 0; }
if [ -z "$DRY" ]; then
  for x in $R $LOGD; do [ -e $x ] && { echo "exists: $x" >&2; exit 1; }; done
  mkdir -p $LOGD $R/pre
  git rev-parse HEAD > $R/commit.txt
  start_watcher
fi
ps_=(); g=0
for ds in cvdev edev; do for d in en2zh zh2en; do
  ps_+=("$(launch $g pre_${ds}_$d "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.sst_hsmm_v1.ctx_offline gen \
    --items $D0/pfb0/dev/$ds/data/items_$d.jsonl --direction $d --n 3 --gpu-mem 0.6 --out $R/pre/${ds}_$d)")
  g=$(( (g + 1) % 8 )); [ -n "$DRY" ] || sleep 20
done; done
waitp "${ps_[@]}"
[ -n "$DRY" ] && { echo "[dry] then pfx1 ON=$ON XF=--ctx 3"; exit 0; }
touch $STOP
PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.ctx_offline score --root $R/pre --out $R/precheck.json > $LOGD/score.log 2>&1 || die "score"
log "ctx precheck: $(tail -3 $LOGD/score.log | tr '\n' ' ')"
python3 -c "import json,sys; sys.exit(0 if json.load(open('$R/precheck.json'))['go'] else 1)" || { touch $R/PRE_NOGO; fin "no-harm precheck NOGO"; }
touch $R/PRE_GO
bash experiments/sst_hsmm_v1/make_pf_report.sh pfx1 $ON $D0/h14/params.json "--ctx 3" eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05 || fin "report script"
source experiments/sst_hsmm_v1/pf_lane_lib2.sh
run pfx1 ROLL=0 FIT=none ONLINE=$ON XF="--ctx 3"
fin "$(ls $D0/pfx1 2>/dev/null | grep -E '^(DONE|FAILED)$' | tr '\n' ' ') test $(ls $D0/p3pfx1 2>/dev/null | grep -E '^(DONE|FAILED)$' | tr '\n' ' ')"
