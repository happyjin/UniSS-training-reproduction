#!/usr/bin/env bash
# change 67 prechecks for TAF-S (hsc30 --taf K), on the card pool:
#  P0 real time: edev shard 0 first 8 items per direction, theta 0.75 (en2zh heal / zh2en base), with and without --taf K
#     -> per-round wall times (combo json round_s); GO iff p95 round time with TAF <= 1.0 s in both directions
#  P1 discrimination: edev + cvdev shard 0, both directions, --taf K --taf-shadow --shadow-final -> taf_p1.py
#  -> $D0/taf/precheck_K<K>.json and $D0/taf/PRE_K<K>_GO or PRE_K<K>_NOGO. env: K (default 8). Refuses existing outputs.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
K=${K:-8}
D0=eval_outputs/sst_hsmm_v1
R=$D0/taf/pre_K$K
LOGD=logs/sst_hsmm_v1/taf_pre_K$K
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
source experiments/sst_hsmm_v1/lib_pool.sh
VER=$D0/v/lora/a1.0000
TH=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
ON=experiments.sst_hsmm_v1.hsc30_si_thinker
if [ -z "$DRY" ]; then
  for x in $R $LOGD $D0/taf/precheck_K$K.json; do [ -e $x ] && { echo "exists: $x" >&2; exit 1; }; done
  mkdir -p $LOGD $R/p0
  git rev-parse HEAD > $R/commit.txt
  for d in en2zh zh2en; do head -8 $D0/pfb0/dev/edev/data/items_${d}_0.jsonl > $R/p0/items_$d.jsonl; done
  start_watcher
fi
vmode() { [ "$1" = en2zh ] && echo heal || echo base; }
vw() { [ "$1" = edev ] && echo $D0/edev/views || echo eval_outputs/mist_w3po_v1/wave15/views; }
ps_=(); g=0
for d in en2zh zh2en; do
  m=$(vmode $d)
  for arm in taf base; do
    XF=""; [ $arm = taf ] && XF="--taf $K"
    ps_+=("$(launch $g p0_${d}_$arm "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m $ON --mode hsc --variant n2 --samples 0 \
      --params $D0/h14/params.json --theta 0.75 --verifier $VER --second-view enh --views $(vw edev)/$d --thinker $TH --direction $d \
      --gpu-mem 0.6 --variants $m@0.75 $XF --items-file $R/p0/items_$d.jsonl --out $R/p0/${d}_$arm)")
    g=$(( (g + 1) % 8 )); [ -n "$DRY" ] || sleep 20
  done
  for ds in edev cvdev; do
    ps_+=("$(launch $g p1_${ds}_$d "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m $ON --mode hsc --variant n2 --samples 0 \
      --params $D0/h14/params.json --theta 0.75 --verifier $VER --second-view enh --views $(vw $ds)/$d --thinker $TH --direction $d \
      --gpu-mem 0.6 --variants $m@0.75 --taf $K --taf-shadow --shadow-final \
      --items-file $D0/pfb0/dev/$ds/data/items_${d}_0.jsonl --out $R/p1/${ds}_$d)")
    g=$(( (g + 1) % 8 )); [ -n "$DRY" ] || sleep 20
  done
done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
touch $STOP
PYTHONPATH=. $PYQ experiments/sst_hsmm_v1/taf_p1.py --root $R --k $K --out $D0/taf/precheck_K$K.json > $LOGD/p1.log 2>&1 || die "p1 analysis"
if python3 -c "import json,sys; sys.exit(0 if json.load(open('$D0/taf/precheck_K$K.json'))['go'] else 1)"; then touch $D0/taf/PRE_K${K}_GO; else touch $D0/taf/PRE_K${K}_NOGO; fi
log "TAF precheck K=$K: $(tail -3 $LOGD/p1.log | tr '\n' ' ')"
