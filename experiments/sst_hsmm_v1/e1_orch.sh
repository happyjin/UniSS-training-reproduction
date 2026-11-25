#!/usr/bin/env bash
# E1 + E2 head (prereg change 27): SDC rollout (sdc_si_thinker --mode rollout = N's policy + features/labels) on the open
# UniST p4 mixtures (e1/data, DeepFilterNet3 views e1/views), 4 shards per direction on cards 0-7 -> e1/roll_<d>_<k>;
# then train_sdc (CPU) -> e2/head.pt. Refuses existing outputs.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
E=eval_outputs/sst_hsmm_v1/e1
R=$E/run
NTH=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
VER=eval_outputs/sst_hsmm_v1/v/lora/a1.0000
LOGD=logs/sst_hsmm_v1/e1run
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
if [ -z "$DRY" ]; then
  [ -e $R ] && { echo "exists: $R" >&2; exit 1; }
  for d in en2zh zh2en; do [ -s $E/views/$d/DONE.json ] || { echo "views $d missing" >&2; exit 1; }; done
  mkdir -p $LOGD $R/data; git rev-parse HEAD > $R/commit.txt
  start_watcher
  for d in en2zh zh2en; do
    $PYQ -I - $E/data/items_$d.jsonl $R/data $d <<'PY' || die "split $d"
import sys
rows = [l for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
n = len(rows); b = [round(k * n / 4) for k in range(5)]
for k in range(4):
    with open(f"{sys.argv[2]}/items_{sys.argv[3]}_{k}.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(rows[b[k]:b[k + 1]])
PY
  done
fi
ps_=(); g=0
for d in en2zh zh2en; do for k in 0 1 2 3; do
  ps_+=("$(launch $g roll_${d}_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.sst_hsmm_v1.sdc_si_thinker --mode rollout \
    --verifier $VER --second-view enh --views $E/views/$d --thinker $NTH --direction $d --gpu-mem 0.6 \
    --items-file $R/data/items_${d}_$k.jsonl --out $E/roll_${d}_$k)")
  g=$((g + 1)); [ -n "$DRY" ] || sleep 40
done; done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
for d in en2zh zh2en; do for k in 0 1 2 3; do [ -s $E/roll_${d}_$k/feats_$d.npz ] || die "roll $d $k"; done; done
touch $STOP
PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.train_sdc --root $E --out eval_outputs/sst_hsmm_v1/e2 > $LOGD/train_sdc.log 2>&1 || die "train_sdc"
touch $R/DONE
log "e1 done"
