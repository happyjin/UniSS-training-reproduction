#!/usr/bin/env bash
# SVC dev diagnostic (after the SVC test, to choose iteration 6 on dev, not on test): N combo (--second-view enh, wave17
# merged_enc_wise05, wave15 DeepFilterNet3 views) and SVC (svc_si_thinker, same + SAP verifier) on wave2_S1 dev items,
# Thinker only (text BLEU, commit stats), 2 shards per direction per system, 8 cards. Output eval_outputs/sst_hsmm_v1/dv.
#   setsid nohup bash experiments/sst_hsmm_v1/dv_orch.sh > logs/sst_hsmm_v1/dv_orch.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
R=eval_outputs/sst_hsmm_v1/dv
DATA=eval_outputs/mist_w3po_v1/wave2_S1/data
VIEWS=eval_outputs/mist_w3po_v1/wave15/views
NTH=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
VER=eval_outputs/sst_hsmm_v1/v/lora/a1.0000
LOGD=logs/sst_hsmm_v1/dv
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
if [ -z "$DRY" ]; then
  [ -e $R ] && { echo "exists: $R (refuse to overwrite)" >&2; exit 1; }
  mkdir -p $LOGD $R/data
  git rev-parse HEAD > $R/commit.txt
  start_watcher
  for d in en2zh zh2en; do
    $PYQ -I - $DATA/items_$d.jsonl $R/data $d <<'PY' || die "split $d"
import json, sys
rows = [l for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
n = len(rows); b = [round(k * n / 2) for k in range(3)]
for k in range(2):
    with open(f"{sys.argv[2]}/items_{sys.argv[3]}_{k}.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(rows[b[k]:b[k + 1]])
print(sys.argv[3], n, b)
PY
  done
fi
ps_=(); g=0
for sys_ in n svc; do for d in en2zh zh2en; do for k in 0 1; do
  if [ $sys_ = n ]; then M=(-m experiments.mist_w3po_v1.voicecond.combo_si_thinker); else M=(-m experiments.sst_hsmm_v1.svc_si_thinker --verifier $VER); fi
  ps_+=("$(launch $g ${sys_}_${d}_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV "${M[@]}" \
    --second-view enh --views $VIEWS/$d --thinker $NTH --direction $d --gpu-mem 0.6 \
    --items-file $R/data/items_${d}_$k.jsonl --out $R/${sys_}_${d}_$k)")
  g=$((g + 1)); [ -n "$DRY" ] || sleep 40
done; done; done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
for sys_ in n svc; do for d in en2zh zh2en; do for k in 0 1; do
  [ -s $R/${sys_}_${d}_$k/blocks_${d}_w1.jsonl ] || die "$sys_ $d $k"
done; done; done
touch $STOP
touch $R/DONE
log "dv done"
