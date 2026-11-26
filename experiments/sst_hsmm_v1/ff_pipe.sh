#!/usr/bin/env bash
# change 48 (S2 faithful finalisation): edev hsc17 (wave17 engine, --draft-lora $DL on non-final reads, params $PRM):
#  en2zh heal@0.75..0.60 (cards 0-2), zh2en base@0.75..0.60 (cards 3-5) -> $R/dev ; drafter_analyze -> $R/dev/choice.json ;
#  then test (regardless) experiments/sst_hsmm_v1/p3${TAG}_report.sh. env: TAG DL PRM, optional WAIT (file). Refuses existing outputs.
#   TAG=.. DL=.. PRM=.. setsid nohup bash experiments/sst_hsmm_v1/ff_pipe.sh > logs/sst_hsmm_v1/$TAG.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
: "${TAG:?}" "${DL:?}" "${PRM:?}"
D0=eval_outputs/sst_hsmm_v1
R=$D0/$TAG
NTH=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
VER=$D0/v/lora/a1.0000
REP=experiments/sst_hsmm_v1/p3${TAG}_report.sh
LOGD=logs/sst_hsmm_v1/$TAG
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
[ -f "$REP" ] && [ -s "$DL/adapter_config.json" ] && [ -s "$PRM" ] || { echo "missing $REP / $DL / $PRM" >&2; exit 1; }
if [ -z "$DRY" ]; then
  for x in $R $D0/p3$TAG; do [ -e $x ] && { echo "exists: $x (refuse to overwrite)" >&2; exit 1; }; done
  mkdir -p $LOGD
  if [ -n "${WAIT:-}" ]; then
    log "$TAG $$: waiting for $WAIT"
    until [ -e "$WAIT" ]; do sleep 60; done
  fi
  mkdir -p $R/dev/data
  git rev-parse HEAD > $R/commit.txt
  for d in en2zh zh2en; do
    $PYQ -I - $D0/edev/data/items_$d.jsonl $R/dev/data $d <<'PY' || die "split $d"
import sys
rows = [l for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
root, d, K = sys.argv[2], sys.argv[3], 3
open(f"{root}/items_{d}.jsonl", "x", encoding="utf-8").writelines(rows)
n = len(rows); b = [round(k * n / K) for k in range(K + 1)]
for k in range(K):
    with open(f"{root}/items_{d}_{k}.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(rows[b[k]:b[k + 1]])
PY
  done
  start_watcher
fi
VE="heal@0.75,heal@0.70,heal@0.65,heal@0.60"
VB="base@0.75,base@0.70,base@0.65,base@0.60"
ps_=(); g=0
for d in en2zh zh2en; do
  if [ $d = en2zh ]; then V=$VE; else V=$VB; fi
  for k in 0 1 2; do
    ps_+=("$(launch $g dev_${d}_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m experiments.sst_hsmm_v1.hsc17_si_thinker --mode hsc --variant n2 --samples 0 \
      --params $PRM --draft-lora $DL --theta 0.75 --verifier $VER --second-view enh --views $D0/edev/views/$d --thinker $NTH \
      --direction $d --gpu-mem 0.6 --variants "$V" --items-file $R/dev/data/items_${d}_$k.jsonl --out $R/dev/run_${d}_$k)")
    g=$((g + 1)); [ -n "$DRY" ] || sleep 40
  done
done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
touch $STOP
for k in 0 1 2; do [ -s $R/dev/run_en2zh_$k/blocks_en2zh_heal_0.60.jsonl ] && [ -s $R/dev/run_zh2en_$k/blocks_zh2en_base_0.60.jsonl ] || die "dev $k"; done
PYTHONPATH=. $PYE -m experiments.sst_hsmm_v1.drafter_analyze --root $R/dev --out $R/dev/choice.json > $LOGD/analyze.log 2>&1 || die "analyze"
log "dev done: $(python3 -c "import json;c=json.load(open('$R/dev/choice.json'));print({d:(c[d]['spec'],c[d]['effective']) for d in c})")"
touch $R/dev/DONE
bash $REP > logs/sst_hsmm_v1/p3${TAG}_report.log 2>&1 < /dev/null 8>&- 9>&-
log "p3$TAG rc=$?"
