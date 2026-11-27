#!/usr/bin/env bash
# portfolio 2026-11-27 (change 55): pf_pipe.sh + ROLL_ITEMS / ROLL_VIEWS (rollout data with items_<d>_<k>.jsonl, k 0-3) + FITX (extra fit args)
# + waits for $REF/dev/RUNS_DONE before the analysis. Was: generic method pipeline on the card pool (lib_pool.sh).
#  env: TAG; ONLINE (Thinker module, default hsc22); XF (extra Thinker flags, every call); ROLL=1 -> N-policy rollouts on the h1 shards
#  with this method's Thinker (en2zh heal@0.75 / zh2en base@0.75 prompts) + FIT (hsmm_anchor --no-gate | hsmm_ctr) -> $R/h/params.json,
#  else PRM (default h14); THINKER (default wave17); GRID (default "0.75 0.70 0.65 0.60"); DEVSETS (default "edev cvdev");
#  CVDEV_DATA (cvdev items dir override, C2); NOANALYZE=1 (B0); TEST=0 skips the test; WAIT (file).
#  dev: <devset> x <dir> x 3 shards -> $R/dev/<devset>/run_<d>_<k>; pf_analyze vs REF (default pfb0) -> $R/dev/choice.json;
#  test: experiments/sst_hsmm_v1/p3$TAG_report.sh (make_pf_report.sh) regardless of the dev verdict. Refuses existing outputs.
#   TAG=.. [..] setsid nohup bash experiments/sst_hsmm_v1/pf_pipe.sh > logs/sst_hsmm_v1/$TAG.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
: "${TAG:?}"
ONLINE=${ONLINE:-experiments.sst_hsmm_v1.hsc22_si_thinker}
XF=${XF:-}
ROLL=${ROLL:-0}
FIT=${FIT:-none}
D0=eval_outputs/sst_hsmm_v1
PRM=${PRM:-$D0/h14/params.json}
THINKER=${THINKER:-eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05}
GRID=${GRID:-"0.75 0.70 0.65 0.60"}
DEVSETS=${DEVSETS:-"edev cvdev"}
TEST=${TEST:-1}
REF=${REF:-$D0/pfb0}
R=$D0/$TAG
H=$R/h
SH=${ROLL_ITEMS:-$D0/h1/run/data}
E1=$D0/e1
VER=$D0/v/lora/a1.0000
REP=experiments/sst_hsmm_v1/p3${TAG}_report.sh
LOGD=logs/sst_hsmm_v1/$TAG
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
source experiments/sst_hsmm_v1/lib_pool.sh
[ "$TEST" = 0 ] || [ -f "$REP" ] || { echo "missing $REP" >&2; exit 1; }
[ "$ROLL" = 0 ] || [ "$FIT" = hsmm_anchor ] || [ "$FIT" = hsmm_ctr ] || [ "$FIT" = hsmm_map ] || { echo "ROLL=1 needs FIT" >&2; exit 1; }
if [ -z "$DRY" ]; then
  for x in $R $D0/p3$TAG $LOGD; do [ -e $x ] && { echo "exists: $x (refuse to overwrite)" >&2; exit 1; }; done
  mkdir -p $LOGD $R
  if [ -n "${WAIT:-}" ]; then
    log "$TAG $$: waiting for $WAIT"
    until [ -e "$WAIT" ]; do
      for f in FAILED NOGO DIVERGED; do [ -e "$(dirname "$WAIT")/$f" ] && { log "upstream ended with $f; $TAG stops"; touch $R/FAILED; exit 0; }; done
      sleep 60
    done
  fi
  git rev-parse HEAD > $R/commit.txt
  echo "ONLINE=$ONLINE XF=$XF ROLL=$ROLL FIT=$FIT PRM=$PRM THINKER=$THINKER GRID=$GRID DEVSETS=$DEVSETS CVDEV_DATA=${CVDEV_DATA:-}" > $R/config.txt
  mkdir -p $H
  start_watcher
fi
vmode() { [ "$1" = en2zh ] && echo heal || echo base; }
# 1. rollouts + fit
if [ "$ROLL" = 1 ]; then
  ps_=(); g=0
  for d in en2zh zh2en; do m=$(vmode $d); for k in 0 1 2 3; do
    ps_+=("$(launch $g roll_${d}_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m $ONLINE --mode rollout --variant n2 --samples 0 --theta 0.75 \
      --variants $m@0.75 --verifier $VER --second-view enh --views ${ROLL_VIEWS:-$E1/views}/$d --thinker $THINKER --direction $d --gpu-mem 0.6 $XF \
      --items-file $SH/items_${d}_$k.jsonl --out $H/roll_${d}_$k)")
    g=$(( (g + 1) % 8 )); [ -n "$DRY" ] || sleep 30
  done; done
  waitp "${ps_[@]}"
  if [ -z "$DRY" ]; then
    for d in en2zh zh2en; do m=$(vmode $d); for k in 0 1 2 3; do
      [ -s $H/roll_${d}_$k/blocks_${d}_${m}_0.75.jsonl ] || die "roll $d $k"
      ln -s blocks_${d}_${m}_0.75.jsonl $H/roll_${d}_$k/blocks_${d}_w1.jsonl
    done; done
    log "rollouts done; fitting $FIT (CPU)"
    NG=""; [ "$FIT" = hsmm_anchor ] && NG=--no-gate
    OMP_NUM_THREADS=32 PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.$FIT fit --roots "$H/roll_*" --pre-out $H/precheck.json --out $H/params.json $NG ${FITX:-} > $LOGD/fit.log 2>&1 || die "fit"
    [ -s $H/params.json ] || die "params"
    log "fit: $(grep -E 'precheck|_80 ' $LOGD/fit.log | cut -c1-160 | tr '\n' ' ')"
  fi
  PRM=$H/params.json
fi
# 2. dev
for ds in $DEVSETS; do
  case $ds in edev) DI=$D0/edev/data;; cvdev) DI=${CVDEV_DATA:-$D0/cvdev/data};; *) die "devset $ds";; esac
  [ -n "$DRY" ] && continue
  mkdir -p $R/dev/$ds/data
  for d in en2zh zh2en; do
    $PYQ -I - $DI/items_$d.jsonl $R/dev/$ds/data $d <<'PY' || die "split $ds $d"
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
done
ps_=(); g=0
for ds in $DEVSETS; do
  case $ds in edev) VW=$D0/edev/views;; cvdev) VW=eval_outputs/mist_w3po_v1/wave15/views;; esac
  for d in en2zh zh2en; do
    m=$(vmode $d); V=""; for x in $GRID; do V="$V,$m@$x"; done; V=${V#,}
    for k in 0 1 2; do
      ps_+=("$(launch $g dev_${ds}_${d}_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g $PYV -m $ONLINE --mode hsc --variant n2 --samples 0 \
        --params $PRM --theta 0.75 --verifier $VER --second-view enh --views $VW/$d --thinker $THINKER --direction $d --gpu-mem 0.6 \
        --variants "$V" $XF --items-file $R/dev/$ds/data/items_${d}_$k.jsonl --out $R/dev/$ds/run_${d}_$k)")
      g=$(( (g + 1) % 8 )); [ -n "$DRY" ] || sleep 30
    done
  done
done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
last=${GRID##* }
for ds in $DEVSETS; do for d in en2zh zh2en; do m=$(vmode $d); for k in 0 1 2; do
  [ -s $R/dev/$ds/run_${d}_$k/blocks_${d}_${m}_$last.jsonl ] || die "dev $ds $d $k"
done; done; done
touch $R/dev/RUNS_DONE
if [ -z "${NOANALYZE:-}" ]; then
  until [ -e $REF/dev/RUNS_DONE ]; do [ -e $REF/FAILED ] && die "reference failed"; sleep 60; done
  IT=""; [ -n "${CVDEV_DATA:-}" ] && IT="--items {\"cvdev\":\"$CVDEV_DATA\"}"
  PYTHONPATH=. $PYE -m experiments.sst_hsmm_v1.pf_analyze --root $R --ref $REF --out $R/dev/choice.json $IT > $LOGD/analyze.log 2>&1 || die "analyze"
  log "dev done: $(python3 -c "import json;c=json.load(open('$R/dev/choice.json'));print({d:(c[d]['spec'],c[d]['screen_effective']) for d in c})")"
fi
touch $R/dev/DONE
# 3. test (regardless of the dev verdict)
if [ "$TEST" = 1 ]; then
  bash $REP > logs/sst_hsmm_v1/p3${TAG}_report.log 2>&1 < /dev/null 8>&- 9>&-
  log "p3$TAG rc=$?"
fi
touch $STOP
touch $R/DONE
