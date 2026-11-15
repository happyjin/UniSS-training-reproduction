#!/usr/bin/env bash
# v11 P0-4 post-review metrics of the rule-selected step 4000 test runs (wb_test_s4000.sh), as post_review_v10 did for
# B / E2 / M0 / M: waits for WB_TEST_S4000_DONE, then per set (ami / ali / en) and TTS seed
#   lat     lat_units_dump (LATU_MODE perword / blockstart) with exactly wb_test.sh lat()'s arguments
#           -> $A/lat/<mode>/<set>_m4_s<k>.json(.units.jsonl)                                   (CPU)
#   pauses  talker_block_pauses over the two talker dirs of the set -> $A/pauses/block_pauses_<set>.jsonl   (CPU)
#   quality talker_quality (UTMOS, WavLM-SV) on seed 1 -> $A/quality/<run>/quality.json  (GPU ~2 GiB, card with the most
#           free memory once used + 4 GiB <= 135 GiB; nvidia-smi re-read before each start)
# then evaluation/post_review_v11_s4000.py -> $A/analysis_v1.json. Marker $A/S4000_POSTPROC_DONE. Existing outputs kept.
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/s4000_postproc.sh
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
A=$JO/post_review_v11_s4000
RT=joint_v2p_20261113T120415Z_s04000_wbzm1em1
GATE=$JL/wb/WB_TEST_S4000_DONE
run() { if [ -n "${DRY_RUN:-}" ]; then echo "DRY: $*" | cut -c1-400; else "$@"; fi; }
latu() {   # <set> <k> <mode>
  local s=$1 k=$2 mode=$3 n f
  n=jtest_m${s}_${RT}__rlplem; [ $k = 2 ] && n=${n}_tts2
  f=$A/lat/$mode/${s}_m4_s$k.json
  [ -e $f ] && return 0
  [ -n "${DRY_RUN:-}" ] || mkdir -p $A/lat/$mode
  case $s in
    ami) run env LATU_MODE=$mode CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.lat_units_dump \
           experiments.mist_w3po_v1.evaluation.on_refset --ref-win $RT_AMI --dr zh2en \
           experiments.mist_w3po_v1.evaluation.speech_latency_lit --run $DEC/$n --talker $SPD/$n/talker --win $WT_AMIC --out $f;;
    ali) run env LATU_MODE=$mode CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.lat_units_dump \
           experiments.mist_w3po_v1.evaluation.on_refset --ref-win $RT_ALI --dr ali \
           experiments.mist_w3po_v1.evaluation.speech_latency_lit --run $DEC/$n --talker $SPD/$n/talker --win $WT_ALIC --out $f;;
    en)  run env LATU_MODE=$mode CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.lat_units_dump \
           experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win $RE_TEST --work $f.work --run $DEC/$n \
           --talker $SPD/$n/talker --win $WT_ENC --out $f;;
  esac
}
pick_card() {
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{ if ($2/1024 + 4 <= 135) print $3-$2, $1 }' | sort -rn | head -1 | awk '{print $2}'
}
quality() {   # <set>
  local n=jtest_m$1_${RT}__rlplem o c
  o=$A/quality/$n/quality.json
  [ -e $o ] && return 0
  if [ -n "${DRY_RUN:-}" ]; then echo "DRY: talker_quality --talker $SPD/$n/talker --items $SPD/$n/talker_in/items.jsonl --out $o"; return 0; fi
  until c=$(pick_card); [ -n "$c" ]; do sleep 60; done
  mkdir -p $A/quality/$n
  TORCH_HOME=/opt/dlami/nvme/jasonleeeli/cache/torch CUDA_VISIBLE_DEVICES=$c $PY -m experiments.mist_w3po_v1.talker.talker_quality --talker $SPD/$n/talker \
    --items $SPD/$n/talker_in/items.jsonl --out $o > $A/quality/$n/quality.$(date -u +%s).log 2>&1 || log "quality failed: $n (card $c)"
}
if [ -z "${DRY_RUN:-}" ]; then
  [ -e $A/analysis_v1.json ] && { log "拒绝覆盖已有输出: $A/analysis_v1.json"; exit 1; }
  until [ -e $GATE ]; do
    ls $JL/wb/test_s4000/FAILED_* > /dev/null 2>&1 && { log "S4000 test failed: no post-processing"; exit 1; }
    sleep 120
  done
  mkdir -p $A/pauses
fi
for s in ami ali en; do
  for k in 1 2; do for mode in perword blockstart; do latu $s $k $mode; done; done
  P=$A/pauses/block_pauses_$s.jsonl
  if [ ! -e $P ]; then
    n=jtest_m${s}_${RT}__rlplem
    run $PY -m experiments.mist_w3po_v1.evaluation.talker_block_pauses --talker $SPD/$n/talker --talker $SPD/${n}_tts2/talker \
      > $P.tmp 2> $P.log && [ -z "${DRY_RUN:-}" ] && command mv -n $P.tmp $P
  fi
  quality $s
done
run env CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.post_review_v11_s4000 --out $A/analysis_v1.json
[ -n "${DRY_RUN:-}" ] || ( set -o noclobber; date -u +%FT%TZ > $A/S4000_POSTPROC_DONE )
log "S4000_POSTPROC_DONE"
