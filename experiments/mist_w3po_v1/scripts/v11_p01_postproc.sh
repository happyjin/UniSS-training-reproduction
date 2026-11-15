#!/usr/bin/env bash
# v11 P0-1 post-processing (prereg reports/mist_w3po_v1/v11_p01_factorial_prereg.zh-CN.md §4), the post-review v10
# analysis inputs for the factorial rows, read-only on finished speech runs:
#   lat     lat_units_dump (LATU_MODE perword / blockstart) with exactly e2_baseline_plem.sh lat()'s arguments
#           -> $A/lat/<mode>/<set>_<row>_s<k>.json(.units.jsonl)          (CPU)
#   pauses  talker_block_pauses over every row's two talker dirs -> $A/pauses/block_pauses_<set>.jsonl   (CPU)
#   quality talker_quality (UTMOS, WavLM-SV) on TTS seed 1 -> $A/quality/<run>/quality.json   (GPU, ~1.7 GiB x 2,
#           TORCH_HOME = the local hub copy (no download); on the card with the most free memory once used + 4 GiB <= 135 GiB; re-read nvidia-smi before each start)
# Existing outputs are kept (each step skipped when its file exists); nothing is overwritten.
#   SETS="devz deve" ROWS_Z="R0:__t1c R1:__rlt1c ..." ROWS_E="..." [DRY_RUN=1] bash v11_p01_postproc.sh
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
A=$JO/v11f_analysis
[ -n "${DRY_RUN:-}" ] && A=/tmp/claude-0/-opt-dlami-nvme-jasonleeeli-projects-UniSS/25116ace-4ea7-4668-a301-0c0d9f82e950/scratchpad/v11f_dry   # dry: no file in the real tree
SETS=${SETS:?}
ROWS_Z=${ROWS_Z:?}
ROWS_E=${ROWS_E:?}
RT1=joint_v8h6k_20261113T073658Z_s06000
setvars() {
  case $1 in
    devz) N=jdev_bz_$RT1;   D=zh2en; WIN=$WZ_DEV; CW=$WZ_DEVC; REF="";      DR="";    SP=dev;  IT=zh2en;;
    deve) N=jdev_be_$RT1;   D=en2zh; WIN=$WE_DEV; CW=$WE_DEVC; REF=$RE_DEV;  DR="";    SP=dev;  IT=en2zh;;
    ami)  N=jtest_bami_$RT1; D=zh2en; WIN=$WT_AMI; CW=$WT_AMIC; REF=$RT_AMI; DR=zh2en; SP=test; IT=zh2en;;
    ali)  N=jtest_bali_$RT1; D=zh2en; WIN=$WT_ALI; CW=$WT_ALIC; REF=$RT_ALI; DR=ali;   SP=test; IT=zh2en;;
    en)   N=jtest_ben_$RT1;  D=en2zh; WIN=$WT_EN;  CW=$WT_ENC;  REF=$RE_TEST; DR="";   SP=test; IT=en2zh;;
    *) echo "bad set $1"; exit 1;;
  esac
}
run() { if [ -n "${DRY_RUN:-}" ]; then echo "DRY: $*" | cut -c1-400; else "$@"; fi; }
latu() {   # <set> <row> <run> <k> <mode>
  local s=$1 row=$2 n=$3 k=$4 mode=$5 f
  setvars $s
  f=$A/lat/$mode/${s}_${row}_s$k.json
  [ -e $f ] && return 0
  mkdir -p $A/lat/$mode
  if [ $D = zh2en ] && [ $SP = dev ]; then
    run env LATU_MODE=$mode CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.lat_units_dump \
      experiments.mist_w3po_v1.evaluation.speech_latency_lit --run $DEC/$n --talker $SPD/$n/talker --win $CW --out $f
  elif [ $D = zh2en ]; then
    run env LATU_MODE=$mode CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.lat_units_dump \
      experiments.mist_w3po_v1.evaluation.on_refset --ref-win $REF --dr $DR \
      experiments.mist_w3po_v1.evaluation.speech_latency_lit --run $DEC/$n --talker $SPD/$n/talker --win $CW --out $f
  else
    run env LATU_MODE=$mode CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.lat_units_dump \
      experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win $REF --work $f.work --run $DEC/$n \
      --talker $SPD/$n/talker --win $CW --out $f
  fi > $f.log 2>&1 || log "lat_units_dump failed: $f"
}
pick_card() {   # card with the most free memory once used + 4 GiB <= 135 GiB; empty if none
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F', ' -v cap=$CAP_MIB '$2 + 4096 <= cap {print $2, $1}' | sort -n | head -1 | awk '{print $2}'
}
qual() {   # <set> <run>
  local s=$1 n=$2 o c
  setvars $s
  o=$A/quality/$n/quality.json
  [ -e $o ] && return 0
  if [ -n "${DRY_RUN:-}" ]; then echo "DRY: talker_quality --talker $SPD/$n/talker --items $SPD/$n/talker_in/items.jsonl --out $o"; return 0; fi
  until c=$(pick_card); [ -n "$c" ]; do sleep 60; done
  mkdir -p $A/quality/$n
  TORCH_HOME=/opt/dlami/nvme/jasonleeeli/cache/torch CUDA_VISIBLE_DEVICES=$c $PY -m experiments.mist_w3po_v1.talker.talker_quality --talker $SPD/$n/talker \
    --items $SPD/$n/talker_in/items.jsonl --out $o > $A/quality/$n/quality.$(date -u +%s).log 2>&1 || log "quality failed: $n (card $c)"
}
for s in $SETS; do
  setvars $s
  case $s in devz|ami|ali) RS=$ROWS_Z;; *) RS=$ROWS_E;; esac
  dirs=()
  for rs in $RS; do
    row=${rs%%:*}; sfx=${rs#*:}; n=$N$sfx
    for k in 1 2; do
      [ $k = 1 ] && r=$n || r=${n}_tts2
      [ -s $SPD/$r/talker/results.jsonl ] || { log "missing talker results: $r"; continue; }
      dirs+=($SPD/$r/talker)
      for mode in perword blockstart; do latu $s $row $r $k $mode; done
    done
    qual $s $n
  done
  P=$A/pauses/block_pauses_$s.jsonl
  if [ ! -e $P ]; then
    mkdir -p $A/pauses
    run $PY -m experiments.mist_w3po_v1.evaluation.talker_block_pauses $(printf -- '--talker %s ' "${dirs[@]}") > $P.tmp 2> $P.log \
      && [ -z "${DRY_RUN:-}" ] && command mv -n $P.tmp $P
  fi
  log "postproc $s done"
done
log "V11F_POSTPROC_DONE $SETS"
