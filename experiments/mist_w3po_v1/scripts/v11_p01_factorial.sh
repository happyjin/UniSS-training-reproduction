#!/usr/bin/env bash
# v11 P0-1 PA factorial ablation (docs/mist_v11_plan.zh-CN.md §3; prereg reports/mist_w3po_v1/v11_p01_factorial_prereg.zh-CN.md;
# user away 2026-11-15, best-option decisions delegated): the deadline baseline B's existing read-merged text
# (<text>_rl, made by e2_baseline_plem.sh) spoken by each speech-layer variant through v11_factorial_orch.py
# (joint_orch.py unchanged + a talker-command rewrite), both TTS seeds, then the chunk-protocol speech latency exactly as
# e2_baseline_plem.sh's lat(). Nothing is decoded on the text side; R0 (B, t1c) and R5 (E2 rlplem) are the existing runs.
#   SETS="devz deve" | "ami ali en"   (one phase per call; test only after the prereg is committed)
#   VARS_Z / VARS_E: variants for the zh2en sets (devz ami ali) / the en2zh sets (deve en)
# Run dirs logs/mist_w3po_v1/joint/v11f/<set>/<variant>; speech/decode dirs <text>__rl<variant>[_tts2];
# marker logs/mist_w3po_v1/joint/v11f/DONE_<PHASE>. Refuses to overwrite an existing run dir or marker.
#   PHASE=dev SETS="devz deve" VARS_Z="t1c c1s2 c1b1s2 c1b1s5 c1b1s8 plemk1" VARS_E="..." [PORT0=20000] [DRY_RUN=1] \
#     bash experiments/mist_w3po_v1/scripts/v11_p01_factorial.sh
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/joint/v11f/<phase>.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
PORT0=${PORT0:-20000}
PHASE=${PHASE:?PHASE=dev|test}
SETS=${SETS:?}
VARS_Z=${VARS_Z:?}
VARS_E=${VARS_E:?}
VD=$JL/v11f
DONE_M=$VD/DONE_$PHASE
B2=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/thinker_merged
RT1=joint_v8h6k_20261113T073658Z_s06000
FORCH="$PY $SC/v11_factorial_orch.py"
flag() { ( set -o noclobber; echo "$(date -u +%FT%TZ) $*" > $VD/FAILED_${PHASE}_$(date -u +%s%N) ); log "V11F_STOP: $*"; }
setvars() {   # as e2_baseline_plem.sh
  case $1 in
    devz) N=jdev_bz_$RT1;   D=zh2en; K=dev;  WIN=$WZ_DEV; CW=$WZ_DEVC; REF="";      DR="";    SP=dev;;
    deve) N=jdev_be_$RT1;   D=en2zh; K=dev;  WIN=$WE_DEV; CW=$WE_DEVC; REF=$RE_DEV;  DR="";    SP=dev;;
    ami)  N=jtest_bami_$RT1; D=zh2en; K=test; WIN=$WT_AMI; CW=$WT_AMIC; REF=$RT_AMI; DR=zh2en; SP=test;;
    ali)  N=jtest_bali_$RT1; D=zh2en; K=ali;  WIN=$WT_ALI; CW=$WT_ALIC; REF=$RT_ALI; DR=ali;   SP=test;;
    en)   N=jtest_ben_$RT1;  D=en2zh; K=test; WIN=$WT_EN;  CW=$WT_ENC;  REF=$RE_TEST; DR="";   SP=test;;
    *) echo "bad set $1"; exit 1;;
  esac
}
lat() {   # <set> <speech run name>: e2_baseline_plem.sh's lat(), verbatim
  local s=$1 n=$2 O f
  setvars $s
  O=$TE/${n%_tts2}; mkdir -p $O; f=$O/lat_chunk_$n.json
  [ -s $f ] && return 0
  if [ $D = zh2en ] && [ $SP = dev ]; then
    CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.speech_latency_lit --run $DEC/$n --talker $SPD/$n/talker \
      --win $CW --out $f > $f.log 2>&1 || log "chunk LAAL failed: $n"
  elif [ $D = zh2en ]; then
    CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.on_refset --ref-win $REF --dr $DR \
      experiments.mist_w3po_v1.evaluation.speech_latency_lit --run $DEC/$n --talker $SPD/$n/talker --win $CW --out $f > $f.log 2>&1 || log "chunk LAAL failed: $n"
  else
    CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win $REF --work $f.work --run $DEC/$n \
      --talker $SPD/$n/talker --win $CW --out $f > $f.log 2>&1 || log "chunk LAAL failed: $n"
  fi
}
chain() {   # <set> <variant> <port0>
  local s=$1 v=$2 p=$3 pe=""
  setvars $s
  local R=$VD/$s/$v nm=${N}__rl$v
  if [ $D = en2zh ] && [ $v = plemk1 ]; then
    local tx; tx=$(en2zh_talker)
    [ -n "$tx" ] && [ -e $tx/uniss/model.safetensors ] || { flag "$nm: no En->Zh PL-EM export ('$tx')"; return 1; }
    pe="--plem-export $tx"
  fi
  local C="$FORCH --variant $v --dir $D --set $K --base $B2 --run $R --name $nm --ckpt - --speech-from ${N}_rl $pe --window 1 --cap135 --scorer-port0 $p"
  if [ -n "${DRY_RUN:-}" ]; then
    echo "$s/$v: $C"; [ -e $DEC/${N}_rl/score_text.json ] && echo "  rl text ok" || echo "  RL TEXT MISSING: ${N}_rl"
    $C --dry 2>&1 | tail -1; return 0
  fi
  [ -e $R ] && { flag "$R exists (not overwritten)"; return 1; }
  [ -e $DEC/${N}_rl/score_text.json ] || { flag "no read-merged text ${N}_rl"; return 1; }
  mkdir -p $VD/$s
  setsid nohup $C < /dev/null > $R.launch.log 2>&1 9>&- &
  log "$nm orchestrator pid $!"
  wait_done $R; [ -z "$(orch_failed $R)" ] || { flag "$nm failed ($(tail -1 $R/orch.log | cut -c1-200))"; return 1; }
  for x in $nm ${nm}_tts2; do lat $s $x; done
  log "$nm done"
}
if [ -z "${DRY_RUN:-}" ]; then
  [ -e $DONE_M ] && { log "拒绝覆盖已有输出: $DONE_M"; exit 1; }
  mkdir -p $VD
fi
log "V11F $PHASE: sets [$SETS] zh2en vars [$VARS_Z] en2zh vars [$VARS_E]"
PIDS=""; k=0
for s in $SETS; do
  case $s in devz|ami|ali) VS=$VARS_Z;; *) VS=$VARS_E;; esac
  for v in $VS; do
    if [ -n "${DRY_RUN:-}" ]; then chain $s $v $((PORT0 + 10 * k)); k=$((k + 1)); continue; fi
    ( chain $s $v $((PORT0 + 10 * k)) ) &
    PIDS="$PIDS $!"; k=$((k + 1))
    sleep 90
  done
done
[ -n "${DRY_RUN:-}" ] && exit 0
rc=0; for p in $PIDS; do wait $p || rc=1; done
[ $rc = 0 ] && ! ls $VD/FAILED_${PHASE}_* > /dev/null 2>&1 || { log "V11F_STOP: a chain failed ($(cat $VD/FAILED_${PHASE}_* 2>/dev/null | tr '\n' ' ' | cut -c1-300))"; exit 1; }
( set -o noclobber; echo "utc $(date -u +%FT%TZ)" > $DONE_M ) || log "marker exists"
log "V11F_${PHASE}_DONE"
