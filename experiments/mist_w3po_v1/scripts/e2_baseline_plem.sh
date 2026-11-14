#!/usr/bin/env bash
# E2 of docs/mist_v10_plan.zh-CN.md (user 2026-11-14 ~21:45 Prague: "E2现在跑"): the strict joint baseline's Thinker
# (joint v8 SFT s6000, its existing text decodes jdev_b{z,e}_<RT1> / jtest_b{ami,ali,en}_<RT1>) with the method's speech
# layer: read-level merge (joint_lib.sh read_merge; zh2en ' ', en2zh '') -> PL-EM talker (zh2en: the existing zh2en
# PL-EM; En->Zh: (a)'s selected export), two TTS seeds -> chunk-protocol speech latency. Separates the Thinker's and the
# talker's share of the method's gain. Nothing is decoded again on the text side; nothing is chosen.
# Run dirs logs/mist_w3po_v1/joint/e2/{dev,test}/<tag>; names <text name>__rlplem[_tts2]; marker
# logs/mist_w3po_v1/joint/e2/E2_DONE. Refuses to overwrite an existing run dir / marker. All 8 cards, --cap135.
#   [PORT0=19800] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/e2_baseline_plem.sh
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/joint/e2/e2.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
PORT0=${PORT0:-19800}
E2D=$JL/e2
DONE_M=$E2D/E2_DONE
B2=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/thinker_merged
RT1=joint_v8h6k_20261113T073658Z_s06000
flag() { ( set -o noclobber; echo "$(date -u +%FT%TZ) $*" > $E2D/FAILED_$(date -u +%s%N) ); log "E2_STOP: $*"; }
setvars() {   # <set devz|deve|ami|ali|en> -> N (text run) D K WIN CW REF DR SEP SP (dev|test)
  case $1 in
    devz) N=jdev_bz_$RT1;   D=zh2en; K=dev;  WIN=$WZ_DEV; CW=$WZ_DEVC; REF="";      DR="";    SEP=' '; SP=dev;;
    deve) N=jdev_be_$RT1;   D=en2zh; K=dev;  WIN=$WE_DEV; CW=$WE_DEVC; REF=$RE_DEV;  DR="";    SEP='';  SP=dev;;
    ami)  N=jtest_bami_$RT1; D=zh2en; K=test; WIN=$WT_AMI; CW=$WT_AMIC; REF=$RT_AMI; DR=zh2en; SEP=' '; SP=test;;
    ali)  N=jtest_bali_$RT1; D=zh2en; K=ali;  WIN=$WT_ALI; CW=$WT_ALIC; REF=$RT_ALI; DR=ali;   SEP=' '; SP=test;;
    en)   N=jtest_ben_$RT1;  D=en2zh; K=test; WIN=$WT_EN;  CW=$WT_ENC;  REF=$RE_TEST; DR="";   SEP='';  SP=test;;
  esac
}
lat() {   # <set> <speech run name>: chunk-protocol speech latency (same tools as joint_dev.sh / joint_test.sh)
  local s=$1 n=$2 O f
  setvars $s
  O=$TE/${n%_tts2}; mkdir -p $O; f=$O/lat_chunk_$n.json
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
chain() {   # <set> <port0>
  local s=$1 p=$2 tx pe=""
  setvars $s
  local R=$E2D/$SP/b${s}_sp
  if [ $D = en2zh ]; then
    tx=$(en2zh_talker)
    [ -n "$tx" ] && [ -e $tx/uniss/model.safetensors ] || { flag "$N: no En->Zh PL-EM export ('$tx')"; return 1; }
    pe="--plem-export $tx"
  fi
  local C="$JORCH --dir $D --set $K --base $B2 --run $R --name ${N}__rlplem --ckpt - --speech-from ${N}_rl --talker plem $pe --window 1 --cap135 --scorer-port0 $p"
  if [ -n "${DRY_RUN:-}" ]; then echo "b${s}_sp: $C"; [ -e $DEC/$N/hyp.jsonl ] && echo "  text ok: $N" || echo "  TEXT MISSING: $N"; return 0; fi
  [ -e $R ] && { flag "$R exists (not overwritten)"; return 1; }
  read_merge $N "$SEP" $WIN $E2D/$SP/merge_$N.log || { flag "merge $N ($(tail -1 $E2D/$SP/merge_$N.log | cut -c1-200))"; return 1; }
  setsid nohup $C < /dev/null > $R.launch.log 2>&1 &
  log "${N}__rlplem speech orchestrator pid $!"
  wait_done $R; [ -z "$(orch_failed $R)" ] || { flag "${N}__rlplem speech failed ($(tail -1 $R/orch.log | cut -c1-200))"; return 1; }
  for x in ${N}__rlplem ${N}__rlplem_tts2; do lat $s $x; done
  log "${N}__rlplem done"
}
if [ -n "${DRY_RUN:-}" ]; then
  k=0; for s in devz deve ami ali en; do chain $s $((PORT0 + 10 * k)); k=$((k + 1)); done
  exit 0
fi
[ -e $DONE_M ] && { log "拒绝覆盖已有输出: $DONE_M"; exit 1; }
mkdir -p $E2D/dev $E2D/test
log "E2: strict baseline $RT1 text + rl + PL-EM talker (dev zh2en/en2zh, test ami/ali/en), En->Zh talker $(en2zh_talker)"
PIDS=""; k=0
for s in devz deve ami ali en; do
  ( chain $s $((PORT0 + 10 * k)) ) &
  PIDS="$PIDS $!"; k=$((k + 1))
  sleep 120
done
rc=0; for p in $PIDS; do wait $p || rc=1; done
[ $rc = 0 ] && ! ls $E2D/FAILED_* > /dev/null 2>&1 || { log "E2_STOP: a chain failed ($(cat $E2D/FAILED_* 2>/dev/null | tr '\n' ' ' | cut -c1-300))"; exit 1; }
( set -o noclobber; echo "$RT1 utc $(date -u +%FT%TZ)" > $DONE_M ) || log "marker exists"
log "E2_DONE"
