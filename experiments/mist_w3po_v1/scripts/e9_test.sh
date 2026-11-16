#!/usr/bin/env bash
# E9 cascade baseline, TEST, run once (reports/mist_w3po_v1/multispk_s2st_baselines_prereg_v2.zh-CN.md §1 S3, §4;
# results reports/mist_w3po_v1/e9_cascade_multispk_results.zh-CN.md). Recognition-then-attribution cascade with MIST's
# training data: the v8 speaker-blind Simul model of each direction (thinker_gstar_s2dlnosd_{zh2en,en2zh}_v8, 3000
# iterations, speaker stream zeroed) decodes the online test windows -> every segment gets a speaker by the v8 causal
# relabelling rule -> B's base Talker (T1c, B's flags, two TTS seeds) speaks it -> scored as B.
# Per set (ami = AMI Zh->En, ali = AliMeeting Zh->En, en = AMI En->Zh), exactly joint_test.sh's BASE_TALKER=t1c chain with
# e9_orch.py in place of joint_orch.py and one CPU step between decode and speech:
#   1. decode   e9_orch.py --plain-thinker --ckpt - --talker none (B's decode command, zero-sd module, blind model;
#               8 shards by meeting, --cap135, cards 4-7 only: --exclude 0 1 2 3)       -> decode/jtest_<s>_cascade_blind_s03000
#   2. relabel  relabel_causal_run (overlap_score.relabel_causal, unchanged) -> <name>_rel; score_run on it (text
#               tcpBLEU all / ov / 1spk); chunk-protocol text LAAL (joint_test.sh lat)
#   3. speech   e9_orch.py --speech-from <name>_rel --talker t1c (B's T1c chain, both seeds)  -> <name>_rel__t1c[_tts2]
#               (ASR Whisper-large-v3 / Paraformer-zh, asr_hyp, score_run --asr = tcpASR-BLEU)
#   4. latency  chunk-protocol speech latency per seed (joint_test.sh lat) and Table 1's latency units (table1_lat.sh:
#               LAAL_block / LAAL_spread / StartOffset / EndOffset)
# The three chains run in parallel, started 120 s apart (as joint_test.sh). Line E9_TEST_DONE at the end; the summary is
# e9_summary.py's. Every tool refuses to overwrite; an existing run dir stops that chain.
#   [DRY_RUN=1] [SETS="ami ali en"] bash experiments/mist_w3po_v1/scripts/e9_test.sh
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/e9_cascade/test.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
E9="$PY experiments/mist_w3po_v1/scripts/e9_orch.py"
TV=logs/mist_w3po_v1/e9_cascade
NZ=eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dlnosd_zh2en_v8/thinker_merged
NE=eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dlnosd_en2zh_v8/thinker_merged
STEP=s03000
XC="--exclude 0 1 2 3"
SETS=${SETS:-"ami ali en"}
flag() { ( set -o noclobber; echo "$(date -u +%FT%TZ) $*" > $TV/FAILED_$(date -u +%s%N) ); log "E9_STOP: $*"; }
setvars() {   # <set> -> D K WIN CW REF DR SS B (joint_test.sh setvars + score set + blind model)
  case $1 in
    ami) D=zh2en; K=test; WIN=$WT_AMI; CW=$WT_AMIC; REF=$RT_AMI; DR=zh2en; SS=zh2en; B=$NZ;;
    ali) D=zh2en; K=ali;  WIN=$WT_ALI; CW=$WT_ALIC; REF=$RT_ALI; DR=ali;   SS=ali;   B=$NZ;;
    en)  D=en2zh; K=test; WIN=$WT_EN;  CW=$WT_ENC;  REF=$RE_TEST; DR="";   SS=en2zh; B=$NE;;
  esac
}
lat() {   # joint_test.sh lat, verbatim: <set> <run name> [<talker dir>] -> chunk-protocol LAAL json
  local s=$1 n=$2 tk=${3:-} O f
  setvars $s
  O=$TE/${n%_tts2}; mkdir -p $O
  if [ -z "$tk" ]; then f=$O/textlat_chunk_$n.json; A="--text-only"; else f=$O/lat_chunk_$n.json; A="--talker $tk"; fi
  if [ $D = zh2en ]; then
    CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.on_refset --ref-win $REF --dr $DR \
      experiments.mist_w3po_v1.evaluation.speech_latency_lit --run $DEC/$n $A --win $CW --out $f > $f.log 2>&1 || log "chunk LAAL failed: $n"
  else
    CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win $REF --work $f.work --run $DEC/$n $A \
      --win $CW --out $f > $f.log 2>&1 || log "chunk LAAL failed: $n"
  fi
}
chain() {   # <set>
  local s=$1 n r sp C1 C2
  setvars $s
  n=jtest_${s}_cascade_blind_$STEP; r=${n}_rel; sp=${r}__t1c
  C1="$E9 --dir $D --set $K --base $B --run $TV/$s --name $n --ckpt - --plain-thinker --talker none --window 1 --cap135 $XC"
  C2="$E9 --dir $D --set $K --base $B --run $TV/${s}_sp --name $sp --ckpt - --speech-from $r --talker t1c --window 1 --cap135 $XC"
  if [ -n "${DRY_RUN:-}" ]; then
    echo "$s decode: $C1"; $C1 --dry | tail -1
    echo "$s relabel: $PY -m experiments.mist_w3po_v1.evaluation.relabel_causal_run --src $n --windows $WIN"
    echo "$s score: $PY -m experiments.mist_w3po_v1.evaluation.score_run --run $DEC/$r --windows $WIN --ref-windows $REF --set $SS"
    echo "$s speech: $C2"; $C2 --dry | tail -1
    for x in $TV/$s $TV/${s}_sp $DEC/$n $DEC/$r $DEC/$sp $DEC/${sp}_tts2 $SPD/$sp $SPD/${sp}_tts2; do
      [ -e $x ] && echo "EXISTS (must not): $x"; done
    return 0
  fi
  for x in $TV/$s $TV/${s}_sp $DEC/$n $DEC/$r $DEC/$sp $DEC/${sp}_tts2 $SPD/$sp $SPD/${sp}_tts2; do
    [ -e $x ] && { flag "$s: 拒绝覆盖已有输出 $x"; return 1; }; done
  setsid nohup $C1 < /dev/null > $TV/$s.launch.log 2>&1 &
  log "$n decode orchestrator pid $!"
  sleep 20; [ -e $TV/$s/orch.log ] || { flag "$n decode orchestrator did not start ($(tail -2 $TV/$s.launch.log | tr '\n' ' ' | cut -c1-300))"; return 1; }
  wait_done $TV/$s; [ -z "$(orch_failed $TV/$s)" ] || { flag "$n decode failed ($(tail -1 $TV/$s/orch.log | cut -c1-200))"; return 1; }
  log "$n decoded: $(tail -1 $DEC/$n/score.log 2>/dev/null | cut -c1-200)"
  CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.relabel_causal_run --src $n --windows $WIN \
    > $TV/relabel_$s.log 2>&1 || { flag "$n relabel failed ($(tail -1 $TV/relabel_$s.log | cut -c1-200))"; return 1; }
  CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.score_run --run $DEC/$r --windows $WIN --ref-windows $REF \
    --set $SS > $DEC/$r/score.log 2>&1 || { flag "$r score failed"; return 1; }
  log "$r: $(tail -1 $DEC/$r/score.log | cut -c1-200)"
  lat $s $r
  setsid nohup $C2 < /dev/null > $TV/${s}_sp.launch.log 2>&1 &
  log "$sp speech orchestrator pid $!"
  sleep 20; [ -e $TV/${s}_sp/orch.log ] || { flag "$sp speech orchestrator did not start ($(tail -2 $TV/${s}_sp.launch.log | tr '\n' ' ' | cut -c1-300))"; return 1; }
  wait_done $TV/${s}_sp; [ -z "$(orch_failed $TV/${s}_sp)" ] || { flag "$sp speech failed ($(tail -1 $TV/${s}_sp/orch.log | cut -c1-200))"; return 1; }
  for x in $sp ${sp}_tts2; do lat $s $x $SPD/$x/talker; done
  bash experiments/mist_w3po_v1/scripts/table1_lat.sh $s $sp ${sp}_tts2 > $TV/table1_lat_$s.log 2>&1 || log "table1_lat $s rc=$?"
  log "$s done: $sp $(grep -h '^asr' $DEC/$sp/score_asr.log $DEC/${sp}_tts2/score_asr.log 2>/dev/null | cut -c1-120 | tr '\n' ' ')"
}
if [ -n "${DRY_RUN:-}" ]; then
  for s in $SETS; do chain $s; done
  exit 0
fi
mkdir -p $TV
log "E9 TEST once: sets $SETS; blind models $NZ / $NE; cards 4-7"
PIDS=""
for s in $SETS; do
  ( chain $s ) &
  PIDS="$PIDS $!"
  sleep 120
done
rc=0; for p in $PIDS; do wait $p || rc=1; done
if [ $rc = 0 ] && ! ls $TV/FAILED_* > /dev/null 2>&1; then log "E9_TEST_DONE"
else log "E9_STOP: a chain failed ($(cat $TV/FAILED_* 2>/dev/null | tr '\n' ' ' | cut -c1-300))"; fi
