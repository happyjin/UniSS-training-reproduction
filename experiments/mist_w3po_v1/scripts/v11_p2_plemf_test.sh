#!/usr/bin/env bash
# v11 P2 test (reports/mist_w3po_v1/v11_p2_fluency_pa_prereg.zh-CN.md §5): only for the directions whose PA-F passed
# the dev gate (PASS, set by the caller from the dev gate_2seed files), each test set decoded ONCE with E2's rlplem
# chain and only the Talker swapped: zh2en PA-F -> AMI Zh->En (ami) and AliMeeting (ali); en2zh PA-F -> AMI En->Zh (en).
# B's test text (jtest_b<s>_<RT1>_rl), both TTS seeds, gate_2seed vs B+PA (descriptive), then perword / blockstart
# chunk latency, pauses, UTMOS/SIM seed 1 for PA-F and B+PA. Nothing is chosen on test. Marker
# logs/mist_v11/p2/P2_TEST_DONE. Refuses to overwrite. Checked with bash -n and DRY_RUN=1 only; never test-run.
#   PASS="zh2en en2zh" [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/v11_p2_plemf_test.sh
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
PASS=${PASS:?directions that passed the dev gate}
P2=logs/mist_v11/p2
B2=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/thinker_merged
RT1=joint_v8h6k_20261113T073658Z_s06000
QDIR=$P2/quality
[ -e $P2/P2_TEST_DONE ] && { log "拒绝覆盖已有输出: $P2/P2_TEST_DONE"; exit 1; }

setvars() {   # <set ami|ali|en>
  case $1 in
    ami) N=jtest_bami_$RT1; D=zh2en; K=test; CW=$WT_AMIC; REF=$RT_AMI; DR=zh2en;;
    ali) N=jtest_bali_$RT1; D=zh2en; K=ali;  CW=$WT_ALIC; REF=$RT_ALI; DR=ali;;
    en)  N=jtest_ben_$RT1;  D=en2zh; K=test; CW=$WT_ENC;  REF=$RE_TEST; DR="";;
  esac
}
lat_all() {   # <set> <run name>
  local s=$1 n=$2 O f m
  setvars $s
  O=$TE/${n%_tts2}; mkdir -p $O $P2/lat; f=$O/lat_chunk_$n.json
  if [ $D = zh2en ]; then
    [ -e $f ] || CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.on_refset --ref-win $REF --dr $DR \
      experiments.mist_w3po_v1.evaluation.speech_latency_lit --run $DEC/$n --talker $SPD/$n/talker --win $CW --out $f > $f.log 2>&1 || log "perword failed: $n"
    for m in perword blockstart; do
      [ -e $P2/lat/${m}_$n.json ] || LATU_MODE=$m CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.lat_units_dump \
        experiments.mist_w3po_v1.evaluation.on_refset --ref-win $REF --dr $DR experiments.mist_w3po_v1.evaluation.speech_latency_lit \
        --run $DEC/$n --talker $SPD/$n/talker --win $CW --out $P2/lat/${m}_$n.json > $P2/lat/${m}_$n.log 2>&1 || log "$m failed: $n"
    done
  else
    [ -e $f ] || CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win $REF --work $f.work --run $DEC/$n \
      --talker $SPD/$n/talker --win $CW --out $f > $f.log 2>&1 || log "perword failed: $n"
    for m in perword blockstart; do
      [ -e $P2/lat/${m}_$n.json ] || LATU_MODE=$m CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.lat_units_dump \
        experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win $REF --work $P2/lat/${m}_$n.json.work --run $DEC/$n \
        --talker $SPD/$n/talker --win $CW --out $P2/lat/${m}_$n.json > $P2/lat/${m}_$n.log 2>&1 || log "$m failed: $n"
    done
  fi
}
quality() {
  local n=$1 c
  [ -e $QDIR/$n/quality.json ] && return 0
  c=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $2 - $3, $1}' | sort -rn | head -n 1 | awk '{print $2}')
  mkdir -p $QDIR/$n
  CUDA_VISIBLE_DEVICES=$c $PY -m experiments.mist_w3po_v1.talker.talker_quality --talker $SPD/$n/talker \
    --items $SPD/$n/talker_in/items.jsonl --out $QDIR/$n/quality.json > $QDIR/$n/quality.log 2>&1 || log "quality failed: $n"
}
chain() {   # <set> <port0>
  local s=$1 p=$2 R C x
  setvars $s
  x=$(cat $P2/EXPORT_$D 2>/dev/null)
  if [ -z "$x" ] && [ -n "${DRY_RUN:-}" ]; then   # dry only: the original PA exports stand in
    [ $D = zh2en ] && x=eval_outputs/mist_w3po_v1/talker_crf/talker_plem_20261111T001759Z/export \
      || x=eval_outputs/mist_w3po_v1/talker_crf/talker_plem_en2zh_20261114T045019Z/export_i0400
  fi
  [ -n "$x" ] || { log "P2_TEST_STOP $s: no export for $D"; return 1; }
  R=$P2/test_$s
  if [ $D = zh2en ]; then
    C="$PY $SC/joint_orch_plemx.py --plem-x $x --dir zh2en --set $K --base $B2 --run $R --name ${N}__rlplemf --ckpt - --speech-from ${N}_rl --talker plem --window 1 --cap135 --scorer-port0 $p --gate-ref ${N}__rlplem ${N}__rlplem_tts2"
  else
    C="$JORCH --dir en2zh --set $K --base $B2 --run $R --name ${N}__rlplemf --ckpt - --speech-from ${N}_rl --talker plem --plem-export $x --window 1 --cap135 --scorer-port0 $p --gate-ref ${N}__rlplem ${N}__rlplem_tts2"
  fi
  if [ -n "${DRY_RUN:-}" ]; then echo "test $s: $C"; $C --dry 2>&1 | tail -n 1; return 0; fi
  [ -e $R ] && { log "P2_TEST_STOP $s: $R exists"; return 1; }
  setsid nohup $C < /dev/null > $R.launch.log 2>&1 9>&- &
  log "$s test orchestrator pid $!"
  wait_done $R
  [ -z "$(orch_failed $R)" ] || { log "P2_TEST_STOP $s: orchestrator failed ($(tail -n 1 $R/orch.log | cut -c1-200))"; return 1; }
  for n in ${N}__rlplemf ${N}__rlplemf_tts2 ${N}__rlplem ${N}__rlplem_tts2; do lat_all $s $n; done
  mkdir -p $P2/pauses
  for n in ${N}__rlplemf ${N}__rlplemf_tts2 ${N}__rlplem ${N}__rlplem_tts2; do
    [ -e $P2/pauses/$n.json ] || CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.talker_block_pauses \
      --talker $SPD/$n/talker > $P2/pauses/$n.json 2> $P2/pauses/$n.log || log "pauses failed: $n"
  done
  quality ${N}__rlplemf; quality ${N}__rlplem
  log "P2_TEST $s done"
}
SETS=""
for d in $PASS; do [ $d = zh2en ] && SETS="$SETS ami ali"; [ $d = en2zh ] && SETS="$SETS en"; done
if [ -n "${DRY_RUN:-}" ]; then k=0; for s in $SETS; do chain $s $((19990 + 10 * k)); k=$((k + 1)); done; exit 0; fi
log "v11 P2 test: PASS '$PASS' -> sets$SETS"
PIDS=""; k=0
for s in $SETS; do ( chain $s $((19990 + 10 * k)) ) & PIDS="$PIDS $!"; k=$((k + 1)); sleep 120; done
for p in $PIDS; do wait $p; done
( set -o noclobber; date -u +%FT%TZ > $P2/P2_TEST_DONE ) || true
log "P2_TEST_DONE"
