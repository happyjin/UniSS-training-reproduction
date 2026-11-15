#!/usr/bin/env bash
# v11 P2 test (prereg reports/mist_w3po_v1/v11_p2_fluency_pa_prereg.zh-CN.md §5): only directions that passed the dev
# gate are tested, once each. Zh->En passed (dev speech +0.05, P(diff<=0) 0.33); En->Zh failed (-0.09, CI [-0.12,-0.04])
# and is not tested. Exactly v11_p2_plemf_v2.sh's dev chain with --set test|ali: B's read-merged test text
# (jtest_b{ami,ali}_<RT1>_rl), joint_orch_plemx.py --plem-x <PA-F export> --talker plem, both TTS seeds, gate_2seed vs
# E2's B+PA (<N>__rlplem, _tts2); then the chunk-protocol perword latency as v11_p01_factorial.sh lat().
# Downstream: v11_p01_postproc.sh with the row R6:__rlplemf (lat_units / pauses / quality) and v11_factorial_analysis.
# Marker logs/mist_v11/p2/P2_TEST_DONE. Refuses to overwrite run dirs. Checked with bash -n and DRY_RUN=1 only.
#   [SETS="ami ali"] [PORT0=21400] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/v11_p2_plemf_test.sh
# Run detached: setsid nohup bash ... > logs/mist_v11/p2/test.log 2>&1 < /dev/null 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
SETS=${SETS:-"ami ali"}
PORT0=${PORT0:-21400}
P2=logs/mist_v11/p2
B2=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/thinker_merged
RT1=joint_v8h6k_20261113T073658Z_s06000
X=$(cat $P2/EXPORT_zh2en)
setvars() {
  case $1 in
    ami) N=jtest_bami_$RT1; K=test; CW=$WT_AMIC; REF=$RT_AMI; DR=zh2en;;
    ali) N=jtest_bali_$RT1; K=ali;  CW=$WT_ALIC; REF=$RT_ALI; DR=ali;;
    *) echo "bad set $1"; exit 1;;
  esac
}
lat() {   # <set> <run name>: v11_p01_factorial.sh lat() for zh2en test sets
  local s=$1 n=$2 O f
  setvars $s
  O=$TE/${n%_tts2}; mkdir -p $O; f=$O/lat_chunk_$n.json
  [ -s $f ] && return 0
  CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.on_refset --ref-win $REF --dr $DR \
    experiments.mist_w3po_v1.evaluation.speech_latency_lit --run $DEC/$n --talker $SPD/$n/talker --win $CW --out $f > $f.log 2>&1 || log "chunk LAAL failed: $n"
}
chain() {   # <set> <port0>
  local s=$1 p=$2 R C
  setvars $s
  R=$P2/test_$s
  C="$PY $SC/joint_orch_plemx.py --plem-x $X --dir zh2en --set $K --base $B2 --run $R --name ${N}__rlplemf --ckpt - --speech-from ${N}_rl --talker plem --window 1 --cap135 --scorer-port0 $p --gate-ref ${N}__rlplem ${N}__rlplem_tts2"
  if [ -n "${DRY_RUN:-}" ]; then
    echo "$s: $C"; [ -e $DEC/${N}_rl/score_text.json ] && echo "  rl text ok" || echo "  RL TEXT MISSING"
    $C --dry 2>&1 | tail -n 1; return 0
  fi
  [ -e $R ] && { log "P2T_STOP $s: $R exists (not overwritten)"; return 1; }
  setsid nohup $C < /dev/null > $R.launch.log 2>&1 9>&- &
  log "$s test orchestrator pid $!"
  wait_done $R
  [ -z "$(orch_failed $R)" ] || { log "P2T_STOP $s: orchestrator failed ($(tail -n 1 $R/orch.log | cut -c1-200))"; return 1; }
  for n in ${N}__rlplemf ${N}__rlplemf_tts2; do lat $s $n; done
  log "$s done (gate: $(ls $TE/${N}__rlplemf/gate_*.json 2>/dev/null))"
}
[ -e $X/uniss/model.safetensors ] || { log "no PA-F export $X"; exit 1; }
if [ -z "${DRY_RUN:-}" ] && [ -e $P2/P2_TEST_DONE ]; then log "拒绝覆盖已有输出: $P2/P2_TEST_DONE"; exit 1; fi
PIDS=""; k=0
for s in $SETS; do
  if [ -n "${DRY_RUN:-}" ]; then chain $s $((PORT0 + 10 * k)); k=$((k + 1)); continue; fi
  ( chain $s $((PORT0 + 10 * k)) ) &
  PIDS="$PIDS $!"; k=$((k + 1)); sleep 60
done
[ -n "${DRY_RUN:-}" ] && exit 0
rc=0; for p in $PIDS; do wait $p || rc=1; done
[ $rc = 0 ] || { log "P2T_STOP: a chain failed"; exit 1; }
( set -o noclobber; date -u +%FT%TZ > $P2/P2_TEST_DONE ) || true
log "P2_TEST_DONE"
