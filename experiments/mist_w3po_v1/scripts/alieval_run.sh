#!/usr/bin/env bash
# v11 P0-2, run once (reports/mist_w3po_v1/v11_p02_alieval_prereg.zh-CN.md; user 2026-11-15: plan A, best-option
# authority): B, B+PA, M (s3500, wait bias -1) and M4000 (s4000, wait bias -1) on the AliMeeting Eval clean set, with
# exactly the v10 test recipes (joint_test.sh BASE_TALKER=t1c baseline row, e2_baseline_plem.sh B+PA, wb_test.sh /
# wb_test_s4000.sh method rows) through alieval_orch.py (only the AliMeeting set definition swapped). Per row: text
# decode (8 shards by meeting, window 1.0, --cap135) -> chunk text LAAL -> speech: B: T1c (no merge) and rl + PL-EM;
# M / M4000: rl + PL-EM; two TTS seeds -> chunk speech LAAL. Nothing is chosen. Names jaev_{b,m,m4}_<run tag>; run
# dirs logs/mist_w3po_v1/joint/alieval/<row>; marker logs/mist_w3po_v1/joint/alieval/ALIEVAL_DONE. Refuses to overwrite.
#   [PORT0=20600] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/alieval_run.sh
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/joint/alieval.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
PORT0=${PORT0:-20600}
TV=$JL/alieval
DONE_M=$TV/ALIEVAL_DONE
AORCH="$PY $SC/alieval_orch.py"
WIN=$DD/windows_alieval_c2g06_v1/windows_test_online_ali.jsonl
CW=$DD/windows_alieval_c2g06_chunk_v1/windows_dev_online.jsonl
REF=data/diarist_alimeeting_refs_v1/windows_eval_ref_human.jsonl
B2=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/thinker_merged
IDN=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/identity
CKD=$CRF/train/joint_v2p_20261113T120415Z/ckpt
RT1=joint_v8h6k_20261113T073658Z_s06000
NB=jaev_b_$RT1
NM=jaev_m_joint_v2p_20261113T120415Z_s03500_wbzm1
NM4=jaev_m4_joint_v2p_20261113T120415Z_s04000_wbzm1
flag() { ( set -o noclobber; echo "$(date -u +%FT%TZ) $*" > $TV/FAILED_$(date -u +%s%N) ); log "ALIEVAL_STOP: $*"; }
lat() {   # <run name> [<talker dir>] -> chunk-protocol LAAL json (text-only without a talker), as wb_test.sh zh2en
  local n=$1 tk=${2:-} O f A
  O=$TE/${n%_tts2}; mkdir -p $O
  if [ -z "$tk" ]; then f=$O/textlat_chunk_$n.json; A="--text-only"; else f=$O/lat_chunk_$n.json; A="--talker $tk"; fi
  CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.on_refset --ref-win $REF --dr ali \
    experiments.mist_w3po_v1.evaluation.speech_latency_lit --run $DEC/$n $A --win $CW --out $f > $f.log 2>&1 || log "chunk LAAL failed: $n"
}
run_orch() {   # <run dir> <command...>: start detached, wait, check
  local R=$1; shift
  if [ -n "${DRY_RUN:-}" ]; then echo "$(basename $R): $*"; "$@" --dry | tail -1; return 0; fi
  [ -e $R ] && { flag "$R exists (not overwritten)"; return 1; }
  setsid nohup "$@" < /dev/null > $R.launch.log 2>&1 &
  log "$(basename $R) orchestrator pid $!"
  wait_done $R; [ -z "$(orch_failed $R)" ] || { flag "$(basename $R) failed ($(tail -1 $R/orch.log | cut -c1-200))"; return 1; }
}
plem() {   # <text name> <row> <port>: read-level merge -> PL-EM speech (two seeds) -> chunk speech LAAL
  local n=$1 r=$2 p=$3 x
  if [ -z "${DRY_RUN:-}" ]; then
    read_merge $n ' ' $WIN $TV/merge_$n.log || { flag "merge $n ($(tail -1 $TV/merge_$n.log | cut -c1-200))"; return 1; }
  fi
  run_orch $TV/${r}_pa $AORCH --dir zh2en --set ali --base $B2 --run $TV/${r}_pa --name ${n}__rlplem --ckpt - \
    --speech-from ${n}_rl --talker plem --window 1 --cap135 --scorer-port0 $p || return 1
  [ -n "${DRY_RUN:-}" ] && return 0
  for x in ${n}__rlplem ${n}__rlplem_tts2; do lat $x $SPD/$x/talker; done
  log "${n}__rlplem done"
}
row_b() {
  run_orch $TV/b $AORCH --dir zh2en --set ali --base $B2 --run $TV/b --name $NB --ckpt $IDN --talker none --window 1 --cap135 || return 1
  [ -z "${DRY_RUN:-}" ] && lat $NB
  ( run_orch $TV/b_t1c $AORCH --dir zh2en --set ali --base $B2 --run $TV/b_t1c --name ${NB}__t1c --ckpt - --speech-from $NB \
      --talker t1c --window 1 --cap135 || exit 1
    [ -n "${DRY_RUN:-}" ] && exit 0
    for x in ${NB}__t1c ${NB}__t1c_tts2; do lat $x $SPD/$x/talker; done; log "${NB}__t1c done" ) &
  local pt=$!
  plem $NB b $PORT0; local rc=$?
  wait $pt || rc=1
  return $rc
}
row_m() {   # <row m|m4> <name> <step dir> <port>
  local r=$1 n=$2 ck=$3 p=$4
  run_orch $TV/$r $AORCH --wait-bias -1 --dir zh2en --set ali --base $B2 --run $TV/$r --name $n --ckpt $ck --talker none --window 1 --cap135 || return 1
  [ -z "${DRY_RUN:-}" ] && lat $n
  plem $n $r $p
}
for f in $WIN $CW $REF $B2/sd_proj.pt $IDN $CKD/step_03500/DONE $CKD/step_04000/DONE; do [ -e $f ] || { log "missing $f"; exit 1; }; done
if [ -n "${DRY_RUN:-}" ]; then
  row_b; row_m m $NM $CKD/step_03500 $((PORT0 + 10)); row_m m4 $NM4 $CKD/step_04000 $((PORT0 + 20)); exit 0
fi
[ -e $DONE_M ] && { log "拒绝覆盖已有输出: $DONE_M (runs once)"; exit 1; }
for n in $NB $NM $NM4; do [ -e $DEC/$n ] && { log "拒绝覆盖已有输出: $DEC/$n"; exit 1; }; done
mkdir -p $TV
log "ALIEVAL once: B $NB, M $NM, M4000 $NM4"
( row_b ) & P1=$!
sleep 120
( row_m m $NM $CKD/step_03500 $((PORT0 + 10)) ) & P2=$!
sleep 120
( row_m m4 $NM4 $CKD/step_04000 $((PORT0 + 20)) ) & P3=$!
rc=0; for p in $P1 $P2 $P3; do wait $p || rc=1; done
[ $rc = 0 ] && ! ls $TV/FAILED_* > /dev/null 2>&1 || { log "ALIEVAL_STOP: a row failed ($(cat $TV/FAILED_* 2>/dev/null | tr '\n' ' ' | cut -c1-300))"; exit 1; }
( set -o noclobber; echo "utc $(date -u +%FT%TZ)" > $DONE_M ) || log "marker exists"
log "ALIEVAL_DONE"
