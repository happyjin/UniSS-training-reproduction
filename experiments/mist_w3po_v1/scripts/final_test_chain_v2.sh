#!/usr/bin/env bash
# v2 (coordinator 2026-11-12 ~20:18 Prague: start the VE1 / D1v test decodes NOW on cards 4-7, in parallel with the A'
# decodes; group 1 speech first once A' is decoded). Changes from final_test_chain.sh, nothing else:
#   - ADOPT: an A' decode orchestrator already running for this set (run dir present, started by final_test_chain.sh,
#     whose driver was stopped by PID) is adopted -- not relaunched; its outputs are allowed to exist
#   - the VE1 and D1v decodes start right away with --exclude 0 1 2 3 (cards 4-7; the coordinator moves the line by
#     rewriting <run>/exclude_cards, new file + mv -f); the ami set's decodes start after every shard of the ali set's
#     decode of the same Thinker has started (--dec-after: the longer AliMeeting decodes first)
#   - steps 3-6 unchanged (group 1 chains once A' is decoded and merged; (d) / (e) after their decode)
# FINAL TEST evaluation of one test set (reports/mist_w3po_v1/final_test_prereg_v1.zh-CN.md §4). Each step once, fixed
# configurations, no selection on test. Every orchestrator is final_test_orch.py (thinker_t_eval_orch.py with the set's
# C2 gap-0.6 test windows, reference windows, talker data), --cap135, no --exclude (all 8 cards; the run's exclude_cards
# file is empty), 8 decode shards, 8 talker shards per TTS seed (as dev):
#   1. A' decode (v2' step_03000 finalpost, 1.0 s reads, --talker none)                    run final_test/<set>_aprime
#   2. read-level merge A' -> <A'>_rl (merge_read_segments.py; registered rule: continue when the merge itself loses
#      <= 2 windows -- pair_lost <= 2 -- with a complete copy, the windows a known deviation; else FT_MERGE_STOP: that
#      chain is not started, ask the coordinator)
#   3. group 1, 60 s apart: (c) A'rl + PL-EM (main system), (b) A' + PL-EM, (a) A' c1+b1+s2; then the VE1 and D1v
#      decodes (60 s apart)
#   4. per Thinker (background): decode DONE -> merge -> (d) VE1rl + PL-EM / (e) D1vrl + PL-EM
#   5. after each speech chain (background): post-hoc measures (final_test_post.sh: chunk-protocol latency, talker WER,
#      block / pause stats, UTMOS / SIM)
#   6. FT_GROUP1_DONE <set> once (a)-(c) and their post-hoc measures are done; then the gates / paired UTMOS of group 1;
#      FT_ALL_DONE <set> at the end (gates / paired UTMOS of (d), (e) vs (c))
#   SET=ami|ali [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/final_test_chain_v2.sh
# Run detached: setsid nohup bash .../final_test_chain_v2.sh > logs/mist_w3po_v1/final_test/chain_v2_<set>.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
SET=${SET:?ami or ali}
case $SET in ami) NW=3973; P0=20100; WS=zh2en;; ali) NW=4065; P0=20200; WS=ali;; *) echo "SET ami|ali"; exit 1;; esac
export PYTHONPATH=. TMPDIR=/opt/dlami/nvme/jasonleeeli/work/mist_w3po_v1/tmp
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
E=logs/mist_w3po_v1/final_test
DEC=eval_outputs/mist_w3po_v1/decode
SPD=eval_outputs/mist_w3po_v1/speech
Q=eval_outputs/mist_w3po_v1/talker_crf/quality
O=eval_outputs/mist_w3po_v1/crf/t_eval/final_test_v1/$SET
WIN=eval_outputs/mist_w3po_v1/data/windows_test_c2g06_v1/windows_test_online_$WS.jsonl
T=eval_outputs/mist_w3po_v1/crf/train
CKA=$T/crf_v2p_20261110T075131Z/ckpt/step_03000
CKV=$T/ve1_20261111T214454Z/ckpt/step_03000
CKD=$T/d1v_20261112T120413Z/ckpt/step_03000
FA=ft_${SET}_aprime_s3000; FV=ft_${SET}_ve1_s3000; FD=ft_${SET}_d1v_s3000
NC=${FA}__rlplem; NB=${FA}__plem; NA=${FA}__c1b1s2; ND=${FV}__rlplem; NE=${FD}__rlplem
ORCH="$PY experiments/mist_w3po_v1/scripts/final_test_orch.py --set $SET --win $WIN --window 1 --cap135"
dcmd() { echo "$ORCH --run $E/${SET}_$1 --name $2 --ckpt $3 --talker none --scorer-port0 $4"; }
G2X="--exclude 0 1 2 3"
afterx() { [ $SET = ami ] && echo "--dec-after $E/ali_$1 ft_ali_$1_s3000"; }
scmd() { echo "$ORCH --run $E/${SET}_$1 --name $2 --ckpt - --speech-from $3 --talker $4 --scorer-port0 $5"; }
POST="bash experiments/mist_w3po_v1/scripts/final_test_post.sh"
log() { echo "$(date -u +%FT%TZ) $*"; }
wait_done() { until [ -e $1/events/DONE ]; do sleep 30; done; }
clean_done() { grep -q 'DONE; failed/skipped \[\]' $1/orch.log; }
jget() { $PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get(sys.argv[2], 'missing'))" "$1" "$2" 2>/dev/null || echo missing; }
EXIST=""
ADOPT=0
[ -e $E/${SET}_aprime/orch.log ] && ADOPT=1
for r in ve1 d1v aprlplem applem apc1b1s2 ve1rlplem d1vrlplem; do [ -e $E/${SET}_$r ] && EXIST="$EXIST $E/${SET}_$r"; done
for n in $FV $FD ${FA}_rl ${FV}_rl ${FD}_rl; do [ -e $DEC/$n ] && EXIST="$EXIST $DEC/$n"; done
[ $ADOPT = 0 ] && [ -e $DEC/$FA ] && EXIST="$EXIST $DEC/$FA"
for n in $NC $NB $NA $ND $NE; do for d in $DEC/$n $DEC/${n}_tts2 $SPD/$n $SPD/${n}_tts2 $Q/$n $Q/${n}_tts2; do
  [ -e $d ] && EXIST="$EXIST $d"; done; done
[ -e $O ] && [ -n "$(ls -A $O)" ] && EXIST="$EXIST $O"   # v2: the empty dir the stopped driver made is fine
CKOK=1
for c in $CKA $CKV $CKD; do for f in DONE lora.pt extra.pt; do [ -e $c/$f ] || { CKOK=0; echo "missing $c/$f"; }; done; done
[ "$(wc -l < $WIN)" = $NW ] || { CKOK=0; echo "$WIN: not $NW windows"; }
if [ -n "${DRY_RUN:-}" ]; then
  echo "set $SET ($NW windows, $WIN); ports $P0-$((P0 + 73)) in use: $(ss -ltn 2>/dev/null | awk '{print $4}' | grep -cE ":(2[01][0-9][0-9][0-9])$")"
  echo "existing outputs (must be none):${EXIST:- none}; checkpoints / windows ok: $CKOK; adopt A' decode: $ADOPT"
  echo "1  $(dcmd aprime $FA $CKA $P0)"
  echo "2  merge $FA -> ${FA}_rl"
  echo "3c $(scmd aprlplem $NC ${FA}_rl plem $((P0 + 10)))"
  echo "3b $(scmd applem $NB $FA plem $((P0 + 20)))"
  echo "3a $(scmd apc1b1s2 $NA $FA c1b1s2 $((P0 + 30)))"
  echo "0v $(dcmd ve1 $FV $CKV $((P0 + 40))) $G2X $(afterx ve1)"
  echo "0e $(dcmd d1v $FD $CKD $((P0 + 50))) $G2X $(afterx d1v)"
  echo "4d $(scmd ve1rlplem $ND ${FV}_rl plem $((P0 + 60)))"
  echo "4e $(scmd d1vrlplem $NE ${FD}_rl plem $((P0 + 70)))"
  echo "5  $POST (SET=$SET NAME=<speech name>); 6 gates: $POST gates"
  $(dcmd aprime $FA $CKA $P0) --dry | tail -n 1
  $(scmd aprlplem $NC ${FA}_rl plem $((P0 + 10))) --dry | tail -n 1
  $(scmd apc1b1s2 $NA $FA c1b1s2 $((P0 + 30))) --dry | tail -n 1
  exit 0
fi
[ -z "$EXIST" ] || { log "拒绝覆盖已有输出:$EXIST"; exit 1; }
[ $CKOK = 1 ] || { log "checkpoints / windows missing: not started"; exit 1; }
mkdir -p $E $O
launch() {  # <run tag> <command words...>
  local r=$1; shift
  setsid nohup "$@" < /dev/null > $E/${SET}_$r.launch.log 2>&1 &
  log "launched ${SET}_$r pid $!: $*"
  sleep 20
  [ -e $E/${SET}_$r/orch.log ] || { log "FT_LAUNCH_FAILED ${SET}_$r (see $E/${SET}_$r.launch.log)"; return 1; }
}
merge() {   # <decode name> -> <decode name>_rl, the registered pair_lost rule
  local S=$1 Dd=$DEC/$1_rl ML=$E/${SET}_merge_$1.log
  if $PY -m experiments.mist_w3po_v1.crf.merge_read_segments --src $S --dst ${S}_rl > $ML 2>&1; then
    log "merged $S: $(tail -n 1 $ML | cut -c1-300)"; return 0
  fi
  local PL PO WN HL
  PL=$(jget $Dd/merge_read_segments.json pair_lost); PO=$(jget $Dd/merge_read_segments.json pair_ok)
  WN=$(jget $Dd/merge_read_segments.json windows); HL=$(wc -l < $Dd/hyp.jsonl 2>/dev/null || echo 0)
  if grep -q 'pairing lost' $ML && [ "$WN" = $NW ] && [ "$HL" = $NW ] && [ -s $Dd/score_text.json ] \
     && [ "$PL" != missing ] && [ "$PL" -le 2 ]; then
    log "FT_MERGE_DEVIATION $S: pair_lost $PL (<= 2, registered rule), pair_ok $PO / $WN, copy complete: continuing (known deviation)"
    return 0
  fi
  log "FT_MERGE_STOP $S: pair_lost $PL, pair_ok $PO / $WN, hyp lines $HL (see $ML): chain not started, ask the coordinator"
  return 1
}
chain_end() {   # <run tag> <speech name>: wait, report, post-hoc measures
  wait_done $E/${SET}_$1
  if clean_done $E/${SET}_$1; then log "FT_CHAIN_DONE ${SET}_$1"; else log "FT_CHAIN_FAILED ${SET}_$1: $(grep 'DONE;' $E/${SET}_$1/orch.log)"; fi
  SET=$SET NAME=$2 $POST > $O/post_$2.log 2>&1
  log "post-hoc $2 rc=$? ($(tail -n 1 $O/post_$2.log | cut -c1-200))"
}
thinker2() {    # <decode tag> <decode name> <speech tag> <speech name> <speech port>
  wait_done $E/${SET}_$1
  clean_done $E/${SET}_$1 || { log "FT_DECODE_FAILED ${SET}_$1: $(grep 'DONE;' $E/${SET}_$1/orch.log)"; return 1; }
  log "FT_DECODE_DONE ${SET}_$1: $(cut -c1-200 $DEC/$2/score.log 2>/dev/null | tail -n 1)"
  SET=$SET NAME=$2 $POST textlat > $O/post_text_$2.log 2>&1 &
  merge $2 || return 1
  launch $3 $(scmd $3 $4 ${2}_rl plem $5) || return 1
  chain_end $3 $4
}
if [ $ADOPT = 1 ]; then log "adopted the running A' decode orchestrator $E/${SET}_aprime"
else launch aprime $(dcmd aprime $FA $CKA $P0) || exit 1; fi
launch ve1 $(dcmd ve1 $FV $CKV $((P0 + 40))) $G2X $(afterx ve1) && { thinker2 ve1 $FV ve1rlplem $ND $((P0 + 60)) & }
sleep 60
launch d1v $(dcmd d1v $FD $CKD $((P0 + 50))) $G2X $(afterx d1v) && { thinker2 d1v $FD d1vrlplem $NE $((P0 + 70)) & }
wait_done $E/${SET}_aprime
clean_done $E/${SET}_aprime || { log "FT_DECODE_FAILED ${SET}_aprime: $(grep 'DONE;' $E/${SET}_aprime/orch.log); stop"; exit 1; }
log "FT_DECODE_DONE ${SET}_aprime: $(cut -c1-200 $DEC/$FA/score.log 2>/dev/null | tail -n 1)"
SET=$SET NAME=$FA $POST textlat > $O/post_text_$FA.log 2>&1 &
G1=""
if merge $FA; then
  launch aprlplem $(scmd aprlplem $NC ${FA}_rl plem $((P0 + 10))) && { chain_end aprlplem $NC & G1="$G1 $!"; }
  sleep 60
fi
launch applem $(scmd applem $NB $FA plem $((P0 + 20))) && { chain_end applem $NB & G1="$G1 $!"; }
sleep 60
launch apc1b1s2 $(scmd apc1b1s2 $NA $FA c1b1s2 $((P0 + 30))) && { chain_end apc1b1s2 $NA & G1="$G1 $!"; }
for p in $G1; do wait $p; done
SET=$SET $POST gates1 > $O/post_gates1.log 2>&1
log "FT_GROUP1_DONE $SET (gates / paired UTMOS rc=$?)"
wait
SET=$SET $POST gates2 > $O/post_gates2.log 2>&1
log "FT_ALL_DONE $SET (gates / paired UTMOS rc=$?)"
