#!/usr/bin/env bash
# After the VE1-s2 training (reports/mist_w3po_v1/thinker_ve1_s2_confirm_prereg_v1.zh-CN.md §1-§3):
#   1. wait for CRF_RUN_EXIT; rc=0 with step_03000/DONE, else stop (slot kept for inspection); release the slot (rename)
#   2. dev decode (8 shards, 1.0 s reads, --talker none: decode + text score + text latency)   run t_eval/ve1s2
#   3. read-level merged copy (crf/merge_read_segments.py) -> <decode>_rl; dev5 cut of the full decode (t_dev5_cut.sh,
#      CPU, descriptive) in the background
#   4. PL-EM speech chain on the merged copy, 2 TTS seeds, gates vs A' (c1b1s2), A' + PL-EM and A'rl + PL-EM
#      (primary: speech vs A'rl + PL-EM; text vs A')                      run t_eval/ve1s2rlplem
#   Every stage --cap135 --exclude 4 5 6 7 (cards 0-3 only; RC5's quick check / evaluations on 4-7 keep priority).
#   RUN=<training run id> [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/ve1s2_after_train.sh
# Run detached: setsid nohup bash .../ve1s2_after_train.sh > logs/mist_w3po_v1/ve1s2/after_train.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
RUN=${RUN:?}
export PYTHONPATH=.
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
SL=logs/mist_w3po_v1/slots
SLOT=slotA
E=logs/mist_w3po_v1/t_eval
DEC=eval_outputs/mist_w3po_v1/decode
CK=eval_outputs/mist_w3po_v1/crf/train/$RUN/ckpt/step_03000
F=dev_ve1s2_s3000
A=dev_crf_v2p_s3000_finalpost_s8_c2g06
EX="--cap135 --exclude 4 5 6 7"
ORCH="$PY experiments/mist_w3po_v1/scripts/thinker_t_eval_orch.py"
REFS="--gate-ref $A ${A}_tts2 --gate-ref ${A}__plem_c1b1s2 ${A}__plem_c1b1s2_tts2 --gate-ref ${A}__rlplem ${A}__rlplem_tts2"
DCMD="$ORCH --run $E/ve1s2 --name $F --ckpt $CK --talker none --window 1 --scorer-port0 19331 $EX"
PCMD="$ORCH --run $E/ve1s2rlplem --name ${F}__rlplem --ckpt - --speech-from ${F}_rl --talker plem --window 1 $REFS --scorer-port0 19341 $EX"
MCMD="$PY -m experiments.mist_w3po_v1.crf.merge_read_segments --src $F --dst ${F}_rl"
log() { echo "$(date -u +%FT%TZ) $*"; }
wait_done() { until [ -e $1/events/DONE ]; do sleep 30; done; }
clean_done() { grep -q 'DONE; failed/skipped \[\]' $1/orch.log; }
EXIST=""
for d in $E/ve1s2 $E/ve1s2rlplem $DEC/$F $DEC/${F}_rl $DEC/${F}__rlplem $DEC/dev5cut_$F; do [ -e $d ] && EXIST="$EXIST $d"; done
REFOK=1
for r in $A ${A}_tts2 ${A}__plem_c1b1s2 ${A}__plem_c1b1s2_tts2 ${A}__rlplem ${A}__rlplem_tts2; do
  [ -s $DEC/$r/score_asr.json ] || { REFOK=0; echo "missing gate reference $DEC/$r/score_asr.json"; }
done
if [ -n "${DRY_RUN:-}" ]; then
  echo "run log: logs/mist_w3po_v1/$RUN.log ($([ -e logs/mist_w3po_v1/$RUN.log ] && echo present || echo 'not yet')); checkpoint: $CK"
  echo "existing outputs (must be none):${EXIST:- none}; gate references ok: $REFOK"
  echo "ports 19331-19344 in use: $(ss -ltn 2>/dev/null | grep -cE ':(1933[0-9]|1934[0-4]) ')"
  echo "decode: $DCMD"; echo "merge: $MCMD"; echo "dev5 cut: bash experiments/mist_w3po_v1/scripts/t_dev5_cut.sh $F"
  echo "speech PL-EM: $PCMD"
  exit 0
fi
[ -z "$EXIST" ] || { log "拒绝覆盖已有输出:$EXIST"; exit 1; }
[ $REFOK = 1 ] || { log "gate references missing: not started"; exit 1; }
L=logs/mist_w3po_v1/$RUN.log
until [ -e $L ] && grep -q CRF_RUN_EXIT $L; do sleep 30; done
log "$RUN exit: $(grep CRF_RUN_EXIT $L)"
if ! grep -q 'CRF_RUN_EXIT rc=0' $L || [ ! -e $CK/DONE ]; then log "$RUN did not finish cleanly: no evaluation (slot kept)"; exit 1; fi
[ -e $SL/$SLOT.owner ] && grep -q "$RUN" $SL/$SLOT.owner && \
  command mv -f $SL/$SLOT.owner $SL/$SLOT.released_$(date -u +%Y%m%dT%H%M%SZ) && log "$SLOT released"
mkdir -p $E
setsid nohup $DCMD < /dev/null > $E/ve1s2.launch.log 2>&1 &
log "dev decode orchestrator pid $!: $DCMD"
sleep 20
[ -e $E/ve1s2/orch.log ] || { log "decode orchestrator did not start (see $E/ve1s2.launch.log)"; exit 1; }
wait_done $E/ve1s2
clean_done $E/ve1s2 || { log "decode orchestrator DONE with failed/skipped jobs: $(grep 'DONE;' $E/ve1s2/orch.log); stop"; exit 1; }
log "decode DONE"
$MCMD > $E/ve1s2_merge.log 2>&1 || { log "merge failed (see $E/ve1s2_merge.log)"; exit 1; }
log "merged: $(tail -n 1 $E/ve1s2_merge.log | cut -c1-300)"
setsid nohup bash experiments/mist_w3po_v1/scripts/t_dev5_cut.sh $F < /dev/null > logs/mist_w3po_v1/ve1s2/dev5cut.log 2>&1 &
log "dev5 cut pid $!"
setsid nohup $PCMD < /dev/null > $E/ve1s2rlplem.launch.log 2>&1 &
log "PL-EM speech orchestrator pid $!: $PCMD"
wait_done $E/ve1s2rlplem
log "PL-EM speech DONE: $(grep 'DONE;' $E/ve1s2rlplem/orch.log)"
log "VE1S2_ALL_DONE"
