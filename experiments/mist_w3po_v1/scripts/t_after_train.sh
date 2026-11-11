#!/usr/bin/env bash
# After a T4 / T5 training (prereg §T4 / §T5; coordinator 2026-11-11 ~17:40 UTC: the dev5 quick check runs on the
# experiment's own cards, no eval on training cards): wait for CRF_RUN_EXIT; if rc=0 with step_03000/DONE, release the
# slot (rename), run the dev5 text quick check on CARDS (thinker_t_eval_orch --talker none --win dev5, 4 window shards,
# --window 1.0 unless READ_MS=500, --cap135; the other 4 cards excluded), stall diag vs A' dev5, GO lines
# (t_quick_go.py). GO -> the dev full eval (8 shards, c1+b1+s2, 2 TTS seeds, gate vs A'), excluding the cards of any slot
# still owned (training) at that moment. When the orchestrators are over, the guard's standby supervisors take the cards.
#   RUN=<training run id> TAG=<t4|t5> CARDS=4,5,6,7 SLOT=slotB PORT0=<scorer port> [DRY_RUN=1] bash t_after_train.sh
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
RUN=${RUN:?}; TAG=${TAG:?}; CARDS=${CARDS:?}; SLOT=${SLOT:?}; PORT0=${PORT0:?}
READ_MS=${READ_MS:-1000}
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
TR=eval_outputs/mist_w3po_v1/crf/train
SL=logs/mist_w3po_v1/slots
E=logs/mist_w3po_v1/t_eval
W5=eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1_dev5/windows_dev_online.jsonl
A=dev_crf_v2p_s3000_finalpost_s8_c2g06
WX="--window $(awk -v m=$READ_MS 'BEGIN{printf "%g", m/1000}')"
Q=dev5_${TAG}_s3000
F=dev_${TAG}_s3000
QO=eval_outputs/mist_w3po_v1/crf/t_eval/$Q
case $CARDS in 4,5,6,7) EXQ="0 1 2 3";; 0,1,2,3) EXQ="4 5 6 7";; *) echo "CARDS 0,1,2,3 or 4,5,6,7"; exit 1;; esac
log() { echo "$(date -u +%FT%TZ) $*"; }
QCMD="$PY experiments/mist_w3po_v1/scripts/thinker_t_eval_orch.py --run $E/${TAG}quick --name $Q --ckpt $TR/$RUN/ckpt/step_03000 \
--talker none --win $W5 --shards 4 --shard-by window $WX --scorer-port0 $PORT0 --exclude $EXQ --cap135"
if [ -n "${DRY_RUN:-}" ]; then echo "$QCMD"; exit 0; fi
L=logs/mist_w3po_v1/$RUN.log
until [ -e $L ] && grep -q CRF_RUN_EXIT $L; do sleep 30; done
log "$RUN exit: $(grep CRF_RUN_EXIT $L)"
if ! grep -q 'CRF_RUN_EXIT rc=0' $L || [ ! -e $TR/$RUN/ckpt/step_03000/DONE ]; then
  log "$RUN did not finish cleanly: no quick check (slot kept for inspection)"; exit 1
fi
[ -e $SL/$SLOT.owner ] && grep -q "$RUN" $SL/$SLOT.owner && \
  command mv -f $SL/$SLOT.owner $SL/$SLOT.released_$(date -u +%Y%m%dT%H%M%SZ) && log "$SLOT released"
mkdir -p $E
setsid nohup $QCMD < /dev/null > $E/${TAG}quick.launch.log 2>&1 &
log "quick check orchestrator pid $!"
until [ -e $E/${TAG}quick/events/DONE ]; do sleep 30; done
log "quick check DONE"
python3 eval_outputs/mist_w3po_v1/speech_laal_v1/analysis_20261111/t1_stall_diag.py eval_outputs/mist_w3po_v1/decode/$Q/hyp.jsonl \
  eval_outputs/mist_w3po_v1/decode/plcrf_dev5_aprule/hyp.jsonl eval_outputs/mist_w3po_v1/crf/t0_agree_dev5/analysis/pieces.jsonl \
  $W5 > $QO/stall_diag.txt 2>&1
$PY experiments/mist_w3po_v1/scripts/t_quick_go.py $Q $QO $W5 > $QO/go.log 2>&1
log "GO: $(cat $QO/go.log | head -c 400)"
if grep -q '"GO": true' $QO/go.json; then
  EXF=""
  [ -e $SL/slotA.owner ] && EXF="$EXF 0 1 2 3"
  [ -e $SL/slotB.owner ] && EXF="$EXF 4 5 6 7"
  FCMD="$PY experiments/mist_w3po_v1/scripts/thinker_t_eval_orch.py --run $E/$TAG --name $F --ckpt $TR/$RUN/ckpt/step_03000 \
--talker c1b1s2 --gate-ref $A ${A}_tts2 $WX --scorer-port0 $((PORT0 + 10)) --cap135 --exclude $EXF"
  setsid nohup $FCMD < /dev/null > $E/$TAG.launch.log 2>&1 &
  log "GO -> full dev eval orchestrator pid $! (excluded:$EXF)"
else
  log "NOT GO: no full eval"
fi
