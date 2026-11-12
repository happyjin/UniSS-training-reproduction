#!/usr/bin/env bash
# After the D1v or J1-v2 training (thinker_dagger_d1v_prereg_v1 §3.1 / thinker_joint_crf_v2_prereg_v1 §4.1); the shape
# of pmc_after_train.sh / pmcnoc_after_train.sh, one script for both tags:
#   1. wait for CRF_RUN_EXIT; rc=0 with step_03000/DONE, else stop (slot kept for inspection); release the slot (rename)
#   2. dev decode (8 shards, 1.0 s reads, --talker none) on the training's 4 cards (--exclude the other 4)
#   3. read-level merged copy (crf/merge_read_segments.py) -> dev_<tag>_s3000_rl. Pre-authorised continuation (the
#      coordinator's 2026-11-12 11:00 rule, registered in both prereg): the tool stopped ONLY on "pairing lost", the
#      merged copy is complete (hyp lines = windows = 3618, score_text.json, merge_read_segments.json) and pair_ok >=
#      3589 -> continue, the windows listed as a known deviation; any other stop -> stop and report
#   4. dev5 cut + the registered GO lines (v4a_dev5_go.py) in the background
#   5. PL-EM speech chain on the merged copy, 2 TTS seeds; gates vs A', A'rl + PL-EM (registered), VE1rl + PL-EM
#      (descriptive)
#   6. listening -> crf/t_eval/listen_<tag>_v1/ (talker WER, block / pause stats on the CPU; UTMOS / SIM on ONE
#      non-training card; UTMOS paired vs VE1rl + PL-EM, A'rl + PL-EM, A' + PL-EM)
# The whole chain runs whether or not the dev5 quick check is GO (registered); the verdict needs GO.
#   TAG=d1v|j1v2 RUN=<training run id> [CARDS=0,1,2,3] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/v4a_after_train.sh
# Run detached: setsid nohup bash .../v4a_after_train.sh > logs/mist_w3po_v1/<tag>/after_train.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
TAG=${TAG:?d1v or j1v2}
RUN=${RUN:?}
CARDS=${CARDS:-0,1,2,3}
case $TAG in d1v) P1=19601; P2=19611;; j1v2) P1=19701; P2=19711;; *) echo "TAG d1v or j1v2"; exit 1;; esac
case $CARDS in 0,1,2,3) SLOT=slotA; EX="--cap135 --exclude 4 5 6 7"; ORDER="0 1 2 3 4 5 6 7";;
               4,5,6,7) SLOT=slotB; EX="--cap135 --exclude 0 1 2 3"; ORDER="4 5 6 7 0 1 2 3";;
               *) echo "CARDS 0,1,2,3 or 4,5,6,7"; exit 1;; esac
export TMPDIR=/opt/dlami/nvme/jasonleeeli/work/mist_w3po_v1/tmp HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface \
  TORCH_HOME=/opt/dlami/nvme/jasonleeeli/cache/torch XDG_CACHE_HOME=/opt/dlami/nvme/jasonleeeli/cache \
  PYTHONPATH=.:third_party/Megatron-LM OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
SL=logs/mist_w3po_v1/slots
E=logs/mist_w3po_v1/t_eval
LP=logs/mist_w3po_v1/$TAG
DEC=eval_outputs/mist_w3po_v1/decode
SP=eval_outputs/mist_w3po_v1/speech
Q=eval_outputs/mist_w3po_v1/talker_crf/quality
O=eval_outputs/mist_w3po_v1/crf/t_eval/listen_${TAG}_v1
W=eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1/windows_dev_online.jsonl
G=logs/mist_w3po_v1/talker_crf_v1/exclude_cards
CK=eval_outputs/mist_w3po_v1/crf/train/$RUN/ckpt/step_03000
F=dev_${TAG}_s3000
N=${F}__rlplem
A=dev_crf_v2p_s3000_finalpost_s8_c2g06
V1=dev_ve1_s3000__rlplem
ORCH="$PY experiments/mist_w3po_v1/scripts/thinker_t_eval_orch.py"
REFS="--gate-ref $A ${A}_tts2 --gate-ref ${A}__rlplem ${A}__rlplem_tts2 --gate-ref $V1 ${V1}_tts2"
DCMD="$ORCH --run $E/$TAG --name $F --ckpt $CK --talker none --window 1 --scorer-port0 $P1 $EX"
PCMD="$ORCH --run $E/${TAG}rlplem --name $N --ckpt - --speech-from ${F}_rl --talker plem --window 1 $REFS --scorer-port0 $P2 $EX"
MCMD="$PY -m experiments.mist_w3po_v1.crf.merge_read_segments --src $F --dst ${F}_rl"
MLOG=$E/${TAG}_merge.log
log() { echo "$(date -u +%FT%TZ) $*"; }
wait_done() { until [ -e $1/events/DONE ]; do sleep 30; done; }
clean_done() { grep -q 'DONE; failed/skipped \[\]' $1/orch.log; }
card() { local c; for c in $ORDER; do grep -qw $c $G 2>/dev/null || { echo $c; return; }; done; }
jget() { $PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get(sys.argv[2], 'missing'))" "$1" "$2" 2>/dev/null || echo missing; }
EXIST=""
for d in $E/$TAG $E/${TAG}rlplem $DEC/$F $DEC/${F}_rl $DEC/$N $DEC/${N}_tts2 $DEC/dev5cut_$F \
         eval_outputs/mist_w3po_v1/crf/t_eval/dev5cut_$F $Q/$N $Q/${N}_tts2 $O $MLOG; do [ -e $d ] && EXIST="$EXIST $d"; done
REFOK=1
for r in $A ${A}_tts2 ${A}__rlplem ${A}__rlplem_tts2 $V1 ${V1}_tts2; do
  [ -s $DEC/$r/score_asr.json ] || { REFOK=0; echo "missing gate reference $DEC/$r/score_asr.json"; }
done
for r in ${A}__plem_c1b1s2 ${A}__plem_c1b1s2_tts2 ${A}__rlplem ${A}__rlplem_tts2 $V1 ${V1}_tts2; do
  [ -s $Q/$r/quality.json.items.jsonl ] || { REFOK=0; echo "missing UTMOS reference $Q/$r"; }
done
if [ -n "${DRY_RUN:-}" ]; then
  echo "tag $TAG; cards $CARDS ($SLOT, $EX); run log: logs/mist_w3po_v1/$RUN.log ($([ -e logs/mist_w3po_v1/$RUN.log ] && echo present || echo 'not yet')); checkpoint: $CK"
  echo "existing outputs (must be none):${EXIST:- none}; references ok: $REFOK"
  echo "ports $P1-$((P1 + 23)) and $P2-$((P2 + 23)) in use: $(ss -ltn 2>/dev/null | awk '{print $4}' | grep -cE ":($P1|$P2|$((P1+1))|$((P2+1)))$")"
  echo "decode: $DCMD"; echo "merge: $MCMD  (pre-authorised continuation only on 'pairing lost' + complete copy + pair_ok >= 3589)"
  echo "dev5 cut + GO: bash experiments/mist_w3po_v1/scripts/t_dev5_cut.sh $F && $PY experiments/mist_w3po_v1/scripts/v4a_dev5_go.py $F $TAG"
  echo "speech PL-EM: $PCMD"
  echo "listening card now: $(card)"
  for n in $N ${N}_tts2; do
    echo "quality: CUDA_VISIBLE_DEVICES=<card> $PY -m experiments.mist_w3po_v1.talker.talker_quality --talker $SP/$n/talker --items $SP/$n/talker_in/items.jsonl --out $Q/$n/quality.json"
    echo "gap: $PY -m experiments.mist_w3po_v1.evaluation.speech_gap_decomp --run $DEC/$n --speech $SP/$n --win $W --out $O/gap_$n"
  done
  echo "pauses: $PY -m experiments.mist_w3po_v1.evaluation.talker_block_pauses --talker $SP/$N/talker --talker $SP/${N}_tts2/talker > $O/block_pauses_$TAG.jsonl"
  echo "utmos paired: $PY experiments/mist_w3po_v1/scripts/pmc_listen_compare.py --cand $Q/$N $Q/${N}_tts2 --ref ve1rl_plem $Q/$V1 $Q/${V1}_tts2 --ref aprime_rl_plem $Q/${A}__rlplem $Q/${A}__rlplem_tts2 --ref aprime_plem $Q/${A}__plem_c1b1s2 $Q/${A}__plem_c1b1s2_tts2 > $O/utmos_paired.json"
  exit 0
fi
[ -z "$EXIST" ] || { log "拒绝覆盖已有输出:$EXIST"; exit 1; }
[ $REFOK = 1 ] || { log "references missing: not started"; exit 1; }
L=logs/mist_w3po_v1/$RUN.log
until [ -e $L ] && grep -q CRF_RUN_EXIT $L; do sleep 30; done
log "$RUN exit: $(grep CRF_RUN_EXIT $L)"
if ! grep -q 'CRF_RUN_EXIT rc=0' $L || [ ! -e $CK/DONE ]; then log "$RUN did not finish cleanly: no evaluation (slot kept)"; exit 1; fi
[ -e $SL/$SLOT.owner ] && grep -q "$RUN" $SL/$SLOT.owner && \
  command mv -f $SL/$SLOT.owner $SL/$SLOT.released_$(date -u +%Y%m%dT%H%M%SZ) && log "$SLOT released"
mkdir -p $E $LP
setsid nohup $DCMD < /dev/null > $E/$TAG.launch.log 2>&1 &
log "dev decode orchestrator pid $!: $DCMD"
sleep 20
[ -e $E/$TAG/orch.log ] || { log "decode orchestrator did not start (see $E/$TAG.launch.log)"; exit 1; }
wait_done $E/$TAG
clean_done $E/$TAG || { log "decode orchestrator DONE with failed/skipped jobs: $(grep 'DONE;' $E/$TAG/orch.log); stop"; exit 1; }
log "decode DONE"
if $MCMD > $MLOG 2>&1; then
  log "merged: $(tail -n 1 $MLOG | cut -c1-300)"
else
  MJ=$DEC/${F}_rl/merge_read_segments.json
  PL=$(jget $MJ pair_lost); PO=$(jget $MJ pair_ok); WN=$(jget $MJ windows)
  HL=$(wc -l < $DEC/${F}_rl/hyp.jsonl 2>/dev/null || echo 0)
  if grep -q 'pairing lost' $MLOG && [ "$WN" = 3618 ] && [ "$HL" = 3618 ] && [ -s $DEC/${F}_rl/score_text.json ] \
     && [ "$PO" != missing ] && [ "$PO" -ge 3589 ]; then
    log "merge stopped on pairing lost only (pair_lost $PL, pair_ok $PO / $WN, merged copy complete): continuing under the coordinator's pre-authorisation (known deviation)"
  else
    log "merge failed for another reason or below the line (pair_lost $PL, pair_ok $PO / $WN, hyp lines $HL; see $MLOG): stop, report to the coordinator"
    exit 1
  fi
fi
setsid nohup bash -c "bash experiments/mist_w3po_v1/scripts/t_dev5_cut.sh $F && $PY experiments/mist_w3po_v1/scripts/v4a_dev5_go.py $F $TAG" \
  < /dev/null > $LP/dev5cut_go.log 2>&1 &
log "dev5 cut + GO pid $!"
setsid nohup $PCMD < /dev/null > $E/${TAG}rlplem.launch.log 2>&1 &
log "PL-EM speech orchestrator pid $!: $PCMD"
sleep 20
[ -e $E/${TAG}rlplem/orch.log ] || { log "speech orchestrator did not start (see $E/${TAG}rlplem.launch.log)"; exit 1; }
wait_done $E/${TAG}rlplem
log "PL-EM speech DONE: $(grep 'DONE;' $E/${TAG}rlplem/orch.log)"
mkdir -p $O
for n in $N ${N}_tts2; do
  CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.speech_gap_decomp --run $DEC/$n --speech $SP/$n --win $W \
    --out $O/gap_$n > $O/gap_$n.log 2>&1 &
done
CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.talker_block_pauses --talker $SP/$N/talker \
  --talker $SP/${N}_tts2/talker > $O/block_pauses_$TAG.jsonl 2>> $O/block_pauses_$TAG.log
log "block/pause stats rc=$?"
for n in $N ${N}_tts2; do
  c=$(card); until [ -n "$c" ]; do sleep 60; c=$(card); done
  CUDA_VISIBLE_DEVICES=$c $PY -m experiments.mist_w3po_v1.talker.talker_quality --talker $SP/$n/talker \
    --items $SP/$n/talker_in/items.jsonl --out $Q/$n/quality.json > $SP/$n/quality.log 2>&1
  log "quality $n on card $c rc=$?"
done
$PY experiments/mist_w3po_v1/scripts/pmc_listen_compare.py --cand $Q/$N $Q/${N}_tts2 \
  --ref ve1rl_plem $Q/$V1 $Q/${V1}_tts2 --ref aprime_rl_plem $Q/${A}__rlplem $Q/${A}__rlplem_tts2 \
  --ref aprime_plem $Q/${A}__plem_c1b1s2 $Q/${A}__plem_c1b1s2_tts2 > $O/utmos_paired.json 2> $O/utmos_paired.log
log "UTMOS paired rc=$?"
wait
log "gap jobs done: $(ls $O/gap_${N}*.json 2>/dev/null | tr '\n' ' ')"
log "dev5 GO: $(cat eval_outputs/mist_w3po_v1/crf/t_eval/dev5cut_$F/go_$TAG.json 2>/dev/null | head -c 400)"
log "V4A_ALL_DONE $TAG"
