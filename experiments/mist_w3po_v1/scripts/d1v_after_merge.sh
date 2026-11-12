#!/usr/bin/env bash
# D1v: the rest of v4a_after_train.sh (TAG=d1v) after its merge step, ONLY for the case that step stopped because
# crf/merge_read_segments.py exits 1 on pair_lost > 0 with the merged copy decode/dev_d1v_s3000_rl written in full
# (3618 windows, score_text.json) and pair_ok < 3589, under the user's pre-authorisation registered in
# thinker_dagger_d1v_prereg_v1 amendment 1 (last bullet, 2026-11-12 14:05): the merge itself lost <= 2 pairs ->
# continue, the windows listed as a known deviation. Any other state -> refuse (ask the coordinator).
# Steps after the merge in v4a_after_train.sh copied verbatim (dev5 cut + GO in the background; PL-EM speech chain on
# cards 0-3; listening: talker WER, block / pause stats, UTMOS / SIM on one of cards 0-3, paired UTMOS). Same
# precedent as pmc_after_merge.sh / specclose_after_merge.sh. Cards 4-7 are never used (the baseline agents' cards).
#   PL_WANT=<pair_lost seen in merge_read_segments.json> [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/d1v_after_merge.sh
# Run detached: setsid nohup bash .../d1v_after_merge.sh > logs/mist_w3po_v1/d1v/after_merge.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
TAG=d1v
export TMPDIR=/opt/dlami/nvme/jasonleeeli/work/mist_w3po_v1/tmp HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface \
  TORCH_HOME=/opt/dlami/nvme/jasonleeeli/cache/torch XDG_CACHE_HOME=/opt/dlami/nvme/jasonleeeli/cache \
  PYTHONPATH=.:third_party/Megatron-LM OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
P2=19611
EX="--cap135 --exclude 4 5 6 7"
ORDER="0 1 2 3"
E=logs/mist_w3po_v1/t_eval
LP=logs/mist_w3po_v1/$TAG
DEC=eval_outputs/mist_w3po_v1/decode
SP=eval_outputs/mist_w3po_v1/speech
Q=eval_outputs/mist_w3po_v1/talker_crf/quality
O=eval_outputs/mist_w3po_v1/crf/t_eval/listen_${TAG}_v1
W=eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1/windows_dev_online.jsonl
G=logs/mist_w3po_v1/talker_crf_v1/exclude_cards
F=dev_${TAG}_s3000
N=${F}__rlplem
A=dev_crf_v2p_s3000_finalpost_s8_c2g06
V1=dev_ve1_s3000__rlplem
ORCH="$PY experiments/mist_w3po_v1/scripts/thinker_t_eval_orch.py"
REFS="--gate-ref $A ${A}_tts2 --gate-ref ${A}__rlplem ${A}__rlplem_tts2 --gate-ref $V1 ${V1}_tts2"
PCMD="$ORCH --run $E/${TAG}rlplem --name $N --ckpt - --speech-from ${F}_rl --talker plem --window 1 $REFS --scorer-port0 $P2 $EX"
MLOG=$E/${TAG}_merge.log
MJ=$DEC/${F}_rl/merge_read_segments.json
log() { echo "$(date -u +%FT%TZ) $*"; }
wait_done() { until [ -e $1/events/DONE ]; do sleep 30; done; }
card() { local c; for c in $ORDER; do grep -qw $c $G 2>/dev/null || { echo $c; return; }; done; }
jget() { $PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get(sys.argv[2], 'missing'))" "$1" "$2" 2>/dev/null || echo missing; }
PL_WANT=${PL_WANT:?pair_lost seen in merge_read_segments.json}
PL=$(jget $MJ pair_lost); PO=$(jget $MJ pair_ok); WN=$(jget $MJ windows)
HL=$(wc -l < $DEC/${F}_rl/hyp.jsonl 2>/dev/null || echo 0)
EXIST=""
for d in $E/${TAG}rlplem $DEC/$N $DEC/${N}_tts2 $SP/$N $SP/${N}_tts2 $DEC/dev5cut_$F \
         eval_outputs/mist_w3po_v1/crf/t_eval/dev5cut_$F $Q/$N $Q/${N}_tts2 $O; do [ -e $d ] && EXIST="$EXIST $d"; done
REFOK=1
for r in $A ${A}_tts2 ${A}__rlplem ${A}__rlplem_tts2 $V1 ${V1}_tts2; do
  [ -s $DEC/$r/score_asr.json ] || { REFOK=0; echo "missing gate reference $DEC/$r/score_asr.json"; }
done
for r in ${A}__plem_c1b1s2 ${A}__plem_c1b1s2_tts2 ${A}__rlplem ${A}__rlplem_tts2 $V1 ${V1}_tts2; do
  [ -s $Q/$r/quality.json.items.jsonl ] || { REFOK=0; echo "missing UTMOS reference $Q/$r"; }
done
STATE_OK=0
[ "$PL" = "$PL_WANT" ] && [ "$PL" != missing ] && [ "$PL" -ge 1 ] && [ "$PL" -le 2 ] && [ "$WN" = 3618 ] && [ "$HL" = 3618 ] \
  && [ -s $DEC/${F}_rl/score_text.json ] && grep -q 'pairing lost' $MLOG 2>/dev/null && STATE_OK=1
if [ -n "${DRY_RUN:-}" ]; then
  echo "merge state: pair_lost $PL (want $PL_WANT, must be 1-2), pair_ok $PO / windows $WN, merged hyp lines $HL, score_text $([ -s $DEC/${F}_rl/score_text.json ] && echo ok || echo missing), 'pairing lost' in $MLOG: $(grep -c 'pairing lost' $MLOG 2>/dev/null || echo 0) -> state ok $STATE_OK"
  echo "existing outputs (must be none):${EXIST:- none}; references ok: $REFOK; ports $P2-$((P2 + 3)) in use: $(ss -ltn 2>/dev/null | grep -cE ":($P2|$((P2+1))|$((P2+2))|$((P2+3))) ")"
  echo "dev5 cut + GO: bash experiments/mist_w3po_v1/scripts/t_dev5_cut.sh $F && $PY experiments/mist_w3po_v1/scripts/v4a_dev5_go.py $F $TAG"
  echo "speech PL-EM: $PCMD"
  echo "listening card now (0-3 only): $(card)"
  exit 0
fi
[ $STATE_OK = 1 ] || { log "merge state is not the pre-authorised one (pair_lost $PL, want $PL_WANT, must be 1-2; pair_ok $PO / $WN; hyp $HL): stop, ask the coordinator"; exit 1; }
[ -z "$EXIST" ] || { log "拒绝覆盖已有输出:$EXIST"; exit 1; }
[ $REFOK = 1 ] || { log "references missing: not started"; exit 1; }
log "continuing after the merge under the user's pre-authorisation (D1v prereg amendment 1: merge lost <= 2 pairs; known deviation: pair_lost $PL, pair_ok $PO / $WN)"
log "merged: $(tr -d '\n ' < $MJ | cut -c1-300)"
mkdir -p $LP
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
