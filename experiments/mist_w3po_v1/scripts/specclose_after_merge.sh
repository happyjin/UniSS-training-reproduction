#!/usr/bin/env bash
# SpecClose: the rest of specclose_eval.sh after its merge step, ONLY for the case that step stopped because
# crf/merge_read_segments.py exits 1 on pair_lost > 0 with the merged copy decode/dev_specclose_aprime_s3000_rl written
# in full (3624 windows), and only after the coordinator / user decided to continue (2026-11-12: pair_lost 3, pair_ok
# 3587 / 3624; neither the built-in line nor the user's "merge lost <= 2" rule held). PL_WANT = the pair_lost value the
# coordinator saw (explicit, so the launch cannot run on another failure). Same precedent as pmc_after_merge.sh.
# Steps after the merge in specclose_eval.sh copied verbatim (dev5 cut + GO in the background; PL-EM speech chain on
# CARDS; attribution error; LAAL breakdown; listening metrics; paired UTMOS).
#   DECISION="<coordinator decision>" PL_WANT=<n> [CARDS="4 5 6 7"] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/specclose_after_merge.sh
# Run detached: setsid nohup bash .../specclose_after_merge.sh > logs/mist_w3po_v1/specclose/after_merge.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
export TMPDIR=/opt/dlami/nvme/jasonleeeli/work/mist_w3po_v1/tmp HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface \
  TORCH_HOME=/opt/dlami/nvme/jasonleeeli/cache/torch XDG_CACHE_HOME=/opt/dlami/nvme/jasonleeeli/cache \
  PYTHONPATH=.:third_party/Megatron-LM OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
CARDS=${CARDS:-"4 5 6 7"}
EXCL=$(for c in 0 1 2 3 4 5 6 7; do echo " $CARDS " | grep -q " $c " || echo -n "$c "; done)
E=logs/mist_w3po_v1/t_eval
LP=logs/mist_w3po_v1/specclose
DEC=eval_outputs/mist_w3po_v1/decode
SP=eval_outputs/mist_w3po_v1/speech
Q=eval_outputs/mist_w3po_v1/talker_crf/quality
DV=eval_outputs/mist_w3po_v1/detector_v1
O=$DV/eval_v1
WS=$DV/windows_specclose_v1/windows_dev_online.jsonl
C2=eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1/windows_dev_online.jsonl
PRE=$DV/specclose_precheck_v1.json
G=logs/mist_w3po_v1/talker_crf_v1/exclude_cards
CK=eval_outputs/mist_w3po_v1/crf/train/crf_v2p_20261110T075131Z/ckpt/step_03000
F=dev_specclose_aprime_s3000
N=${F}__rlplem
A=dev_crf_v2p_s3000_finalpost_s8_c2g06
EX="--cap135 --exclude $EXCL"
ORCH="$PY experiments/mist_w3po_v1/scripts/specclose_eval_orch.py"
REFS="--gate-ref $A ${A}_tts2 --gate-ref ${A}__rlplem ${A}__rlplem_tts2 --gate-ref ${A}__plem_c1b1s2 ${A}__plem_c1b1s2_tts2"
DCMD="$ORCH --run $E/specclose --name $F --ckpt $CK --talker none --window 1 --win $WS --scorer-port0 19601 $EX"
PCMD="$ORCH --run $E/specclose_rlplem --name $N --ckpt - --speech-from ${F}_rl --talker plem --window 1 --win $WS $REFS --scorer-port0 19611 $EX"
MCMD="$PY -m experiments.mist_w3po_v1.crf.merge_read_segments --src $F --dst ${F}_rl"
GCMD="$PY experiments/mist_w3po_v1/scripts/specclose_dev5_go.py $F $WS"
ACMD="$PY -m experiments.mist_w3po_v1.evaluation.attribution_error --cand $DEC/$F --cand-win $WS --ref $DEC/$A --ref-win $C2 --out $O/attribution_vs_aprime.json"
LBPY=eval_outputs/mist_w3po_v1/speech_laal_v1/analysis_20261111/laal_breakdown.py
log() { echo "$(date -u +%FT%TZ) $*"; }
wait_done() { until [ -e $1/events/DONE ]; do sleep 30; done; }
clean_done() { grep -q 'DONE; failed/skipped \[\]' $1/orch.log; }
card() { local c; for c in $CARDS; do grep -qw $c $G 2>/dev/null || { echo $c; return; }; done; }
DECISION=${DECISION:?the coordinator decision text}
PL_WANT=${PL_WANT:?pair_lost the coordinator saw}
MJ=$DEC/${F}_rl/merge_read_segments.json
PL=$(python3 -c "import json; print(json.load(open('$MJ'))['pair_lost'])" 2>/dev/null || echo missing)
HL=$(wc -l < $DEC/${F}_rl/hyp.jsonl 2>/dev/null || echo 0)
WN=$(python3 -c "import json; print(json.load(open('$MJ'))['windows'])" 2>/dev/null || echo missing)
EXIST=""
for d in $E/specclose_rlplem $DEC/$N $DEC/${N}_tts2 $SP/$N $SP/${N}_tts2 $DEC/dev5cut_$F eval_outputs/mist_w3po_v1/crf/t_eval/dev5cut_$F \
         $Q/$N $Q/${N}_tts2 $O/attribution_vs_aprime.json $O/utmos_paired.json; do [ -e $d ] && EXIST="$EXIST $d"; done
if [ -n "${DRY_RUN:-}" ]; then
  echo "decision: $DECISION; pair_lost $PL (want $PL_WANT); merged hyp lines $HL / windows $WN; score_text $([ -s $DEC/${F}_rl/score_text.json ] && echo ok || echo missing)"
  echo "existing outputs (must be none):${EXIST:- none}; cards $CARDS -> --exclude $EXCL; ports 19611-19614 in use: $(ss -ltn 2>/dev/null | grep -cE ':1961[1-4] ')"
  echo "dev5 GO: $GCMD"; echo "speech PL-EM: $PCMD"; echo "attribution: $ACMD"; echo "listening card now: $(card)"
  exit 0
fi
[ "$PL" = "$PL_WANT" ] && [ "$HL" = "$WN" ] && [ -s $DEC/${F}_rl/score_text.json ] && grep -q 'pairing lost' $E/specclose_merge.log \
  || { log "merge state is not the one the coordinator decided on (pair_lost $PL, want $PL_WANT; hyp $HL / $WN): stop"; exit 1; }
[ -z "$EXIST" ] || { log "拒绝覆盖已有输出:$EXIST"; exit 1; }
log "continuing after the merge on the coordinator's decision (known deviation: pair_lost $PL): $DECISION"
log "merged: $(cat $MJ | tr -d '\n ' | cut -c1-300)"
setsid nohup $GCMD < /dev/null > $LP/dev5_go.log 2>&1 &
log "dev5 cut + GO pid $!"
setsid nohup $PCMD < /dev/null > $E/specclose_rlplem.launch.log 2>&1 &
log "PL-EM speech orchestrator pid $!: $PCMD"
CUDA_VISIBLE_DEVICES= $ACMD > $O/attribution_vs_aprime.log 2>&1
log "attribution rc=$?"
wait_done $E/specclose_rlplem
log "PL-EM speech DONE: $(grep 'DONE;' $E/specclose_rlplem/orch.log)"
CUDA_VISIBLE_DEVICES= $PY $LBPY $DEC/$N $SP/$N/talker $WS > $O/laal_breakdown_$N.txt 2>&1
log "laal breakdown rc=$?"
for n in $N ${N}_tts2; do
  CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.speech_gap_decomp --run $DEC/$n --speech $SP/$n --win $WS \
    --out $O/gap_$n > $O/gap_$n.log 2>&1 &
done
CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.talker_block_pauses --talker $SP/$N/talker \
  --talker $SP/${N}_tts2/talker > $O/block_pauses_specclose.jsonl 2>> $O/block_pauses_specclose.log
log "block/pause stats rc=$?"
for n in $N ${N}_tts2; do
  c=$(card); until [ -n "$c" ]; do sleep 60; c=$(card); done
  CUDA_VISIBLE_DEVICES=$c $PY -m experiments.mist_w3po_v1.talker.talker_quality --talker $SP/$n/talker \
    --items $SP/$n/talker_in/items.jsonl --out $Q/$n/quality.json > $SP/$n/quality.log 2>&1
  log "quality $n on card $c rc=$?"
done
$PY experiments/mist_w3po_v1/scripts/pmc_listen_compare.py --cand $Q/$N $Q/${N}_tts2 \
  --ref aprime_rl_plem $Q/${A}__rlplem $Q/${A}__rlplem_tts2 \
  --ref aprime_plem $Q/${A}__plem_c1b1s2 $Q/${A}__plem_c1b1s2_tts2 > $O/utmos_paired.json 2> $O/utmos_paired.log
log "UTMOS paired rc=$?"
wait
log "gap jobs done: $(ls $O/gap_${N}*.json 2>/dev/null | tr '\n' ' ')"
log "dev5 GO: $(cat eval_outputs/mist_w3po_v1/crf/t_eval/dev5cut_$F/go_specclose.json 2>/dev/null | head -c 400)"
log "SPECCLOSE_ALL_DONE"
