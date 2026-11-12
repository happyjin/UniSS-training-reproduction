#!/usr/bin/env bash
# SpecClose dev evaluation (reports/mist_w3po_v1/thinker_detector_specclose_prereg_v1.zh-CN.md §3-§7). Eval-type job,
# no training: A' (v2' step_03000, 1.0 s reads) decoded on the SpecClose dev windows, then the A'rl + PL-EM speech chain.
#   0. precheck: detector_v1/specclose_precheck_v1.json GO, or ALLOW_NOGO="<coordinator decision, logged>" (the
#      registered precheck said NOT GO; a run is then an exploratory, descriptive evaluation)
#   1. dev decode, 8 shards, --talker none (decode + text score + text latency) on the SpecClose windows,
#      decode module crf/decode_crf_ckpt_specclose (specclose_eval_orch.py)               run t_eval/specclose
#   2. read-level merged copy (crf/merge_read_segments.py) -> <decode>_rl; a stop for pair_lost only, with the merged
#      copy written and pair_ok / windows >= 3589 / 3618, continues (the PMC pre-authorisation rule; logged)
#   3. dev5 cut + GO lines (specclose_dev5_go.py, CPU) in the background; the speech chain does not wait
#   4. PL-EM speech chain on the merged copy, 2 TTS seeds; gates: gate_0 vs A' (text gate; speech vs A' c1b1s2),
#      gate_1 vs A'rl + PL-EM (primary speech gate), gate_2 vs A' + PL-EM (descriptive)    run t_eval/specclose_rlplem
#   5. CPU metrics: attribution error vs A' (evaluation/attribution_error.py); speech LAAL by unit duration
#      (laal_breakdown.py, seed 1); listening: talker WER (speech_gap_decomp), block / pause stats, UTMOS / SIM
#      (talker_quality on ONE card of CARDS not in the shared exclusion file), paired UTMOS vs A'rl + PL-EM and A' + PL-EM
# CARDS = the cards the coordinator gives this job (default "4 5 6 7"); every other card is passed as --exclude, and the
# shared exclusion file (training slots) is honoured by the orchestrators as always (--cap135: <= 135 GiB per card,
# two decodes per card, 60 s stagger). Ports 19601 (decode) and 19611-19614 (speech scorers).
#   CARDS="4 5 6 7" [ALLOW_NOGO="..."] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/specclose_eval.sh
# Run detached: setsid nohup bash .../specclose_eval.sh > logs/mist_w3po_v1/specclose/eval.log 2>&1 < /dev/null &
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
GO=$(python3 -c "import json; print(json.load(open('$PRE'))['GO'])" 2>/dev/null || echo missing)
EXIST=""
for d in $E/specclose $E/specclose_rlplem $DEC/$F $DEC/${F}_rl $DEC/$N $DEC/${N}_tts2 $SP/$N $SP/${N}_tts2 $DEC/dev5cut_$F \
         eval_outputs/mist_w3po_v1/crf/t_eval/dev5cut_$F $Q/$N $Q/${N}_tts2 $O; do [ -e $d ] && EXIST="$EXIST $d"; done
REFOK=1
for r in $A ${A}_tts2 ${A}__rlplem ${A}__rlplem_tts2 ${A}__plem_c1b1s2 ${A}__plem_c1b1s2_tts2; do
  [ -s $DEC/$r/score_asr.json ] || { REFOK=0; echo "missing gate reference $DEC/$r/score_asr.json"; }
done
for r in ${A}__rlplem ${A}__rlplem_tts2 ${A}__plem_c1b1s2 ${A}__plem_c1b1s2_tts2; do
  [ -s $Q/$r/quality.json.items.jsonl ] || { REFOK=0; echo "missing UTMOS reference $Q/$r"; }
done
[ -s $WS ] && [ -e $CK/DONE ] || { REFOK=0; echo "missing windows $WS or checkpoint $CK"; }
if [ -n "${DRY_RUN:-}" ]; then
  echo "cards: $CARDS -> --exclude $EXCL; precheck GO: $GO; ALLOW_NOGO: ${ALLOW_NOGO:-unset}"
  echo "existing outputs (must be none):${EXIST:- none}; references ok: $REFOK"
  echo "ports 19601-19614 in use: $(ss -ltn 2>/dev/null | grep -cE ':(1960[0-9]|1961[0-4]) ')"
  echo "decode: $DCMD"; $DCMD --dry | tail -1
  echo "merge: $MCMD"; echo "dev5 GO: $GCMD"
  echo "speech PL-EM: $PCMD"
  echo "attribution: $ACMD"
  echo "laal breakdown: $PY $LBPY $DEC/$N $SP/$N/talker $WS > $O/laal_breakdown_$N.txt"
  echo "listening card now: $(card)"
  for n in $N ${N}_tts2; do
    echo "quality: CUDA_VISIBLE_DEVICES=<card> $PY -m experiments.mist_w3po_v1.talker.talker_quality --talker $SP/$n/talker --items $SP/$n/talker_in/items.jsonl --out $Q/$n/quality.json"
    echo "gap: $PY -m experiments.mist_w3po_v1.evaluation.speech_gap_decomp --run $DEC/$n --speech $SP/$n --win $WS --out $O/gap_$n"
  done
  echo "pauses: $PY -m experiments.mist_w3po_v1.evaluation.talker_block_pauses --talker $SP/$N/talker --talker $SP/${N}_tts2/talker > $O/block_pauses_specclose.jsonl"
  echo "utmos paired: $PY experiments/mist_w3po_v1/scripts/pmc_listen_compare.py --cand $Q/$N $Q/${N}_tts2 --ref aprime_rl_plem $Q/${A}__rlplem $Q/${A}__rlplem_tts2 --ref aprime_plem $Q/${A}__plem_c1b1s2 $Q/${A}__plem_c1b1s2_tts2 > $O/utmos_paired.json"
  exit 0
fi
[ -z "$EXIST" ] || { log "拒绝覆盖已有输出:$EXIST"; exit 1; }
[ $REFOK = 1 ] || { log "references missing: not started"; exit 1; }
if [ "$GO" != "True" ]; then
  [ -n "${ALLOW_NOGO:-}" ] || { log "precheck GO = $GO and no ALLOW_NOGO: not started"; exit 1; }
  log "precheck GO = $GO; started on the coordinator's decision (exploratory, descriptive): $ALLOW_NOGO"
fi
mkdir -p $E $LP $O
setsid nohup $DCMD < /dev/null > $E/specclose.launch.log 2>&1 &
log "dev decode orchestrator pid $!: $DCMD"
sleep 20
[ -e $E/specclose/orch.log ] || { log "decode orchestrator did not start (see $E/specclose.launch.log)"; exit 1; }
wait_done $E/specclose
clean_done $E/specclose || { log "decode orchestrator DONE with failed/skipped jobs: $(grep 'DONE;' $E/specclose/orch.log); stop"; exit 1; }
log "decode DONE"
if ! $MCMD > $E/specclose_merge.log 2>&1; then
  MJ=$DEC/${F}_rl/merge_read_segments.json
  OKM=$(python3 -c "
import json; m=json.load(open('$MJ')); print(int(m['pair_lost'] > 0 and m['pair_ok'] * 3618 >= 3589 * m['windows']))" 2>/dev/null || echo 0)
  if grep -q 'pairing lost' $E/specclose_merge.log && [ -s $DEC/${F}_rl/hyp.jsonl ] && [ "$OKM" = 1 ]; then
    log "merge stopped on pair_lost only; known deviation, continuing (PMC pre-authorisation rule): $(cat $MJ | tr -d '\n ' | cut -c1-200)"
  else
    log "merge failed (see $E/specclose_merge.log): stop"; exit 1
  fi
fi
log "merged: $(tail -n 1 $E/specclose_merge.log | cut -c1-300)"
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
