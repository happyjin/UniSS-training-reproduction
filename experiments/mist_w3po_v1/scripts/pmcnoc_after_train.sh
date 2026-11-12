#!/usr/bin/env bash
# After the PMC-noC training (reports/mist_w3po_v1/thinker_pmc_noc_prereg_v1.zh-CN.md §3): pmc_after_train.sh with new
# names / dirs / ports and cards 0-3 (slotA, --exclude 4 5 6 7); everything else verbatim:
#   1. wait for CRF_RUN_EXIT; rc=0 with step_03000/DONE, else stop (slot kept for inspection); release the slot (rename)
#   2. dev decode (8 shards, 1.0 s reads, --talker none)                                  run t_eval/pmcnoc (port 19501)
#   3. read-level merged copy (crf/merge_read_segments.py) -> dev_pmcnoc_s3000_rl; a pair_lost-only stop is handled by
#      the coordinator's pre-authorisation (pmcnoc_after_merge.sh)
#   4. dev5 cut + PMC's GO lines (descriptive for PMC-noC) in the background
#   5. PL-EM speech chain on the merged copy, 2 TTS seeds; gates vs A', A'rl + PL-EM, VE1rl + PL-EM (all descriptive;
#      the primary PMC - PMC-noC comparison is run on the CPU afterwards)                run t_eval/pmcnocrlplem (19511)
#   6. listening -> crf/t_eval/listen_pmcnoc_v1/ (UTMOS / SIM on ONE non-training card, first of 0..7 not excluded)
#   RUN=<training run id> [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/pmcnoc_after_train.sh
# Run detached: setsid nohup bash .../pmcnoc_after_train.sh > logs/mist_w3po_v1/pmcnoc/after_train.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
RUN=${RUN:?}
export TMPDIR=/opt/dlami/nvme/jasonleeeli/work/mist_w3po_v1/tmp HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface \
  TORCH_HOME=/opt/dlami/nvme/jasonleeeli/cache/torch XDG_CACHE_HOME=/opt/dlami/nvme/jasonleeeli/cache \
  PYTHONPATH=.:third_party/Megatron-LM OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
SL=logs/mist_w3po_v1/slots
SLOT=slotA
E=logs/mist_w3po_v1/t_eval
LP=logs/mist_w3po_v1/pmcnoc
DEC=eval_outputs/mist_w3po_v1/decode
SP=eval_outputs/mist_w3po_v1/speech
Q=eval_outputs/mist_w3po_v1/talker_crf/quality
O=eval_outputs/mist_w3po_v1/crf/t_eval/listen_pmcnoc_v1
W=eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1/windows_dev_online.jsonl
G=logs/mist_w3po_v1/talker_crf_v1/exclude_cards
CK=eval_outputs/mist_w3po_v1/crf/train/$RUN/ckpt/step_03000
F=dev_pmcnoc_s3000
N=${F}__rlplem
A=dev_crf_v2p_s3000_finalpost_s8_c2g06
V1=dev_ve1_s3000__rlplem
EX="--cap135 --exclude 4 5 6 7"
ORCH="$PY experiments/mist_w3po_v1/scripts/thinker_t_eval_orch.py"
REFS="--gate-ref $A ${A}_tts2 --gate-ref ${A}__rlplem ${A}__rlplem_tts2 --gate-ref $V1 ${V1}_tts2"
DCMD="$ORCH --run $E/pmcnoc --name $F --ckpt $CK --talker none --window 1 --scorer-port0 19501 $EX"
PCMD="$ORCH --run $E/pmcnocrlplem --name $N --ckpt - --speech-from ${F}_rl --talker plem --window 1 $REFS --scorer-port0 19511 $EX"
MCMD="$PY -m experiments.mist_w3po_v1.crf.merge_read_segments --src $F --dst ${F}_rl"
log() { echo "$(date -u +%FT%TZ) $*"; }
wait_done() { until [ -e $1/events/DONE ]; do sleep 30; done; }
clean_done() { grep -q 'DONE; failed/skipped \[\]' $1/orch.log; }
card() { local c; for c in 0 1 2 3 4 5 6 7; do grep -qw $c $G 2>/dev/null || { echo $c; return; }; done; }
EXIST=""
for d in $E/pmcnoc $E/pmcnocrlplem $DEC/$F $DEC/${F}_rl $DEC/$N $DEC/${N}_tts2 $DEC/dev5cut_$F \
         eval_outputs/mist_w3po_v1/crf/t_eval/dev5cut_$F $Q/$N $Q/${N}_tts2 $O; do [ -e $d ] && EXIST="$EXIST $d"; done
REFOK=1
for r in $A ${A}_tts2 ${A}__rlplem ${A}__rlplem_tts2 $V1 ${V1}_tts2; do
  [ -s $DEC/$r/score_asr.json ] || { REFOK=0; echo "missing gate reference $DEC/$r/score_asr.json"; }
done
for r in ${A}__plem_c1b1s2 ${A}__plem_c1b1s2_tts2 ${A}__rlplem ${A}__rlplem_tts2 $V1 ${V1}_tts2; do
  [ -s $Q/$r/quality.json.items.jsonl ] || { REFOK=0; echo "missing UTMOS reference $Q/$r"; }
done
if [ -n "${DRY_RUN:-}" ]; then
  echo "run log: logs/mist_w3po_v1/$RUN.log ($([ -e logs/mist_w3po_v1/$RUN.log ] && echo present || echo 'not yet')); checkpoint: $CK"
  echo "existing outputs (must be none):${EXIST:- none}; references ok: $REFOK"
  echo "ports 19501-19524 in use: $(ss -ltn 2>/dev/null | grep -cE ':(1950[1-9]|1951[0-9]|1952[0-4]) ')"
  echo "decode: $DCMD"; echo "merge: $MCMD"
  echo "dev5 cut + GO: bash experiments/mist_w3po_v1/scripts/t_dev5_cut.sh $F && $PY experiments/mist_w3po_v1/scripts/pmc_dev5_go.py $F"
  echo "speech PL-EM: $PCMD"
  echo "listening card now: $(card)"
  for n in $N ${N}_tts2; do
    echo "quality: CUDA_VISIBLE_DEVICES=<card> $PY -m experiments.mist_w3po_v1.talker.talker_quality --talker $SP/$n/talker --items $SP/$n/talker_in/items.jsonl --out $Q/$n/quality.json"
    echo "gap: $PY -m experiments.mist_w3po_v1.evaluation.speech_gap_decomp --run $DEC/$n --speech $SP/$n --win $W --out $O/gap_$n"
  done
  echo "pauses: $PY -m experiments.mist_w3po_v1.evaluation.talker_block_pauses --talker $SP/$N/talker --talker $SP/${N}_tts2/talker > $O/block_pauses_pmcnoc.jsonl"
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
setsid nohup $DCMD < /dev/null > $E/pmcnoc.launch.log 2>&1 &
log "dev decode orchestrator pid $!: $DCMD"
sleep 20
[ -e $E/pmcnoc/orch.log ] || { log "decode orchestrator did not start (see $E/pmcnoc.launch.log)"; exit 1; }
wait_done $E/pmcnoc
clean_done $E/pmcnoc || { log "decode orchestrator DONE with failed/skipped jobs: $(grep 'DONE;' $E/pmcnoc/orch.log); stop"; exit 1; }
log "decode DONE"
$MCMD > $E/pmcnoc_merge.log 2>&1 || { log "merge failed (see $E/pmcnoc_merge.log)"; exit 1; }
log "merged: $(tail -n 1 $E/pmcnoc_merge.log | cut -c1-300)"
setsid nohup bash -c "bash experiments/mist_w3po_v1/scripts/t_dev5_cut.sh $F && $PY experiments/mist_w3po_v1/scripts/pmc_dev5_go.py $F" \
  < /dev/null > $LP/dev5cut_go.log 2>&1 &
log "dev5 cut + GO pid $!"
setsid nohup $PCMD < /dev/null > $E/pmcnocrlplem.launch.log 2>&1 &
log "PL-EM speech orchestrator pid $!: $PCMD"
wait_done $E/pmcnocrlplem
log "PL-EM speech DONE: $(grep 'DONE;' $E/pmcnocrlplem/orch.log)"
mkdir -p $O
for n in $N ${N}_tts2; do
  CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.speech_gap_decomp --run $DEC/$n --speech $SP/$n --win $W \
    --out $O/gap_$n > $O/gap_$n.log 2>&1 &
done
CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.talker_block_pauses --talker $SP/$N/talker \
  --talker $SP/${N}_tts2/talker > $O/block_pauses_pmcnoc.jsonl 2>> $O/block_pauses_pmcnoc.log
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
log "dev5 GO: $(cat eval_outputs/mist_w3po_v1/crf/t_eval/dev5cut_$F/go_pmc.json 2>/dev/null | head -c 400)"
log "PMCNOC_ALL_DONE"
