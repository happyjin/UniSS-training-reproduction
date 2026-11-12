#!/usr/bin/env bash
# PMC-noC: the rest of pmcnoc_after_train.sh after its merge step, ONLY for the case that step stopped because
# crf/merge_read_segments.py exits 1 on pair_lost > 0 (merged copy decode/dev_pmcnoc_s3000_rl written in full). The
# coordinator's pre-authorisation (2026-11-12 ~11:00 Prague, for PMC and PMC-noC alike): continue only on a pair_lost-only
# stop with a complete merged copy and pair_ok >= 3589 / 3618; PL_WANT = the pair_lost value seen (explicit).
# Steps 4-6 of pmcnoc_after_train.sh verbatim (= pmc_after_merge.sh with PMC-noC's names / dirs / ports / cards 0-3).
#   RUN=<training run id> PL_WANT=<n> [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/pmcnoc_after_merge.sh
# Run detached: setsid nohup bash .../pmcnoc_after_merge.sh > logs/mist_w3po_v1/pmcnoc/after_merge.log 2>&1 < /dev/null &
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
PL_WANT=${PL_WANT:?}
EXIST=""
for d in $E/pmcnocrlplem $DEC/$N $DEC/${N}_tts2 $DEC/dev5cut_$F eval_outputs/mist_w3po_v1/crf/t_eval/dev5cut_$F $Q/$N $Q/${N}_tts2 $O \
         $LP/dev5cut_go.log; do [ -e $d ] && EXIST="$EXIST $d"; done
MJ=$DEC/${F}_rl/merge_read_segments.json
PL=$(python3 -c "import json; print(json.load(open('$MJ'))['pair_lost'])" 2>/dev/null || echo missing)
PO=$(python3 -c "import json; d = json.load(open('$MJ')); print(d['pair_ok'], '/', d['windows'])" 2>/dev/null || echo missing)
DONE_OK=0; grep -q 'DONE; failed/skipped \[\]' $E/pmcnoc/orch.log 2>/dev/null && DONE_OK=1
STOPPED=0; grep -q 'merge failed' $LP/after_train.log 2>/dev/null && grep -q 'pairing lost' $E/pmcnoc_merge.log 2>/dev/null && STOPPED=1
if [ -n "${DRY_RUN:-}" ]; then
  echo "decode run clean: $DONE_OK; after_train stopped at merge on pair_lost: $STOPPED; pair_lost: $PL (want $PL_WANT); pair_ok $PO"
  echo "existing outputs (must be none):${EXIST:- none}; ports 19511-19514 in use: $(ss -ltn 2>/dev/null | grep -cE ':1951[1-4] ')"
  echo "dev5 cut + GO: bash experiments/mist_w3po_v1/scripts/t_dev5_cut.sh $F && $PY experiments/mist_w3po_v1/scripts/pmc_dev5_go.py $F"
  echo "speech PL-EM: $PCMD"
  exit 0
fi
[ -z "$EXIST" ] || { log "拒绝覆盖已有输出:$EXIST"; exit 1; }
[ $DONE_OK = 1 ] && [ $STOPPED = 1 ] && [ "$PL" = "$PL_WANT" ] || { log "decode clean $DONE_OK / stopped on pair_lost $STOPPED / pair_lost $PL (want $PL_WANT): not started"; exit 1; }
log "continuing on the merged copy (pair_lost $PL, pair_ok $PO)"
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
