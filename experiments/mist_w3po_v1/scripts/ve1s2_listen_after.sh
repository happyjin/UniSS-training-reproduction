#!/usr/bin/env bash
# VE1-s2 (reports/mist_w3po_v1/thinker_ve1_s2_confirm_prereg_v1.zh-CN.md §3, descriptive): the listening metrics and the
# >= 10 s splits after the read-level PL-EM speech chain (coordinator 2026-11-12 ~08:00 Prague: "the same listening metrics").
# Waits for <run>/events/DONE of logs/mist_w3po_v1/t_eval/ve1s2rlplem (dev_ve1s2_s3000__rlplem{,_tts2}), then:
#   talker WER    evaluation/speech_gap_decomp.py (CPU)             -> crf/t_eval/listen_rl_v1/gap_<name>*
#   blocks/pauses evaluation/talker_block_pauses.py (CPU)           -> crf/t_eval/listen_rl_v1/block_pauses_ve1s2.jsonl
#   >= 10 s text  evaluation/early_commit_diag --light vs A' (CPU)   -> crf/t_eval/dev_ve1s2_s3000/early_commit_diag_light_fulldev.json
#   speech LAAL by unit duration: laal_breakdown.py, seed 1 (CPU)   -> crf/t_eval/dev_ve1s2_s3000__rlplem/laal_breakdown_*.txt
#   UTMOS / SIM   talker/talker_quality.py on ONE non-training card (first of 0..7 not in the shared exclusion file =
#                 training slots; waits while all are excluded); both seeds one after the other
# Refuses existing outputs.   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/ve1s2_listen_after.sh
# Run detached: setsid nohup bash .../ve1s2_listen_after.sh > logs/mist_w3po_v1/ve1s2/listen_after.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
export TMPDIR=/opt/dlami/nvme/jasonleeeli/work/mist_w3po_v1/tmp HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface \
  TORCH_HOME=/opt/dlami/nvme/jasonleeeli/cache/torch XDG_CACHE_HOME=/opt/dlami/nvme/jasonleeeli/cache \
  PYTHONPATH=.:third_party/Megatron-LM OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
R=logs/mist_w3po_v1/t_eval/ve1s2rlplem
F=dev_ve1s2_s3000
N=${F}__rlplem
A=dev_crf_v2p_s3000_finalpost_s8_c2g06
D=eval_outputs/mist_w3po_v1/decode
S=eval_outputs/mist_w3po_v1/speech
Q=eval_outputs/mist_w3po_v1/talker_crf/quality
T=eval_outputs/mist_w3po_v1/crf/t_eval
O=$T/listen_rl_v1
W=eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1/windows_dev_online.jsonl
G=logs/mist_w3po_v1/talker_crf_v1/exclude_cards
BP=$O/block_pauses_ve1s2.jsonl
EC=$T/$F/early_commit_diag_light_fulldev
LB=$T/$N/laal_breakdown_$N.txt
LBPY=eval_outputs/mist_w3po_v1/speech_laal_v1/analysis_20261111/laal_breakdown.py
log() { echo "$(date -u +%FT%TZ) $*"; }
card() { local c; for c in 0 1 2 3 4 5 6 7; do grep -qw $c $G 2>/dev/null || { echo $c; return; }; done; }
EXIST=""
for n in $N ${N}_tts2; do for p in $Q/$n $O/gap_$n.json $O/gap_$n.log; do [ -e $p ] && EXIST="$EXIST $p"; done; done
for p in $BP $EC.json $EC.log $LB; do [ -e $p ] && EXIST="$EXIST $p"; done
ECMD="$PY -m experiments.mist_w3po_v1.evaluation.early_commit_diag --light --run $D/$A --run $D/$F --ref-run $D/$A --win $W --out $EC.json"
LCMD="$PY $LBPY $D/$N $S/$N/talker $W"
if [ -n "${DRY_RUN:-}" ]; then
  echo "orchestrator run $R: $([ -e $R/events/DONE ] && echo DONE || echo 'not done')"; echo "existing outputs (must be none):${EXIST:- none}"
  echo "card now: $(card)"; for n in $N ${N}_tts2; do
    echo "quality: CUDA_VISIBLE_DEVICES=<card> $PY -m experiments.mist_w3po_v1.talker.talker_quality --talker $S/$n/talker --items $S/$n/talker_in/items.jsonl --out $Q/$n/quality.json"
    echo "gap: $PY -m experiments.mist_w3po_v1.evaluation.speech_gap_decomp --run $D/$n --speech $S/$n --win $W --out $O/gap_$n"; done
  echo "pauses: $PY -m experiments.mist_w3po_v1.evaluation.talker_block_pauses --talker $S/$N/talker --talker $S/${N}_tts2/talker > $BP"
  echo "early commit: $ECMD"; echo "laal breakdown: $LCMD > $LB"
  exit 0
fi
[ -z "$EXIST" ] || { log "拒绝覆盖已有输出:$EXIST"; exit 1; }
until [ -e $R/events/DONE ]; do sleep 60; done
log "speech chain DONE: $(grep 'DONE;' $R/orch.log)"
mkdir -p $O
for n in $N ${N}_tts2; do
  CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.speech_gap_decomp --run $D/$n --speech $S/$n --win $W \
    --out $O/gap_$n > $O/gap_$n.log 2>&1 &
done
CUDA_VISIBLE_DEVICES= $ECMD > $EC.log 2>&1 &
CUDA_VISIBLE_DEVICES= $LCMD > $LB 2>&1
log "laal breakdown rc=$?"
CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.talker_block_pauses --talker $S/$N/talker --talker $S/${N}_tts2/talker > $BP 2>> $O/block_pauses_ve1s2.log
log "block/pause stats rc=$?"
for n in $N ${N}_tts2; do
  c=$(card); until [ -n "$c" ]; do sleep 60; c=$(card); done
  CUDA_VISIBLE_DEVICES=$c $PY -m experiments.mist_w3po_v1.talker.talker_quality --talker $S/$n/talker \
    --items $S/$n/talker_in/items.jsonl --out $Q/$n/quality.json > $S/$n/quality.log 2>&1
  log "quality $n on card $c rc=$?"
done
wait
log "gap jobs done: $(ls $O/gap_${N}*.json 2>/dev/null | tr '\n' ' '); early commit: $(tail -n 1 $EC.log | cut -c1-200)"
log "VE1S2_LISTEN_DONE"
