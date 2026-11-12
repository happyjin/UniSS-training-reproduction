#!/usr/bin/env bash
# Post-hoc measures of the FINAL TEST (reports/mist_w3po_v1/final_test_prereg_v1.zh-CN.md §3), called by
# final_test_chain.sh. Outputs under eval_outputs/mist_w3po_v1/crf/t_eval/final_test_v1/<set>/ (+ talker_crf/quality/<name>);
# every tool refuses to overwrite. The reference windows / universe of the set go through evaluation/on_refset.py.
#   SET=ami|ali NAME=<speech name> bash final_test_post.sh          one speech chain (both TTS seeds):
#       chunk-protocol latency (speech_latency_lit --win <chunk windows>; the old protocol is the orchestrator's lat_*),
#       talker WER etc. (speech_gap_decomp), block / pause stats (talker_block_pauses), UTMOS / SIM (talker_quality, GPU:
#       a card with >= 8 GiB free by nvidia-smi, one job at a time, waits up to 3 h)
#   SET=.. NAME=<decode name> bash final_test_post.sh textlat        text LAAL of a decode under the chunk protocol
#   SET=.. bash final_test_post.sh gates1 | gates2                   registered comparisons (gate_2seed) + paired UTMOS
# Checked with bash -n only (it runs CPU tools and one small GPU tool per seed).
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
SET=${SET:?ami or ali}
case $SET in ami) WS=zh2en; REF=data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_ref.jsonl;;
             ali) WS=ali; REF=data/diarist_alimeeting_refs_v1/windows_test_ref_human.jsonl;; *) exit 1;; esac
export PYTHONPATH=. TMPDIR=/opt/dlami/nvme/jasonleeeli/work/mist_w3po_v1/tmp HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface \
  TORCH_HOME=/opt/dlami/nvme/jasonleeeli/cache/torch XDG_CACHE_HOME=/opt/dlami/nvme/jasonleeeli/cache OMP_NUM_THREADS=8
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
DEC=eval_outputs/mist_w3po_v1/decode
SPD=eval_outputs/mist_w3po_v1/speech
Q=eval_outputs/mist_w3po_v1/talker_crf/quality
O=eval_outputs/mist_w3po_v1/crf/t_eval/final_test_v1/$SET
WIN=eval_outputs/mist_w3po_v1/data/windows_test_c2g06_v1/windows_test_online_$WS.jsonl
CW=eval_outputs/mist_w3po_v1/data/windows_test_c2g06_chunk_v1/$WS/windows_dev_online.jsonl
ONR="$PY -m experiments.mist_w3po_v1.evaluation.on_refset --ref-win $REF --dr $WS"
FA=ft_${SET}_aprime_s3000; FV=ft_${SET}_ve1_s3000; FD=ft_${SET}_d1v_s3000
NC=${FA}__rlplem; NB=${FA}__plem; NA=${FA}__c1b1s2; ND=${FV}__rlplem; NE=${FD}__rlplem
log() { echo "$(date -u +%FT%TZ) $*"; }
mkdir -p $O
gate() {   # <cand> <ref>
  CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.gate_2seed --cand $DEC/$1 $DEC/${1}_tts2 \
    --ref $DEC/$2 $DEC/${2}_tts2 --out $O/gate_${1}_vs_${2}.json > $O/gate_${1}_vs_${2}.log 2>&1
  log "gate $1 vs $2 rc=$?"
}
paired() {  # <cand> <out tag> then (<name> <ref>)...
  local c=$1 tag=$2; shift 2
  local args=""
  while [ $# -gt 0 ]; do args="$args --ref $1 $Q/$2 $Q/${2}_tts2"; shift 2; done
  CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.listen_paired_meeting --cand $Q/$c $Q/${c}_tts2 $args \
    > $O/utmos_paired_$tag.json 2> $O/utmos_paired_$tag.log
  log "paired UTMOS $tag rc=$?"
}
case ${1:-chain} in
  gates1)
    gate $NC $NA; gate $NC $NB
    paired $NC c_vs_ab a_c1b1s2 $NA b_plem $NB
    exit 0;;
  gates2)
    gate $ND $NC; gate $NE $NC
    paired $ND d_vs_c c_aprl_plem $NC
    paired $NE e_vs_c c_aprl_plem $NC
    exit 0;;
  textlat)
    CUDA_VISIBLE_DEVICES= $ONR experiments.mist_w3po_v1.evaluation.speech_latency_lit --run $DEC/$NAME --text-only --win $CW \
      --out $O/textlat_chunk_$NAME.json > $O/textlat_chunk_$NAME.log 2>&1
    log "text LAAL (chunk) $NAME rc=$?"
    exit 0;;
  chain) ;;
  *) echo "unknown mode $1"; exit 1;;
esac
NAME=${NAME:?speech name}
for n in $NAME ${NAME}_tts2; do
  [ -s $DEC/$n/score_asr.json ] || { log "$n: no score_asr.json (chain incomplete): skipped"; continue; }
  CUDA_VISIBLE_DEVICES= $ONR experiments.mist_w3po_v1.evaluation.speech_latency_lit --run $DEC/$n --talker $SPD/$n/talker \
    --win $CW --out $O/latchunk_$n.json > $O/latchunk_$n.log 2>&1 &
  CUDA_VISIBLE_DEVICES= $ONR experiments.mist_w3po_v1.evaluation.speech_gap_decomp --run $DEC/$n --speech $SPD/$n --win $WIN \
    --out $O/gap_$n > $O/gap_$n.log 2>&1 &
done
CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.talker_block_pauses --talker $SPD/$NAME/talker \
  --talker $SPD/${NAME}_tts2/talker > $O/block_pauses_$NAME.jsonl 2> $O/block_pauses_$NAME.log
log "block / pause stats rc=$?"
qcard() {
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' '$3 - $2 >= 8192 {print $3 - $2, $1}' | sort -nr | head -n 1 | awk '{print $2}'
}
LOCK=/opt/dlami/nvme/jasonleeeli/work/mist_w3po_v1/final_test_quality_$SET.lock
for n in $NAME ${NAME}_tts2; do
  [ -s $SPD/$n/talker/results.jsonl ] || { log "$n: no talker results: quality skipped"; continue; }
  (
    flock 9
    t0=$(date +%s); c=$(qcard)
    until [ -n "$c" ] || [ $(( $(date +%s) - t0 )) -gt 10800 ]; do sleep 60; c=$(qcard); done
    if [ -z "$c" ]; then log "quality $n: no card with >= 8 GiB free for 3 h: skipped"; exit 1; fi
    mkdir -p $Q/$n
    CUDA_VISIBLE_DEVICES=$c $PY -m experiments.mist_w3po_v1.talker.talker_quality --talker $SPD/$n/talker \
      --items $SPD/$n/talker_in/items.jsonl --out $Q/$n/quality.json 9>&- > $SPD/$n/quality.log 2>&1
    log "quality $n on card $c rc=$?"
  ) 9> $LOCK
done
wait
log "latency / gap jobs done: $(ls $O/latchunk_${NAME}*.json $O/gap_${NAME}*.json 2>/dev/null | wc -l) files"
