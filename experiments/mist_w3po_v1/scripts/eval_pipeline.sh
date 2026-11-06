#!/usr/bin/env bash
# One evaluation, end to end: online decode shards (one per entry of CARDS, a card may appear twice to stack two
# 62 GB copies), wait, merge + text score (score_run), then optionally speech (run_speech.sh) on SPEECH_CARDS.
#   NAME=<run> THINKER=<merged> CARDS="0 1 2 3 7 4 5 6" [CONTEXT=2] [POLICY=wait] [LAG_MS=2000] [EXTRA="--lora x"]
#   [SET=dev|test|ali] [SPEECH=1 SPEECH_CARDS="0 1 2 3 4 5 6 7"]
#   setsid nohup bash eval_pipeline.sh > logs/mist_w3po_v1/eval_<NAME>.log 2>&1 < /dev/null &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
export TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface TORCH_HOME=$J/cache/torch XDG_CACHE_HOME=$J/cache
export PYTHONPATH=.:third_party/Megatron-LM PATH=$J/venvs/vllm_omni_030/bin:$PATH
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$J/conda_envs/qwen3omni/bin/python
NAME=${NAME:?}; THINKER=${THINKER:?}; CARDS=${CARDS:?}; SET=${SET:-dev}
case $SET in
  dev) WIN=eval_outputs/mist_w3po_v1/data/windows_zh2en_v1/windows_dev_online.jsonl
       REF=data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_validation_ref.jsonl; SSET=zh2en ;;
  test) WIN=data/online_v6_20261031/windows_test_online_zh2en.jsonl
       REF=data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_ref.jsonl; SSET=zh2en ;;
  ali) WIN=data/online_v6_20261031/windows_test_online_ali.jsonl; REF=data/alimeeting_si_v1/windows_test_ref.jsonl; SSET=ali ;;
esac
D=eval_outputs/mist_w3po_v1/decode/$NAME
if [ -e "$D" ]; then echo "拒绝覆盖已有输出: $D" >&2; exit 1; fi
mkdir -p $D
log() { echo "[$(date -u +%FT%TZ)] $*"; }
set -- $CARDS; N=$#; k=0; pids=()
for c in $CARDS; do
  CUDA_VISIBLE_DEVICES=$c $PY -m experiments.mist_w3po_v1.evaluation.online_decode --thinker $THINKER --windows $WIN \
    --out $D/hyp_s$k.jsonl --policy ${POLICY:-wait} --lag-ms ${LAG_MS:-2000} --context ${CONTEXT:-2} --dir zh2en \
    --batch ${BATCH:-16} --max-gib 120 --shard $k --num-shards $N --shard-by meeting ${EXTRA:-} > $D/infer_s$k.log 2>&1 &
  pids+=($!); log "shard $k card $c pid $!"; k=$((k+1))
  [ $k -lt $N ] && sleep ${STAGGER:-45}
done
for p in "${pids[@]}"; do wait $p; done
nd=$(grep -l W3PO_ONLINE_DONE $D/infer_s*.log | wc -l)
[ "$nd" -eq "$N" ] || { log "DECODE_FAILED ($nd of $N shards done)"; exit 1; }
$PY -m experiments.mist_w3po_v1.evaluation.score_run --run $D --windows $WIN --ref-windows $REF --set $SSET > $D/score.log 2>&1 \
  || { log SCORE_FAILED; tail -3 $D/score.log; exit 1; }
log "TEXT_DONE $(tail -n 1 $D/score.log | cut -c1-400)"
if [ -n "${SPEECH:-}" ]; then
  RUN=$D WINDOWS=$WIN REF=$REF NAME=$NAME SET=$SSET CARDS="${SPEECH_CARDS:-$CARDS}" bash experiments/mist_w3po_v1/scripts/run_speech.sh
fi
log PIPELINE_DONE
