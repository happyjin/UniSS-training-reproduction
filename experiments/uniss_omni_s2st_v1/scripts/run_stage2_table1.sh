#!/usr/bin/env bash
# The full offline CVSS-T test evaluation for a Stage 2 checkpoint, in the
# four metrics of arXiv 2607.19810 Table 1: ASR-BLEU, Text-BLEU, A.PCP and
# SIM-O. Both directions, all 4,897 pairs each.
#
# Stage 1's run lived in a scratchpad and the report quoted it. This is the
# same flow as a committed file, so the reproduction path in the Stage 2
# report points at something that exists.
#
# Two differences from Stage 1's. Generation goes through stage2_generate,
# which runs the model end to end rather than cascading, and the checkpoint
# is a stage-2 export carrying the Thinker's LoRA alongside the Talker.
#
#   CKPT=.../stage2_iter0006000.pt OUT=.../table1_stage2 \
#     bash experiments/uniss_omni_s2st_v1/scripts/run_stage2_table1.sh
#
# LIMIT=24 for a smoke test; LIMIT=0 (the default) is the whole split.
# STAGE=gen or STAGE=metrics runs half of it; the default runs both.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
REPO_ROOT=$PWD

export PYTHONPATH=. PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/opt/dlami/nvme/jasonleeeli/cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

PY="${PY:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python}"
MODELS="${MODELS:-/opt/dlami/nvme/jasonleeeli/evaluation_models}"
TOKENIZED="${TOKENIZED:-$REPO_ROOT/data/cvss_t/tokenized/cvss_t_zh_en_v1}"
CKPT="${CKPT:?set CKPT to an export_stage2_from_megatron output}"
OUT="${OUT:?set OUT to an output directory}"
LIMIT="${LIMIT:-0}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
STAGE="${STAGE:-both}"

PAIRS=("cmn_eng cvss_t_zh_en_test cmn->eng" "eng_cmn cvss_t_en_zh_test eng->cmn")

if [ "$STAGE" = "both" ] || [ "$STAGE" = "gen" ]; then
  mkdir -p "$OUT"
  for pair in "${PAIRS[@]}"; do
    set -- $pair; NAME=$1; PQ=$2; DIR=$3
    # One directory per direction: the project's validator requires a
    # single direction per results file.
    rm -rf "$OUT/$NAME"; mkdir -p "$OUT/$NAME/logs"
    shards=0; for _ in $GPUS; do shards=$((shards + 1)); done
    index=0
    for gpu in $GPUS; do
      CUDA_VISIBLE_DEVICES=$gpu $PY -m experiments.uniss_omni_s2st_v1.evaluation.stage2_generate \
        --checkpoint "$CKPT" --dev-parquet "$TOKENIZED/$PQ.parquet" --limit "$LIMIT" \
        --shard $index --num-shards $shards --output "$OUT/$NAME" \
        > "$OUT/$NAME/logs/gen_$index.log" 2>&1 &
      index=$((index + 1))
    done
    wait
    cat "$OUT/$NAME"/stage2_*.jsonl > "$OUT/$NAME/results.jsonl"
    echo "$DIR : $(wc -l < "$OUT/$NAME/results.jsonl") 条"
  done
  echo "GEN DONE"
fi

if [ "$STAGE" = "both" ] || [ "$STAGE" = "metrics" ]; then
  for pair in "${PAIRS[@]}"; do
    set -- $pair; NAME=$1; PQ=$2; DIR=$3
    D="$OUT/$NAME"
    N=$(wc -l < "$D/results.jsonl")
    echo "===== $DIR  ($N 条) ====="
    rm -rf "$D/metrics"
    # ASR-BLEU, Text-BLEU and A.PCP, through the repo's own harness.
    EXPECTED_PAIRS=$N MODES=tts \
      bash experiments/uniss_omni_s2st_v1/scripts/run_stage1_metrics.sh "$D" "$DIR" 2>&1 | tail -6
    # SIM-O is separate: the repo's speaker_similarity scores against one
    # shared reference, and the paper's definition is per utterance against
    # its own paired reference.
    first_gpu=$(echo $GPUS | awk '{print $1}')
    CUDA_VISIBLE_DEVICES=$first_gpu $PY -m experiments.uniss_omni_s2st_v1.evaluation.sim_o \
      --input "$D/results.jsonl" --output "$D/metrics/sim_o.json" \
      --model "$MODELS/wavlm-base-plus-sv" --device cuda:0 2>&1 | tail -4
  done
  echo METRICS_DONE
fi
