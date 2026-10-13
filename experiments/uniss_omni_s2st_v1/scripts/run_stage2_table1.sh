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

# Never delete. A non-empty output directory is the user's to clear,
# not this script's -- an rm -rf here once destroyed a finished run's
# waveforms because two runs chose the same path.
refuse_if_present() {
  if [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; then
    echo "拒绝覆盖已有输出: $1" >&2
    echo "换一个输出路径，或由你手动清理后重跑。" >&2
    exit 1
  fi
}
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
# "translation" starts the Talker's text stream at the translation rather
# than at the transcript; a diagnostic, not how stage 2 trained.
TALKER_TEXT="${TALKER_TEXT:-full}"
# Optional id allowlist. One file may hold both directions: the ids do
# not overlap, so each run keeps only its own.
SUBSET="${SUBSET:-}"

PAIRS=("cmn_eng cvss_t_zh_en_test cmn->eng" "eng_cmn cvss_t_en_zh_test eng->cmn")

if [ "$STAGE" = "both" ] || [ "$STAGE" = "gen" ]; then
  mkdir -p "$OUT"
  for pair in "${PAIRS[@]}"; do
    set -- $pair; NAME=$1; PQ=$2; DIR=$3
    # One directory per direction: the project's validator requires a
    # single direction per results file.
    refuse_if_present "$OUT/$NAME"; mkdir -p "$OUT/$NAME/logs"
    subset_args=()
    [ -n "$SUBSET" ] && subset_args=(--subset "$SUBSET")
    shards=0; for _ in $GPUS; do shards=$((shards + 1)); done
    index=0
    for gpu in $GPUS; do
      CUDA_VISIBLE_DEVICES=$gpu $PY -m experiments.uniss_omni_s2st_v1.evaluation.stage2_generate \
        --checkpoint "$CKPT" --dev-parquet "$TOKENIZED/$PQ.parquet" --limit "$LIMIT" \
        --shard $index --num-shards $shards --output "$OUT/$NAME" \
        --talker-text "$TALKER_TEXT" "${subset_args[@]}" \
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
    refuse_if_present "$D/metrics"
    # ASR-BLEU, Text-BLEU and A.PCP, through the repo's own harness.
    EXPECTED_PAIRS=$N MODES=tts \
      bash experiments/uniss_omni_s2st_v1/scripts/run_stage1_metrics.sh "$D" "$DIR" 2>&1 | tail -6
    # Text-BLEU. The shared harness looks for a vllm generation file and
    # skips when there is none, which was right for Stage 1 -- a TTS task
    # translates nothing, so there was no generated translation to score.
    # Stage 2 produces one, and it is a Table 1 column, so it is scored
    # here from the same results file. text_metrics reads tgt_lang and
    # picks the zh tokenizer for Chinese targets on its own.
    $PY -m evaluation.text_metrics \
      --input "$D/results.jsonl" --output "$D/metrics/text_bleu.json" \
      --hypothesis-field translated_text --reference-field translation_ref 2>&1 | tail -2
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
