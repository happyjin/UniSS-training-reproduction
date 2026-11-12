#!/usr/bin/env bash
# speech-only ASR + scoring of one finished X-Translator run (paper protocol), then the result json.
#   POINT=xt-default DIR=en2zh CARD=3 bash score_run.sh
set -euo pipefail
: "${POINT:?}" "${DIR:?}" "${CARD:?}"
R=/opt/dlami/nvme/jasonleeeli/projects/UniSS
cd "$R"
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
export HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface MODELSCOPE_CACHE=/opt/dlami/nvme/jasonleeeli/cache/modelscope
export PYTHONPATH=. OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
RD=$R/eval_outputs/baselines_xtranslator_v1/$POINT/realsi_$DIR
RES=$R/paper/baselines_results/xtranslator
mkdir -p "$RES"
$PYE experiments/baselines_xtranslator_v1/score_xt.py "$RD" > /dev/null   # writes results.jsonl (sorted, dup check)
$PYE reports/uniss_si_v4/speech_only.py "$RD"
CUDA_VISIBLE_DEVICES=$CARD $PYE -m evaluation.asr_transcribe --input "$RD/results_speechonly.jsonl" \
  --output "$RD/asr_speech.jsonl" --device cuda:0 --batch-size 4 --resume
$PYE experiments/baselines_xtranslator_v1/score_xt.py "$RD" --require-asr --out "$RES/${POINT}_${DIR}.json"
echo "SCORE_DONE $POINT $DIR"
