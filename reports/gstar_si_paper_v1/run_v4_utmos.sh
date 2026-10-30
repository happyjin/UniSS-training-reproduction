#!/usr/bin/env bash
# Paper v4 speech: UTMOS for every system of gstar_v4_speech_20261030 whose utmos/ has no per_sample_utmos.jsonl.
# The speech jobs' UTMOS step could not fetch the SpeechMOS hub repo (no TORCH_HOME, SSL); the cached copy is in
# /opt/dlami/nvme/jasonleeeli/cache/torch. Card given as $1 (a 4-7 card; the model needs ~2 GB beside other work).
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
CARD="${1:?card}"
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
for out in eval_outputs/uniss_qwen3_omni_v1/gstar_v4_speech_20261030/*/; do
  out=${out%/}
  [ -f ${out}/talker/results_speechonly.jsonl ] || continue
  [ -f ${out}/utmos/per_sample_utmos.jsonl ] && continue
  echo "$(date +%T) utmos ${out}"
  TORCH_HOME=/opt/dlami/nvme/jasonleeeli/cache/torch CUDA_VISIBLE_DEVICES=${CARD} PYTHONPATH=. ${PYE} -m evaluation.utmos_metrics \
    --input ${out}/talker/results_speechonly.jsonl --output-dir ${out}/utmos --device cuda:0 > ${out}/utmos.v4fix.log 2>&1 \
    || echo "$(date +%T) failed ${out}"
done
echo "UTMOS_DONE"
