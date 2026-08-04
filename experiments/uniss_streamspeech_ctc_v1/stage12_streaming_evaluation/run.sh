#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
PYTHON=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python
OUT=$ROOT/reports/uniss_streamspeech_ctc_v1/stage12_stage09_11_bilingual_eval_v1

"$PYTHON" -m experiments.uniss_streamspeech_ctc_v1.stage12_streaming_evaluation.evaluate \
  --inputs \
    "$ROOT/reports/uniss_streamspeech_ctc_v1/stage11_streaming_audio_smoke_eng_cmn_v2.json" \
    "$ROOT/reports/uniss_streamspeech_ctc_v1/stage11_streaming_audio_smoke_cmn_eng_v2.json" \
  --output-json "${OUT}.json" \
  --output-md "${OUT}.md"
