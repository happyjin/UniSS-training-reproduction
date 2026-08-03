#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
PYTHON=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python
STAGE="$ROOT/experiments/uniss_streamspeech_ctc_v1/stage01_data"
SOURCE="$ROOT/data/processed/simul_uniss_subsecond_v1/pilot_15shard/stage_a_source/stage_a_source_manifest.jsonl"
OFFSETS="${SOURCE}.offsets.bin"
OUTPUT="$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage01_data"
WORKERS="${1:-16}"

mkdir -p "$OUTPUT/tokenizer_corpus" "$OUTPUT/tokenizers" "$OUTPUT/sidecar_parts" "$OUTPUT/reports"

seq 0 $((WORKERS - 1)) | xargs -P "$WORKERS" -I '{}' \
  "$PYTHON" "$STAGE/extract_tokenizer_corpus.py" \
    --manifest "$SOURCE" \
    --offsets "$OFFSETS" \
    --output-dir "$OUTPUT/tokenizer_corpus" \
    --worker-index '{}' \
    --num-workers "$WORKERS"

"$PYTHON" "$STAGE/train_tokenizers.py" \
  --corpus-dir "$OUTPUT/tokenizer_corpus" \
  --output-dir "$OUTPUT/tokenizers" \
  --num-threads "$WORKERS"

seq 0 $((WORKERS - 1)) | xargs -P "$WORKERS" -I '{}' \
  "$PYTHON" "$STAGE/build_ctc_sidecar.py" \
    --manifest "$SOURCE" \
    --offsets "$OFFSETS" \
    --tokenizer-dir "$OUTPUT/tokenizers" \
    --output-dir "$OUTPUT/sidecar_parts" \
    --worker-index '{}' \
    --num-workers "$WORKERS"

"$PYTHON" "$STAGE/assemble_sidecar.py" \
  --parts-dir "$OUTPUT/sidecar_parts" \
  --tokenizer-dir "$OUTPUT/tokenizers" \
  --source-manifest "$SOURCE" \
  --output "$OUTPUT/dataset_index.json"

"$PYTHON" "$STAGE/validate_ctc_sidecar.py" \
  --dataset-index "$OUTPUT/dataset_index.json" \
  --output "$OUTPUT/reports/stage01_validation.json"

