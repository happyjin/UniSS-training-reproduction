#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
PYTHON=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python
STAGE="$ROOT/experiments/uniss_streamspeech_ctc_v1/stage02_ctc_probe"
LATENT="$ROOT/data/processed/simul_uniss_subsecond_v2/stage_a_v3_clone_15shard_v1/manifest.jsonl"
SOURCE="$ROOT/data/processed/simul_uniss_subsecond_v1/pilot_15shard/stage_a_source/stage_a_source_manifest.jsonl"
TOKENIZERS="$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage01_data/tokenizers"
OUTPUT="$ROOT/data/processed/uniss_streamspeech_ctc_v1/stage02_ctc_probe"
SOURCE_INDEX="$OUTPUT/source_id_index/source_offsets.sqlite"
WORKERS="${1:-16}"
EXPECTED=1325243

test -f "$TOKENIZERS/tokenizers.json"
mkdir -p "$OUTPUT/parts" "$OUTPUT/source_id_index/parts"

if [[ ! -f "$SOURCE_INDEX" ]]; then
  seq 0 $((WORKERS - 1)) | xargs -P "$WORKERS" -I '{}' \
    "$PYTHON" "$STAGE/extract_source_id_index.py" \
      --source-manifest "$SOURCE" \
      --source-offsets "${SOURCE}.offsets.bin" \
      --output-dir "$OUTPUT/source_id_index/parts" \
      --worker-index '{}' \
      --num-workers "$WORKERS"
  "$PYTHON" "$STAGE/assemble_source_id_index.py" \
    --parts-dir "$OUTPUT/source_id_index/parts" \
    --output "$SOURCE_INDEX" \
    --expected-records 1500000
fi

seq 0 $((WORKERS - 1)) | xargs -P "$WORKERS" -I '{}' \
  "$PYTHON" "$STAGE/build_probe_manifest.py" \
    --latent-manifest "$LATENT" \
    --latent-offsets "${LATENT}.offsets.bin" \
    --source-manifest "$SOURCE" \
    --source-id-index "$SOURCE_INDEX" \
    --tokenizer-dir "$TOKENIZERS" \
    --output-dir "$OUTPUT/parts" \
    --worker-index '{}' \
    --num-workers "$WORKERS"

"$PYTHON" "$STAGE/assemble_probe_manifest.py" \
  --parts-dir "$OUTPUT/parts" \
  --tokenizer-dir "$TOKENIZERS" \
  --latent-manifest "$LATENT" \
  --expected-input-records "$EXPECTED" \
  --output "$OUTPUT/dataset_index.json"
