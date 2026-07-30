# Simul-UniSS Subsecond v1 — Stage A

This directory documents the isolated Stage-A source/frontend preparation used
by Stage B. Historical Simul-UniSS and Phase1–3 directories are read-only.

Stage A currently produces the source-side supervision required to train the
causal audio student:

- BiCodec-reconstructed 16 kHz source FLAC;
- released UniST `source_glm` teacher tokens;
- fixed-rate GLM end times;
- source/target language, text, speaker tokens, and target semantic tokens;
- byte-offset indexed JSONL manifests;
- per-shard and assembled resumability markers.

The marker is deliberately named `STAGE_A_SOURCE_COMPLETE.json`. It does not
claim that the bilingual support alignment required by Stage C/D is complete.

## Smoke

```bash
scripts/simul_uniss_subsecond_v1/prepare_stage_a_pilot.sh --smoke
```

Expected output:

```text
data/processed/simul_uniss_subsecond_v1/smoke/stage_a_source/
```

## Formal 15-shard source preparation

```bash
scripts/simul_uniss_subsecond_v1/prepare_stage_a_pilot.sh --formal
```

Expected output:

```text
data/processed/simul_uniss_subsecond_v1/pilot_15shard/stage_a_source/
```

Each parquet shard is independently resumable. The launcher processes up to
eight shards in parallel and assembles the manifest only after all requested
part markers are valid.
