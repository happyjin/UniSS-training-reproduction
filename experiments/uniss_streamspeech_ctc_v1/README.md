# UniSS StreamSpeech-CTC v1

This experiment tree implements the StreamSpeech-style CTC alignment route for
UniSS without modifying the historical Phase1--3 or simultaneous-S2ST
experiments.

## Data decision

UniST is **not** converted to the original StreamSpeech CVSS-C/mHuBERT-km1000
layout.  The equivalent supervision already exists in the UniST Stage-A source
manifest:

- source speech: `source_audio`
- source text / ASR CTC target: `transcription`
- target text / NAR-S2TT CTC target: `translation`
- source and target languages: `src_lang`, `tgt_lang`
- UniSS synthesis targets: `target_bicodec`, `bicodec_global`

Stage01 therefore adds versioned SentencePiece models, four task/language CTC
targets, and CTC path-length audits as sidecars.  It never rewrites the source
manifest.

## Isolated stages

| Directory | Purpose |
| --- | --- |
| `stage00_audit/` | Input, offset-index, field and environment audit |
| `stage01_data/` | Parallel tokenizer corpus and CTC target sidecars |
| `stage02_ctc_probe/` | Frozen streaming-encoder CTC feasibility probe |
| `stage03_b2_discrete_bridge/` | Low-risk discrete GLM bridge |
| `stage04_ctc_policy/` | CTC-count READ/WRITE policy |
| `stage05_b1_continuous_bridge/` | Continuous hidden-to-Qwen bridge |
| `stage06_nar_semantic/` | NAR semantic generation head |
| `stage07_end_to_end_eval/` | Quality, latency and compute evaluation |

Large generated artifacts are written under:

```text
data/processed/uniss_streamspeech_ctc_v1/
checkpoints/uniss_streamspeech_ctc_v1/
runs/uniss_streamspeech_ctc_v1/
reports/uniss_streamspeech_ctc_v1/
```

Every stage is independently runnable and must pass its local smoke tests before
the next stage is launched.

