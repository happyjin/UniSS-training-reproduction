# Stage01: parallel CTC data preparation

Stage01 keeps the original UniST data immutable.  It trains one English and one
Chinese SentencePiece model, then creates four task/language-conditioned target
streams:

- `asr_eng`, `asr_cmn`
- `nar_s2tt_eng`, `nar_s2tt_cmn`

The two language tokenizers are shared across source/target roles, while the
training heads remain separate.  Every record is audited at both 25 Hz (the
planned StreamSpeech-style frontend) and 12.5 Hz (the current frozen
WhisperVQ/GLM probe).

Run the complete CPU pipeline with 16 workers:

```bash
bash experiments/uniss_streamspeech_ctc_v1/stage01_data/run_parallel.sh 16
```

Generated output is written only to
`data/processed/uniss_streamspeech_ctc_v1/stage01_data/`.

