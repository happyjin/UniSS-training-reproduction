# UniSS Phase3 dense-aligned streaming pilot15 v1

This directory is an isolated experiment. It does not modify the historical
Phase1/Phase2/Phase3, wait-k, StreamSpeech/CTC, or Gradio implementations.

The source set is the accepted formal A4--A8 split derived from fixed UniST
shards 0--14:

- train sessions: 1,325,243;
- validation sessions: 13,469;
- Phase3 initialization: native Megatron iteration 9075;
- policy tick: 160ms;
- training: 8 H200 GPUs, sequence length 18,000, MBS 2, GBS 128;
- coverage: three complete trajectory epochs.

The shuffle unit is a complete dense session/pack. Each coverage epoch gets an
independent restart-stable global permutation. The ordered events inside a
session are data-contract protected and are never shuffled.

Run the real-data smoke:

```bash
bash experiments/uniss_phase3_dense_aligned_streaming_pilot15_v1/scripts/run_data_smoke.sh
```

Build all dense train/validation sessions with CPU parallelism:

```bash
bash experiments/uniss_phase3_dense_aligned_streaming_pilot15_v1/scripts/run_data_full_cpu.sh
```

Generated artifacts live under
`data/processed/uniss_phase3_dense_aligned_streaming_pilot15_v1`; no historical
data or result path is overwritten.
