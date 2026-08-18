# UniSS Phase3-v4 E2E Simultaneous S2ST pilot15 v1

This directory is the isolated implementation of section 27 in
`docs/uniss_training_reproduction/uniss_phase3_v4_quality_first_true_streaming_asr_mt_tts_training_plan.md`.

The formal student is initialized from the complete Stage-A V1 compound
checkpoint (`iter_0000381`).  The causal WhisperVQ frontend, V1 bridge, CTC
diagnostic head, BiCodec, V1 teacher, and Phase3 teacher remain frozen.  One
shared low-learning-rate Qwen student predicts append-only source ASR deltas,
incremental target text deltas, and target semantic deltas.

The first implementation gate is deliberately CPU-only:

1. reuse the immutable 15-shard A4-A8 aligned manifests;
2. convert source word events and safe micro-WRITE spans into one unified
   source-audio/ASR/MT/semantic trajectory;
3. reject any prefix rollback, future leakage, semantic gap/overlap, incomplete
   source GLM coverage, or PCM-time mismatch;
4. preserve every historical experiment and write every generated asset under
   a new `DATA_RUN_ID`.

Validation commands:

```bash
experiments/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/scripts/run_cpu_tests.sh
DATA_RUN_ID=gold_smoke_v1 \
  experiments/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/scripts/build_gold_smoke.sh
```

The smoke builder hashes source audio and validates all generated trajectories.
Checkpoint identity is a SHA256 tree fingerprint over every distributed shard,
not merely the small (and sometimes identical) Megatron `metadata.json` file.
Formal V1 free-running rollout and teacher posterior generation are separate GPU
gates and must not be replaced with teacher-forced placeholders.
