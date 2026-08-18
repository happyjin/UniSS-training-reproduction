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

The native Megatron training layer is implemented in
`training/pretrain_e2e_megatron.py`.  It reconstructs the exact V1 compound
module namespace, strictly loads `iter_0000381`, freezes every
`stage_a_objective.*` parameter, and places only the native Qwen parameters in
the optimizer.  The formal launcher is `scripts/run_e2e_megatron.sh`; it always
uses `--finetune --no-load-optim --no-load-rng` and
`--dist-ckpt-strictness raise_all`.  Formal training remains blocked until an
external gate explicitly sets `formal_training_authorized=true`.

The background formal sequence is intentionally split into gated handoffs:

1. `run_v1_rollout_formal_sequence.sh` builds and audits train/valid V1 rollouts;
2. `wait_for_v1_then_run_phase3_teacher_cache.sh` starts the four formal teacher caches only after both rollout audits pass and GPUs are free;
3. `wait_for_teacher_caches_then_build_task_pools.sh` starts 64-worker CPU construction of immutable train/valid 18k task pools only after all four cache audits pass;
4. formal Megatron training remains blocked until the later GPU smoke, all-family canary and free-running validation authorize its gate.
