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
`--dist-ckpt-strictness raise_unexpected`.  This is the strict finetune mode:
runtime-requested model keys must exist in V1, while checkpoint-only optimizer
and RNG state is deliberately ignored because the run also requires
`--no-load-optim --no-load-rng`.  A separate metadata audit still requires an
exact match for every compound model key before loading.  Formal training
remains blocked until an external gate explicitly sets
`formal_training_authorized=true`.

The background formal sequence is intentionally split into gated handoffs:

1. `run_v1_rollout_formal_sequence.sh` builds train/valid V1 rollouts and the original structural audit;
2. `stratify_rollouts.py` revalidates every gold/rollout pair in parallel and writes an immutable indexed `clean`, `noisy_content`, or `quarantine` manifest plus `QUALITY_GATE.json`;
3. `wait_for_v1_then_run_phase3_teacher_cache.sh` starts the four formal teacher caches only after both strict quality gates pass and GPUs are free;
4. `wait_for_teacher_caches_then_build_task_pools.sh` starts 64-worker CPU construction of immutable train/valid 18k task pools only after both rollout quality gates and all four cache audits pass;
5. formal Megatron training remains blocked until the later GPU smoke, all-family canary and free-running validation authorize its gate.

The post-task-pool smoke/canary handoff is isolated from the rollout and cache
builders.  It waits for both immutable task-pool reports and all four teacher
audits, validates every active family denominator, obtains an exclusive GPU
lock, and waits rather than terminating any existing GPU process.  It then
runs one two-update 8-GPU structural smoke followed by one 8-GPU update for
each of the five task families:

```bash
DATA_RUN_ID=formal_gold_20260818T090515Z \
TASK_POOL_RUN_ID=task_pool_formal_20260818T201500Z \
TEACHER_FORMAL_RUN_ID=teacher_cache_formal_20260818T175859Z \
CANARY_RUN_ID=post_task_pool_canary_formal_20260819T000000Z \
  experiments/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/scripts/launch_post_task_pool_canaries_tmux.sh
```

The launcher defaults to `MBS=2`, `GBS=128`, BF16 and the native Megatron
entrypoint.  It refuses to overwrite any report, checkpoint, TensorBoard or log
directory.  After all six runs it selectively reloads every frozen
`stage_a_objective.*` tensor and requires an exact per-tensor bitwise match with
V1.  A passed `CANARY_REPORT.json` still records
`formal_training_authorized=false`; free-running E-ASR/E-MT/E-S2S validation
remains mandatory before a separate gate may authorize formal training.

If the two-update structural checkpoint passes implementation checks but fails
free-running generation because it has not learned the mixed objective, use an
isolated learning canary.  It must restart from immutable V1, retain the formal
GBS=128/MBS=2 objective and complete teacher caches, use 8 GPUs with
`num_workers=0` on this host, and stop after 10--100 updates.  It cannot bypass
formal authorization and cannot be resumed into formal training:

```bash
LEARNING_RUN_ID=learning_canary_10u_YYYYMMDDTHHMMSSZ \
LEARNING_ITERS=10 \
  experiments/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/scripts/launch_learning_canary_tmux.sh
```

The final checkpoint is bitwise-audited against frozen Stage A and must then be
exported and evaluated with the same frozen 16-record free-running gate.  A
formal 3-coverage-epoch run, if authorized, still restarts from immutable V1;
it must never continue from a learning-canary optimizer state.

The quality policy deliberately does not delete hard content examples.  A
structurally valid rollout with high WER/CER is retained as `noisy_content` so
the student learns robustness to realistic V1 prefixes.  A sample enters
`quarantine` only for malformed WRITE, early EOS, or missing final EOS.  Such a
sample is excluded from streaming-ASR, V1-history MT, and interleaved E2E
supervision, but its gold-history incremental MT and both Phase3 replay tasks
remain available.  Each task-pool report records per-stratum task counts and
supervised-token counts so this filtering cannot be silent.
