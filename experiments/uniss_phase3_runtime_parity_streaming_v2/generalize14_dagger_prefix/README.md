# Generalize14 DAgger prefix canary

Generalize13 succeeds only under oracle teacher-forced history.  This isolated
Megatron experiment starts from its best held-out checkpoint (iteration 50),
keeps the same architecture and trainable scope, and changes the state
distribution used for optimization.

For trajectory microbatches, a no-gradient probe predicts runtime-constrained
Qwen text tokens and causal semantic microblocks.  A scheduled fraction is
shifted into the input prefix without changing sequence length or THD session
boundaries.  A second probe later in the schedule makes the state closer to an
on-policy rollout.  The differentiable pass retains oracle labels, so every
corrupted state receives DAgger-style oracle correction.  A dedicated recovery
CE is reported, action heads see hidden states downstream of model prefixes,
and grouped soft/hard deadline survival has non-zero weight.

```bash
bash experiments/uniss_phase3_runtime_parity_streaming_v2/generalize14_dagger_prefix/prepare_data.sh
bash experiments/uniss_phase3_runtime_parity_streaming_v2/generalize14_dagger_prefix/run_8gpu.sh
```

TensorBoard uses port `6087`.  Promotion still requires strict natural-WRITE
real-PCM success on both seen canary samples and disjoint held-out samples.
