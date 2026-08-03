# Stage-C-after-v3 Bayesian safe-commit

This experiment is isolated from all historical Stage C checkpoints, logs,
TensorBoard runs and launchers. It freezes the selected latent Stage-B-v3
Student and trains only the explicit Bayesian safe-commit gate on the existing
formal 15-shard Micro-WRITE support labels.

The older Stage C expects a classifier-based Student with a 16,385-way GLM
head. Stage-B-v3 instead emits a 1,280-dimensional latent quantized through the
frozen 16,384-entry WhisperVQ codebook. This experiment therefore extracts
tail evidence directly from latent/codebook geometry:

1. nearest-codebook confidence and top-2 margin;
2. nonblank availability and confidence proxy;
3. Student stability probability;
4. source CTC confidence;
5. tail-token persistence;
6. target-capacity probability.

The formal launcher uses eight GPUs. Each DataLoader item decodes one source
audio and packs four independently sampled prefixes from it. A per-GPU record
batch of 512 therefore produces 2,048 prefix forwards per GPU and an effective
global prefix batch of 16,384. The 1,250 formal steps cover approximately 3.86
record epochs, matching the historical launcher's number of passes over source
records while exposing four independently sampled prefixes per decoded audio.

The frozen 121M Student and tail codebook search provide the useful GPU
workload; the Bayesian gate itself is only a small prior plus class-conditional
diagonal Gaussian likelihoods, so GPU power must be interpreted as
feature-extraction throughput rather than gate size.

Commands:

```bash
bash scripts/simul_uniss_subsecond_v3/train_stage_c_after_v3.sh smoke
bash scripts/simul_uniss_subsecond_v3/train_stage_c_after_v3.sh throughput
bash scripts/simul_uniss_subsecond_v3/run_stage_c_after_v3_pipeline.sh
```

TensorBoard uses port `6061`. Stage D is deliberately not started by this
pipeline; the calibrated Stage C quality gate must be inspected first.
