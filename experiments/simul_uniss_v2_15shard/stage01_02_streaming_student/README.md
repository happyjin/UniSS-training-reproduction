# Stage 1/2 — streaming token student and CTC heads

The current implementation trains the causal token student, teacher GLM CTC,
Source CTC, and Target CTC heads jointly, so Stage 1 and Stage 2 share this
folder and checkpoint.

`run_token_8gpu.sh` uses torchrun/DDP on all eight local H200 GPUs. Training
uses a distributed random sampler; validation uses a deterministic distributed
sampler and aggregates metrics across all ranks. Only rank 0 writes checkpoints
and TensorBoard events.

