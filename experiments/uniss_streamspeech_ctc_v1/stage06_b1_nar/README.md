# Stage06: B1 continuous residual bridge

This stage tests the higher-ceiling continuous Qwen interface without discarding
the working B2 baseline. It freezes the Stage04 best causal encoder and hard GLM
bridge, then adds a zero-initialized `768 -> 896` residual on each pooled 80 ms
speech token. The correction is bounded as `0.05 * tanh(Linear(hidden))` after
an unconstrained additive smoke exposed a first-update BF16 gradient overflow:

```text
Qwen speech embedding = frozen B2 hard embedding + 0.05*tanh(Linear(pooled hidden))
```

At step 0 the model is exactly equal to Stage04 B2. Only the residual projection
is optimized through frozen Phase3 target-token NLL. The untrained baseline is
saved as `initial.pt`; `best.pt` can never be replaced by a worse validation NLL.
This is deliberately isolated from the later NAR semantic/RTF experiment.
The trainer also rejects non-finite gradients before the optimizer can corrupt a
checkpoint.

## Formal training engine

`run_megatron_8gpu.sh` is the formal launcher. It uses the repository-pinned
Megatron-LM runtime for single-node eight-GPU data parallelism, Megatron gradient
accumulation/optimizer/checkpointing, micro-batch 1 and global batch 128. The old
`run_8gpu.sh` is retained only as an isolated DDP diagnostic and is not the
formal Stage06 training path.
