# Stage06: B1 continuous residual bridge

This stage tests the higher-ceiling continuous Qwen interface without discarding
the working B2 baseline. It freezes the Stage04 best causal encoder and hard GLM
bridge, then adds a zero-initialized `768 -> 896` residual on each pooled 80 ms
speech token:

```text
Qwen speech embedding = frozen B2 hard embedding + Linear(pooled hidden)
```

At step 0 the model is exactly equal to Stage04 B2. Only the residual projection
is optimized through frozen Phase3 target-token NLL. The untrained baseline is
saved as `initial.pt`; `best.pt` can never be replaced by a worse validation NLL.
This is deliberately isolated from the later NAR semantic/RTF experiment.

