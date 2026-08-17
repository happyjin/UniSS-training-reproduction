# Stage A V8 Megatron smoke pass

- Run: `stage_a_v8_smoke2_20260817T100000Z`
- Decision: **PASS**
- Initialization: immutable Phase3 checkpoint
- GPUs: 8
- Megatron updates: 2/2
- Global batch size: 128
- Sequence length: 18,000
- NaN iterations: 0
- Skipped iterations: 0
- Final checkpoint: saved at iteration 2
- Final validation: completed at iteration 2

The smoke validates the complete executable path rather than only importing
the V8 modules: exact Phase3 distributed checkpoint load, V8 objective
construction, forward/backward, optimizer parameter groups, validation, and
distributed checkpoint save all completed successfully.

Final validation diagnostics:

| Metric | Value |
|---|---:|
| AR-ASR | 9.050784 |
| Source CTC | 17.36149 |
| CTC blank ratio | 0.000000 |
| CTC blank posterior | 0.000544 |
| V8 blank-posterior target | 0.200195 |
| Persistent CTC seed strength | 0.100098 |
| Causal GLM agreement | 0.154319 |
| Teacher-code cosine | 0.956141 |
| Adapter RMS | 0.000550 |

The loss magnitudes are initialization/smoke values, not a quality result.
The important evidence is that the intended V8 target and persistent seed are
active, every metric is finite, blank/code geometry is healthy at startup,
and the complete 8-GPU Megatron path succeeds. This smoke does not authorize
formal training or Stage B; the 255-update long-hold canary remains mandatory.
