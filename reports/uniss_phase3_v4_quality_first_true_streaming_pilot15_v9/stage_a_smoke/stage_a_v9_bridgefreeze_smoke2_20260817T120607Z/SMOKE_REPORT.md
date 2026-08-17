# Stage A V9 bridge-freeze Megatron smoke report

## Outcome

PASS. The isolated V9 entrypoint completed two real Megatron updates on eight
H200 GPUs, ran validation, and saved the iteration-2 distributed checkpoint.
This smoke authorizes the 255-update V9 canary only. It does not authorize the
381-update formal run or Stage B.

## Run identity

- Run ID: `stage_a_v9_bridgefreeze_smoke2_20260817T120607Z`
- Code commit: `e107f43`
- Initialization: immutable Phase3 checkpoint
  `checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4`
- Framework: Megatron, data parallel size 8
- Sequence length: 18,000
- Micro/global batch: 1 / 128
- Updates: 2
- Prefix samples: 256, exact global shuffle seed `20260816`
- Bridge learning rate: `5e-5`
- Log: `logs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v9/stage_a_smoke/stage_a_v9_bridgefreeze_smoke2_20260817T120607Z/train.log`

## Verification

- Both updates completed with zero skipped iterations and zero NaN iterations.
- Validation completed at updates 1 and 2, plus final validation after
  checkpoint save.
- The iteration-2 distributed checkpoint contains all eight rank shards and
  `latest_checkpointed_iteration.txt`.
- Final validation diagnostics were finite:
  - CTC blank ratio: `0.000000`
  - CTC blank posterior: `0.000544`
  - teacher-code cosine: `0.956141`
  - causal GLM agreement: `0.154319`
  - code-adapter RMS: `0.000550`
- Targeted V9 tests: `6 passed`.
- Smoke, canary, and formal dry-runs all resolve to the isolated V9 entrypoint
  and preserve bridge LR `5e-5`.

## Non-blocking warnings

- Installed flash-attn is `2.8.3.post1`, while the local compatibility warning
  lists support through `2.8.1`. The fused attention path nevertheless loaded
  and both training updates completed successfully.
- Megatron emitted its existing process-group cleanup warning at normal exit.

## Next gate

Run the isolated 255-update bridge-freeze canary from the immutable Phase3
checkpoint. Its final validation must satisfy all strict gates:

- CTC blank ratio `<= 0.25`
- CTC blank posterior `<= 0.25`
- teacher-code cosine `>= 0.85`
- causal GLM agreement `>= 0.02`
- code-adapter RMS `<= 0.50`
- zero skipped and zero NaN iterations

Only a passing canary may authorize a fresh V9 formal run. Stage B remains
blocked until the complete formal Stage A subsequently passes.
