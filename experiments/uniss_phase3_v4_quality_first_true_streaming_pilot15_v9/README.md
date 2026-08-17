# UniSS Phase3 v4 quality-first true-streaming pilot15 v9

V9 is the minimal follow-up to the V8 255-update canary. V8 materially reduced
the V7 all-blank collapse, but missed the final gates by narrow margins:
blank ratio `0.3185 > 0.25` and teacher cosine `0.8475 < 0.85`.

V9 preserves the complete V8 objective, Phase3 initialization, data, exact
global shuffle, batch geometry, 18,000-token sequence length, curriculum,
optimizer horizon, and all replay/teacher losses. It changes only the two
failed mechanisms:

1. Increase the differentiable blank decision-margin contribution from
   `0.05` to `0.20`; the already-passing posterior target and persistent seed
   remain unchanged.
2. Freeze the bridge/adapter parameter group when optimizer progress reaches
   the end of the 127-update curriculum. New heads and Qwen groups continue at
   their existing LR floor, but the useful code adapter can no longer drift
   during the 128-update hold.

Every run starts from immutable Phase3. V8/V7 checkpoints are diagnostic only
and must never be resumed. The first authorization run is the same 255-update
shuffled-prefix canary. It never authorizes Stage B; it can authorize only a
new V9 formal run.

After the canary passes, `scripts/run_stage_a_formal_8gpu.sh` runs all 48,768
scheduled samples in 381 updates from immutable Phase3. The isolated
`stage_a_causal_whisper_asr/check_formal.py` gate is the only V9 artifact that
may authorize Stage B.

## Execution order

```bash
bash scripts/run_stage_a_smoke_8gpu.sh
bash scripts/run_stage_a_bridge_freeze_canary_8gpu.sh
python stage_a_causal_whisper_asr/check_canary.py \
  --log <canary-train.log> --output <new-canary-gate.json>
CANARY_AUTHORIZATION=<new-canary-gate.json> \
  bash scripts/run_stage_a_formal_8gpu.sh
```

The canary must finish all 255 updates, consume exactly 32,640 globally
shuffled samples, save the final checkpoint, and pass final validation at the
160-ms curriculum point. The strict quality gates remain: blank argmax ratio
`<= 0.25`, blank posterior `<= 0.25`, teacher cosine `>= 0.85`, causal-code
agreement `>= 0.02`, adapter RMS `<= 0.50`, zero NaN updates, and zero skipped
updates. Formal completion is a separate prerequisite for Stage B.
