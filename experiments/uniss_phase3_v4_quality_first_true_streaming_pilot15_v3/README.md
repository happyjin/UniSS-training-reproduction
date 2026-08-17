# UniSS Phase3 v4 quality-first true-streaming pilot15 v3

This experiment is an isolated repair of the v2 Stage A formal run.  It does
not overwrite v1/v2 code, checkpoints, logs, TensorBoard events, or reports.

The v2 formal run reached an exact validation CTC blank ratio of `1.0` at
iteration 100.  V3 starts again from the immutable Phase3 iteration-9075
checkpoint and adds three explicit objective terms:

1. `ctc_monotonic_seed`: a weak, linearly decayed uniform monotonic
   byte-to-frame pseudo alignment used only during the first 40 percent;
2. `ctc_blank_budget`: a differentiable per-utterance bound on blank posterior
   mass, with weight 20;
3. `codebook_commitment`: MSE to the released source-GLM codebook vector.

The fresh CTC head blank bias is initialized to `-2.0`.  Whisper top layers
remain frozen until 30 percent progress; bottom layers and convolution remain
frozen until 60 percent.  Qwen follows the already validated five-percent
unfreeze rule.

## Required order

1. Run `scripts/run_stage_a_canary_8gpu.sh`.  It uses the real 18k packs,
   global batch 128, two acoustics per pack, one exact globally shuffled
   coverage epoch, and 127 updates.
2. Run `stage_a_causal_whisper_asr/check_canary.py` on its log.  Formal training
   is blocked unless validation reaches at least iteration 96, CTC greedy blank
   ratio is at most 0.95, posterior blank mass respects its dynamic budget,
   causal GLM agreement is at least 0.02, and there are no skipped/NaN steps.
3. Provide the passing JSON as `CANARY_AUTHORIZATION` and run
   `scripts/run_stage_a_formal_8gpu.sh` for the three-coverage, 381-step formal
   replacement.
4. The existing exact 334-row v2 checkpoint/runtime/final gate remains the
   final authority.  A canary pass authorizes only formal Stage A training; it
   never authorizes Stage B.

