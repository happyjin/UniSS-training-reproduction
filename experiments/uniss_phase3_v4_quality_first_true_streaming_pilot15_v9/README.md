# UniSS Phase3 v4 quality-first true-streaming pilot15 v9

V9 is the minimal follow-up to the close V8 long-hold failure. It starts from
the immutable Phase3 checkpoint and never resumes V7/V8 checkpoints.

V8 reduced final blank ratio from V7's 0.9986 to 0.3185, but missed the 0.25
gate. It also missed teacher cosine by only 0.0025. V9 therefore preserves the
successful V8 posterior target and persistent seed while changing only:

- allowed blank fraction: 0.20 -> 0.15;
- decision-margin scale: 0.05 -> 0.20;
- codebook commitment weight: 0.30 -> 0.40;
- adapter residual weight: 0.05 -> 0.10;
- bridge/adapter learning rate: 5e-5 -> 2e-5.

All data, exact global shuffle, 255-update canary geometry, Phase3 replay,
teacher KL, identity CE, CTC/Qwen learning rates, frozen Whisper frontend,
18000 sequence length, GBS 128, and optimizer/curriculum horizons remain
unchanged. Formal and Stage B remain blocked until the final 160-ms canary
gate passes.
