# Phase-A route-aligned constrained GRPO v1

This isolated experiment starts from immutable Phase-A `iter_0000381` and
tests only whether the method can improve the small train-seen protocol.  It
does not make a generalization claim and does not overwrite prior checkpoints,
scripts, reports, or listening audio.

Execution order:

1. 64-update joint SFT warm-up on the audited 15-shard interleaved E2E pool;
2. route the top-eight-layer LoRA through ASR, MT, TTS, and control prompts;
3. produce group-eight free-running rollouts on the eight frozen train demos;
4. train three epochs of constrained GRPO with Phase3 replay, KL, and anchor;
5. compare Phase A, warm-up, and all three RL epochs on the same eight demos;
6. generate continuous, timeline, and left-source/right-translation WAVs.

Latency reward is conditional on ASR, MT, completeness, and spoken-coverage
retention.  A candidate cannot win merely by waiting less, emitting less text,
or ending early.

