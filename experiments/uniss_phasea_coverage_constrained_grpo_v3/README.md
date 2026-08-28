# Phase-A coverage-constrained long-episode GRPO v3

This experiment is isolated from all historical UniSS scripts, data, reports,
checkpoints, and listening audio.  It starts from immutable Phase-A
`iter_0000381` and tests only train-seen behavior on the already-built
15-shard pilot and long-episode pool.

This revision addresses the v2 failure mode: it learned earlier and more
frequent WRITE actions, but translated only about 35% of the frozen teacher
target on average.  It therefore makes teacher-target coverage an absolute
objective rather than retaining the weak historical baseline.

The fixed execution protocol is:

1. stream-audit all 15 existing trajectory-cache parts without regenerating
   source audio or teacher data;
2. retain a balanced deterministic WAIT/WRITE sample plus every event belonging
   to the frozen 64-episode protocol;
3. generate a post-update 64x4 baseline from the final v2 checkpoint;
4. annotate every event with monotonic teacher-target coverage, spoken target
   coverage, empty WRITE, and target-language purity;
5. generate group-four fresh rollouts on the same 64 bidirectional episodes,
   exactly matching the historical Stateful Long-Episode RL comparison geometry;
6. update once, regenerate rollouts, and repeat for three rounds total;
7. generate one final post-training 64x4 rollout used only for evaluation;
8. compare all arms using content coverage before latency.

`FLUSH` is deterministic at true source EOS.  It is not invented as a third
token because the immutable UniSS vocabulary and the audited cache supervise
only `TOKEN_WAIT_READ` and `TOKEN_WRITE_GENERATE`.

Three fresh rounds are not three epochs over stale data: every round produces
new actions, text, semantic audio, old log probabilities, local rewards, and
advantages, and its packed trajectories are consumed exactly once.

The v3 reward does not award latency until a candidate reaches at least 75%
teacher-target coverage and retains ASR/MT quality.  At source EOS, coverage
below 80%, pending TTS items, target-language leakage, repetition, and TTS
failures are explicitly penalized.  The Phase-A encoder, ASR route, and TTS
decoder stay frozen; only the existing top-layer policy/MT/TTS LoRA route is
optimized, with stronger Phase-3 replay (0.50) to preserve quality.
