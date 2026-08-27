# Phase-A event-constrained long-episode GRPO v2

This experiment is isolated from all historical UniSS scripts, data, reports,
checkpoints, and listening audio.  It starts from immutable Phase-A
`iter_0000381` and tests only train-seen behavior on the already-built
15-shard pilot and long-episode pool.

The fixed execution protocol is:

1. stream-audit all 15 existing trajectory-cache parts without regenerating
   source audio or teacher data;
2. retain a balanced deterministic WAIT/WRITE sample plus every event belonging
   to the frozen 64-episode protocol;
3. warm up the existing WAIT/WRITE action-token policy for one shuffled pass;
4. generate group-four fresh rollouts on the same 64 bidirectional episodes,
   exactly matching the historical Stateful Long-Episode RL comparison geometry;
5. update once, regenerate rollouts, and repeat for three rounds total;
6. select the best round rather than assuming the final round is best;
7. compare Phase A, the previous epoch-2 policy, warm-up, and all fresh rounds.

`FLUSH` is deterministic at true source EOS.  It is not invented as a third
token because the immutable UniSS vocabulary and the audited cache supervise
only `TOKEN_WAIT_READ` and `TOKEN_WRITE_GENERATE`.

Three fresh rounds are not three epochs over stale data: every round produces
new actions, text, semantic audio, old log probabilities, local rewards, and
advantages, and its packed trajectories are consumed exactly once.
