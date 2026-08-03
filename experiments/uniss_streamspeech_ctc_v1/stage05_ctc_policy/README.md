# Stage05: CTC-count READ/WRITE policy

This isolated policy implements the StreamSpeech decision rule without changing
the historical pseudo-schedule or action-token controller.

Inputs per chunk:

- source-ASR CTC frame path;
- target NAR-S2TT CTC frame path;
- source and target languages.

Safety constraints added for UniSS:

1. a token position must agree across consecutive chunk observations;
2. already committed tokens are immutable;
3. WRITE is considered only when the stable source count increases;
4. target count must satisfy the configurable `lagging_k` margin;
5. English output stops before the currently unfinished SentencePiece word;
6. finalization may flush the remaining stable target path.

The module reports rollback/conflict diagnostics but never rewrites committed
output.  It is trained by no separate loss and can be attached to any checkpoint
that exposes the two CTC heads.

