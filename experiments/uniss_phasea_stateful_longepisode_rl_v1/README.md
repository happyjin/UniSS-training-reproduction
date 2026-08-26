# UniSS Phase-A Stateful Long-Episode RL v1

This directory is an isolated successor to the historical bounded-window
long-form evaluator and Stage-A GRPO experiment.  It does not overwrite any
historical scripts, checkpoints, manifests, reports, or generated audio.

The execution order is:

1. implement and test an append-only stateful runtime;
2. evaluate the immutable Phase-A `iter_0000381` checkpoint on the four fixed
   long recordings;
3. isolate runtime, ASR, MT, and TTS failure sources;
4. train a conservative long-episode quality-gated RL adapter;
5. compare Phase-A runtime v1/v2, historical A3 runtime v2, and the new RL
   checkpoint with one frozen protocol.

Quality gates select and annotate checkpoints.  They do not stop the remaining
evaluation and report-generation stages.

