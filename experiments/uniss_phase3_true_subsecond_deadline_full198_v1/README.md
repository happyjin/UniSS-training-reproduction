# UniSS Phase3 true-subsecond full198 v1

This directory is an isolated implementation of
`docs/uniss_training_reproduction/uniss_phase3_true_subsecond_deadline_streaming_full198_implementation_plan.md`.

It never writes into historical Phase1/2/3, Student, StreamSpeech, GRPO, or
Gradio experiment paths. The formal run is initialized from the exported
Phase3 v4 checkpoint and is orchestrated by the repository's Megatron runtime.

Formal invariants:

- all 198 UniST training shards are indexed;
- every accepted row materializes Quality replay, Performance replay, early
  trajectory, and middle/late trajectory tasks;
- final train iterations are `ceil(packed_count / 128)`;
- 8 GPUs, micro batch 2, global batch 128, sequence length 18000;
- no core CTC objective;
- launchers refuse to overwrite an existing run unless `RESUME=1` and a
  checkpoint tracker exists.

All generated data, caches, logs, runs, and checkpoints remain under
`/opt/dlami/nvme/jasonleeeli`.
