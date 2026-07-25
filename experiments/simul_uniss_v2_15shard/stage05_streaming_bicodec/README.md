# Stage 5 — streaming BiCodec

- `run_overlap_baseline.sh` performs the no-training overlap/cross-fade replay.
- `run_refinement_8gpu.sh` trains the isolated chunk-boundary refinement with
  eight-GPU DDP, distributed random sampling, fixed validation, and rank-0-only
  checkpoint/TensorBoard writes.

The base BiCodec checkpoint is read-only. The refinement checkpoint contains
only the separately trained prenet and decoder states.

