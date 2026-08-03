# Stage07: end-to-end evaluation

This stage evaluates the frozen Stage06 B1 residual without modifying or
converting the original Megatron distributed checkpoints.  The first gate is a
direction-balanced frozen-Phase3 text probe on exactly the same validation rows
used by the Stage04 B2 baseline.  Only checkpoints that pass that gate proceed
to streaming policy, audio quality, and wall-clock latency evaluation.

Generated artifacts are isolated under:

```text
reports/uniss_streamspeech_ctc_v1/stage07_*/
eval_outputs/uniss_streamspeech_ctc_v1/stage07_*/
```

`checkpoint_io.py` loads only the three B1 residual tensors from a Megatron
`torch_dist` checkpoint.  Frozen Phase3 and Stage04 B2 weights continue to come
from their immutable source checkpoints, avoiding a duplicate multi-gigabyte
export and making the evaluated provenance explicit.
