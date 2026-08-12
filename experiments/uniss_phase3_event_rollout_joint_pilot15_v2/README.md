# Phase3 event-rollout joint SFT: fixed15 v2 frontend repair

This run is a non-overwriting repair of
`uniss_phase3_event_rollout_joint_pilot15_v1`.

The v1 audit compared its real `iter_0000050` and `iter_0000100` exports and
proved that `frontend_projection.weight` remained exactly zero and every
`frontend_adapter` tensor remained bit-identical.  The training log independently
reported `frontend_residual_rms=0` at every observed iteration.  The cause was
the inherited Generalize13 trainable-parameter filter, which retained Qwen
LoRA and runtime heads but omitted the causal frontend.

V2 changes exactly one method variable: both
`true_subsecond_objective.frontend_adapter.*` and
`true_subsecond_objective.frontend_projection.*` are trainable and retain their
dedicated frontend learning-rate group.  Data, exact event grammar, losses,
65/35 trajectory/replay mixture, 18000-token packing, strict global shuffle,
Phase3 v4 initialization and 717-iteration coverage geometry remain unchanged.

All v1 checkpoints, logs, TensorBoard events, exports and reports are retained
as failed-mechanism evidence.  V2 writes unique artifacts and starts fresh from
Phase3 v4 `iter_0009075`.

Checkpoint selection is deliberately two-stage.  Teacher-forced validation
only creates a shortlist.  `evaluation/select_final_checkpoint.py` refuses a
winner until exact-runtime train and validation coverage, target-language
prefix-ASR useful audio, cached/fused parity, objective quality metrics and
paired Phase3 replay retention are all present and pass their explicit hard
gates.  Arbitrary PCM, forced WRITE, all-WAIT, semantic collapse and missing
metrics are rejection conditions.
