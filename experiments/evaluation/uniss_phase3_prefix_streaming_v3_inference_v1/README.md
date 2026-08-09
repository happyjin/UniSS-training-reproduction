# Full198 Phase3 prefix-streaming v3 inference

This directory is independent of all historical training and evaluation paths.
It selects only physically saved checkpoints, exports only the trained LoRA
tensors, and never mutates the Phase3 HF base or Megatron checkpoint.

The runtime is explicitly **source-side prefix/pseudo-streaming**: audio is
revealed in 320/480/640 ms increments and the WhisperVQ/GLM frontend is
cumulatively re-encoded.  It is not a causal Whisper encoder.

Checkpoint selection uses an equal rank-sum across prefix CE, streaming TTS
semantic CE, commit CE, teacher KL, adjacent-prefix consistency and WAIT/WRITE
action CE.  Among saved 500-iteration checkpoints, `iter_0008000` is best.

