# Stage 7 — full-data GRPO bootstrap

This is the same distributed engineering bootstrap as v2, now sampling the
full schedule with a 65,536-record bounded shuffle buffer. It remains gated on
stable SFT output and is not presented as full Qwen-token GRPO.
