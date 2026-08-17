# UniSS Phase3 v4 quality-first true-streaming pilot15 v5

V5 repairs the v4 parameter-routing failure without modifying older runs.  It
starts from immutable Phase3 iteration 9075, freezes the entire released
WhisperVQ frontend, and inserts a zero-initialized low-rank residual adapter
between causal pooling and nearest-code quantization.

The adapter is `1280 -> 128 -> 1280`.  Its output projection is initialized to
zero, so the initial v5 code path is exactly the original Phase3/WhisperVQ code
path.  Only adapter parameters can alter pooled code geometry.  V3 anti-blank
CTC terms, v4 full-codebook identity CE, Phase3 replay, same-prefix teacher KL,
and the early short-chunk curriculum remain active.

Formal Stage A is blocked unless the final iteration-127 validation is at
160 ms and passes identity, CTC, geometry, stability, and checkpoint gates.
Stage B is never authorized by the canary gate.

