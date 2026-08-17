# UniSS Phase3 v4 quality-first true-streaming pilot15 v4

V4 is an isolated repair of the v3 final-160-ms identity failure.  It starts
again from immutable Phase3 iteration 9075 and does not overwrite v1-v3 code,
logs, checkpoints, TensorBoard events, or reports.

V3 eliminated the CTC all-blank basin, but final validation exact source-GLM
agreement fell to `0.009229417`.  Its MSE codebook commitment kept teacher
cosine at `0.892115` but did not make the teacher code the nearest discrete
entry.

V4 therefore keeps every v3 anti-blank term and adds:

1. `codebook_identity_ce`: full 16,384-way cosine classification against the
   immutable released WhisperVQ codebook, temperature `0.07`, weight `0.30`;
2. `teacher_code_margin`: the diagnostic cosine margin between the teacher
   code and the strongest non-teacher code;
3. earlier repeated 160-ms exposure from 65 percent progress onward;
4. Whisper top-layer adaptation from 10 percent progress, while Whisper
   bottom layers and convolution stay frozen to preserve acoustic geometry.

The canary gate recognizes Megatron final-validation syntax and requires the
final iteration-127, 160-ms validation to pass.  A passing canary authorizes
only the 381-step formal Stage A run.  It never authorizes Stage B.

