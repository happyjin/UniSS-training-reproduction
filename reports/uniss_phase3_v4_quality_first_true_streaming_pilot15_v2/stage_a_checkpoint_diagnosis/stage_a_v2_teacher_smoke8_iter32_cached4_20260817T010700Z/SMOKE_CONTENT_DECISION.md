# Stage A v2 iteration-32 smoke content decision

## Decision

**Runtime mechanics PASS; content quality FAIL; formal training remains allowed;
Stage B remains blocked.**

The 32-step run was designed to validate teacher-KL wiring, checkpoint save and
strict resume. It is not a converged ASR checkpoint. Its cached-runtime content
diagnosis confirms that distinction.

| Metric | Result |
|---|---:|
| Evaluations | 4 English rows (2 streaming, 2 causal-full) |
| Cached/recomputed hidden parity | 100% |
| Cached/recomputed GLM token parity | 100% |
| Cached/recomputed bridge parity | 100% |
| Cached/recomputed free generation parity | 100% |
| Committed rollback | 0 |
| Streaming WER | 100.00% |
| Causal-full WER | 88.24% |
| Event-stop success | 50.00% |

Both streaming rows produced empty text and failed the event stop. The two
causal-full rows produced non-empty but inaccurate text. This is not a cached
runtime bug: the independent recomputed reference generated the exact same
tokens. It is the expected non-convergence of a 32-step, four-pack structural
smoke.

The formal Stage A v2 run is authorized only by the already-passing structural
gate (causality, cached/full identity, bridge identity, cache growth, strict
resume). Formal checkpoints must later pass the exact 334-sample Chinese/English
content thresholds and zero-rollback gate before Stage B can start.
