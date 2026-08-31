# uniss_phase3_e2e_rollin_continue_v1

One additional coverage epoch on top of
`endmargin_epoch23_15shard_20260824T190227Z/iter_0002264`, opening the
roll-in END/CONTINUE supervision that every run so far has left at weight zero.

Nothing in `uniss_phase3_v4_e2e_simuls2st_pilot15_v1` is modified: this
directory only sets `RUN_*` environment variables and calls that experiment's
own `scripts/run_e2e_megatron.sh`.  Data, geometry, seed, parallelism and the
frozen Stage-A audit are unchanged.

## What iter_0002264 is

| | |
|---|---|
| lineage | Stage-A v1 `iter_0000381` (failed its own gate) -> e2e structural canary -> `endmargin_epoch1` iter_0001132 -> `endmargin_epoch23` iter_0001207 (epoch 2) -> **iter_0002264 (epoch 3)** |
| data | 15-shard `task_pool_formal_p4_20260820T154500Z`, 1,325,243 records, 5 families, 1.717 B supervised tokens |
| geometry | 2264 updates, MBS 2 / GBS 128, seq 18000, 8 GPU, TP1/PP1, shuffle seed 20260819 |
| wall clock | 13.5 h for 2264 updates (~21.5 s/update) |
| objective | `semantic_end_ce` 0.50, `semantic_end_margin` 0.25 (logit margin 2.0); **every roll-in / continue / binary term 0.00**; prefix corruption 0.00, boundary roll-in rate 0.00 |
| result | cmn ASR 0.089, gold MT coverage 0.211, `natural_eos` 0.50, malformed 9, 6/8 speaking before source EOS |

## Why this configuration

The eager-speak oracle showed the model recognises on 82% of events but
translates or speaks on 17%, and that forcing it through the full grammar every
event lifts `natural_eos` from 0.50 to 1.00 while introducing repetition.  So
the missing supervision is the boundary decision itself -- when is there new
stable content -- and the current objective only ever penalises saying END in
the wrong place under teacher forcing.  It never supervises the decision under
the model's own history, which is the only condition that matters at inference.

`natural_eos` is 0.50 at iterations 1132, 1207 and 2264 alike: three coverage
epochs of the current objective moved it not at all.

## The loss terms

Unchanged, so the run stays comparable:

| term | weight | meaning |
|---|---:|---|
| `asr_ce`, `mt_ce`, `semantic_ce`, `replay_ce`, ... | as before | the content heads and the Phase-3 replay anchor |
| `semantic_end_ce` | 0.50 | teacher-forced cross-entropy at rows whose gold label is `END_SEMANTIC` |
| `semantic_end_margin` | 0.25, margin 2.0 | at the same rows, `relu(max_semantic_logit + 2.0 - end_logit)`: push END above every legal speech token |

Newly opened:

| term | weight | meaning |
|---|---:|---|
| `semantic_rollin_end_ce` | 0.25 | the same END cross-entropy, but only on rows reached through the model's **own** generated history |
| `semantic_rollin_continue_decision_margin` | 0.25, margin 1.0 | **the missing half.** At roll-in rows whose gold label is a speech token, `relu(end_logit + 1.0 - max_semantic_logit)` -- do not say END when you should keep speaking. CONTINUE is treated as set-valued: any legal speech token beats END, so it does not force an exact gold successor |
| `semantic_boundary_binary` | 0.50, margin 1.0 | calibrates the restricted binary score `z = end_logit - max_semantic_logit`. END rows minimise `softplus(1.0 - z)`, premature-END decision rows minimise `softplus(1.0 + z)`. Softplus has **no dead zone**, so the gradient survives once the relu margins are already satisfied, and the END and CONTINUE classes each receive half the weight regardless of their very unequal counts |

Deliberately left at zero:

| term | why |
|---|---|
| `semantic_prefix_corruption_rate` | the trainer refuses it together with boundary roll-in, and the `prefixcorr` canary moved nothing |
| `semantic_continue_margin` | teacher-forced tail only; the roll-in decision margin covers the same call under the condition that matters |
| `content_end_weight` | text END, not speech; out of scope here |

## The rate is not optional

`RUN_SEMANTIC_BOUNDARY_ROLLIN_RATE` gates the masks that all three roll-in
terms select on.  **At rate 0 they are identically zero no matter what weight
they carry** -- the same trap that left `real_prefix_kd`, `prefix_stability` and
`speaker_consistency` at exactly 0.0 for all 717 updates of the content-first
run.  This run sets rate 0.5 with a 100-update ramp, and
`tests/test_launcher_config.py` asserts that no roll-in weight can be enabled
while the rate is zero.

## Honest expectation

This is not a confirmed fix.  Eleven 100-update canaries already tried these
terms in various combinations and none moved gold coverage past 0.159 against a
0.145 baseline -- but every one of those measurements ran under
`append_only_commit`, which we now know caps gold coverage independently of
model quality (0.211 -> 0.495 on this very checkpoint from the commit policy
alone).  That prior evidence is therefore confounded, not negative.

Expected: `natural_eos` and premature END improve, coverage improves modestly.
Risk: the oracle showed that speaking more brings repetition (`of of`,
`new new`, `waiting for the right time to wait for the right time`), and none of
these terms penalise repetition.  Watch `s2s.semantic_length_ratio` and the
session's own text coverage, not just the gate checks.
