# UniSS Phase3-v4 E2E Simultaneous S2ST native Megatron implementation report

> Date: 2026-08-18 UTC
>
> Experiment: `experiments/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/`
>
> Status: native training code and CPU/static gates implemented; formal GPU training not yet authorized

## 1. Outcome

The single-run E2E student now has a native Megatron training entrypoint.  It
does not construct a second Hugging Face Qwen inside Megatron and does not alter
the historical Phase1/2/3, Stage-A, wait-k, StreamSpeech, GRPO, evaluation, or
demo implementations.

The model is initialized from the complete V1 compound checkpoint:

```text
checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/
  stage_a_formal/stage_a_formal8_20260816T224100Z/iter_0000381
```

The native state namespace is reconstructed exactly:

```text
embedding.*
decoder.*
output_layer.*
stage_a_objective.frontend.*
stage_a_objective.bridge_norm.*
stage_a_objective.bridge_projection.*
stage_a_objective.ctc_head.*
```

All `stage_a_objective.*` parameters are frozen and excluded from the
optimizer.  Only the shared native Qwen is trained:

| Parameter group | Max LR | Purpose |
|---|---:|---|
| Qwen transformer body | `2e-6` | jointly learn ASR, MT, semantic and replay tasks |
| tied embedding/lm head | `5e-7` | protect the unified text/GLM/BiCodec vocabulary geometry |
| causal WhisperVQ, bridge and CTC | frozen | retain the selected V1 streaming-ASR frontend |

## 2. Implemented files

| File | Role |
|---|---|
| `training/pretrain_e2e_megatron.py` | native GPT model provider, V1 compound reconstruction, freezing, forward injection, loss reporting and Megatron lifecycle |
| `training/objective.py` | flattened Megatron CE, V1/Phase3 top-k KL, commit KL and distributed numerator/denominator aggregation |
| `training/schedule.py` | restart-exact five-family global schedule, validation blocks and bounded smoke prefix |
| `training/compute_geometry.py` | derive train iterations and warmup from the immutable interleaved task-pool count |
| `training/audit_v1_checkpoint.py` | static Phase3-to-V1 compound key and tree-fingerprint audit |
| `scripts/run_e2e_megatron.sh` | isolated 8-GPU launcher and GPU monitor |
| `tests/test_megatron_entrypoint.py` | exact key, frozen-parameter and chunk-curriculum tests |

## 3. Acoustic forward path

For the two acoustic families, the native forward path is:

```text
16 kHz source PCM
  -> frozen V1 causal WhisperVQ frontend, no_grad, chunk curriculum
  -> detached pooled hidden states
  -> frozen V1 nearest-code and bridge transform
  -> native Qwen embedding lookup at packed acoustic positions
  -> shared native Qwen forward/backward
```

The frontend forward is inside `torch.no_grad()`.  The pooled states are
detached.  The embedding lookup is intentionally outside `no_grad`, so the
Qwen embedding remains trainable at acoustic positions without sending a
gradient into WhisperVQ or the V1 bridge.

The single-run chunk curriculum is:

| Coverage progress | Chunks sampled |
|---:|---|
| 0--10% | 1280 / 960 ms |
| 10--35% | 960 / 640 ms |
| 35--70% | 640 / 320 ms |
| 70--100% | 320 / 160 ms |

## 4. Objective

The implemented active objective is:

```text
1.00 ASR delta CE
+ 1.00 incremental MT delta CE
+ 1.00 target semantic delta CE
+ 0.50 Phase3 quality/performance replay CE
+ 0.30 V1 same-prefix ASR top-k KL
+ 0.25 Phase3 MT/semantic top-k KL
+ 0.20 committed-prefix consistency KL
+ 0.10 balanced boundary/EOS CE
+ 0.00 speaker continuity
```

Speaker continuity is explicitly zero in this version because no genuine
cross-fragment speaker-embedding training sidecar exists.  The launcher rejects
any nonzero speaker-continuity weight.  This follows the plan's fail-open rule
and avoids fabricating supervision.  A real sidecar must be implemented before
restoring the planned `0.10` coefficient.

Each term records an independent numerator and denominator.  DDP reduction
first aggregates detached global numerators and denominators, then scales the
local differentiable numerator by world size over the global denominator.  It
therefore preserves the intended global token mean under uneven per-rank token
counts.  Boundary and EOS are globally normalized separately and then balanced
as two classes.

## 5. Five-family schedule

The optimizer global batch is homogeneous: every rank and every accumulated
microstep in one update uses the same family.  This protects the documented
task probabilities from semantic-token length bias.

```text
early:  ASR 40%, MT 0%,  E2E 20%, quality replay 24%, performance replay 16%
middle: ASR 32.5%, MT 10%, E2E 25%, quality replay 19.5%, performance replay 13%
steady: ASR 25%, MT 20%, E2E 30%, quality replay 15%, performance replay 10%
```

The formal update count is calculated from the final interleaved packed record
count, `GBS=128`, and three primary coverage epochs.  It is not copied from V1
381 updates or Phase3 9075 updates.

## 6. Checkpoint safety

The launcher forces:

```text
--load <V1 compound root>
--finetune
--no-load-optim
--no-load-rng
--dist-ckpt-strictness raise_unexpected
```

`raise_unexpected` is required for this fresh finetune handoff.  The V1
checkpoint contains optimizer and RNG entries from Stage A, but the E2E run
intentionally requests neither (`--no-load-optim --no-load-rng`).  Megatron's
`raise_all` treats those checkpoint-only entries as missing from the requested
state and rejects an otherwise exact model load.  `raise_unexpected` ignores
only checkpoint-only entries while still rejecting runtime-requested keys that
do not exist.  Before loading, the E2E entrypoint independently requires an
exact metadata-key match for every compound model tensor, including all frozen
`stage_a_objective.*` tensors; post-canary bitwise auditing remains mandatory.

The runtime denominator gate uses the objective's canonical metric name
`replay_ce` for both Phase3 replay families.  The task-pool build report keeps
the more descriptive count key `loss:phase3_replay_ce`; these two namespaces
must not be mixed because the packed loss-kind is normalized to `replay_ce`
inside the native Megatron objective.

The static audit on the real checkpoints passed:

| Audit | Result |
|---|---:|
| V1 tree SHA256 | `463ff5645ee3776f2c58343d4720cfb5beb55295972b68dd9f34cc48119fd730` |
| Phase3 native -> V1 native canonical keys | exact |
| Native canonical key count | 14 |
| V1 `stage_a_objective.*` canonical key count | 254 |
| Required frontend/bridge/CTC keys | present |

At model construction, an additional runtime audit compares the complete
native model sharded-state canonical key set with the V1 DCP metadata.  Any
missing or unexpected model key aborts before optimization.

## 7. Fresh real-data smoke

The latest immutable smoke output is:

```text
data/processed/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/
  formal_gold_20260818T090515Z/task_pools/
  task_pool_smoke32_acoustic_v3_20260818T205500Z/
```

Results:

| Check | Result |
|---|---:|
| source records | 32 |
| task families | 5/5 |
| sequence length | 18000 |
| commit-consistency pairs / positions | 362 / 2118 |
| acoustic rows | 64 |
| packed acoustic `source_glm` | present and length-exact |
| packed acoustic `packed_positions` | present and length-exact |
| runtime waveform | finite mono 16 kHz tensor |
| runtime `source_glm / packed_positions / waveform / source_glm_length` | all present |

The first loaded acoustic pack contained four real waveforms with 111040,
72640, 52160 and 102400 samples; their source GLM lengths were 87, 57, 41 and
80 respectively.

## 8. Tests

The complete isolated experiment test suite passes:

```text
65 passed
```

This includes data schemas, rollout, teacher caches, packing, runtime audio,
five-family synchronization, flattened losses, distributed normalization,
checkpoint key checks and frozen/trainable partition checks.

A post-implementation audit also corrected the validation split marker from a
plain `"valid"` string to Megatron's identity-compared `Split.valid` enum.  The
train, validation and bounded-smoke schedules now preserve their exact Split
objects, preventing validation from silently selecting the training batch-size
path.  A dedicated regression test covers all three schedule variants.

The smoke path is also bounded in both the launcher and Python entrypoint:
`--e2e-smoke` permits only one or two optimizer updates, and missing teacher
caches are legal only inside that bounded mode.  Therefore neither flag can be
used to bypass the explicit formal-training authorization gate.

Bounded smoke runs default to zero LR-warmup updates instead of inheriting the
formal geometry's 20-update warmup.  Any explicit warmup must remain between
zero and the selected train-iteration count; a two-update dry run now emits
`--lr-warmup-iters 0`, while a three-update smoke is rejected before torchrun.

The runtime collator now also defensively pads variable-length per-pack
`cu_seqlens` with trailing copies of 18000, exactly as Megatron's
packed-sequence flattener expects.  Current real packs are already expanded to
18001 entries by `boundaries_to_cu_seqlens`, so this is forward-compatible
hardening rather than a claim that current MBS-2 data was broken.  The padding
is stripped before FlashAttention and does not add or merge attention-visible
tokens.

## 9. Formal-training status and next gates

Formal training remains intentionally blocked.  The current gold gate still
has `formal_training_authorized=false`.  The following must complete before a
formal run can start:

1. finish train and valid V1 free-running rollout audits and strict quality stratification;
2. finish train and valid V1/Phase3 teacher caches;
3. build fresh formal train/valid task pools with the latest acoustic sidecar;
4. verify all active teacher denominators are nonzero over the validation interval;
5. run a 1--2 update 8-GPU structural/numerical smoke from V1;
6. run an all-family canary, free-running E-ASR/E-MT/E-S2S validation and frozen-parameter bitwise audit;
7. only then emit a gate with `formal_training_authorized=true` and start the three-coverage run.

The existing background rollout must not be stopped or replaced by synthetic
work.  The two idle waiters may be restarted in place when their gate logic is
hardened, because they have not yet produced teacher or task-pool outputs.

An additional immutable handoff is implemented for the post-cache CPU stage.
After all four V1/Phase3 train/valid teacher-cache audits exist and pass, a
separate waiter launches 64-worker construction of the formal train and valid
18k five-family task pools.  This handoff does not create or authorize a formal
training gate; GPU smoke, all-family canary and free-running validation remain
mandatory.

## 10. Strict rollout quality strata added during the formal run

The first rollout audit only guaranteed immutable alignment and summarized
quality; its status was not affected by malformed WRITE, early EOS, or content
error rate.  The formal handoff now adds a second, parallel hard gate that
classifies every sample exactly once:

- `clean`: no protocol error and English WER <= 0.30 or Chinese CER <= 0.20;
- `noisy_content`: protocol-valid but above the clean content threshold;
- `quarantine`: malformed WRITE, early EOS, or missing final EOS.

The hard gate requires at least 60% of samples to remain eligible for
rollout-dependent supervision, quarantine <= 40%, final EOS >= 99%, and at
least one accepted sample for every observed source language.  These limits
were checked against the largest 768-record ppg24 smoke: 69.27% accepted,
30.73% quarantined, and 100% final EOS.  Its weighted English WER was 52.53%;
that value is reported but intentionally is not a deletion gate.

The task-pool builder consumes the indexed strata manifest and enforces the
following routing:

| stratum | streaming ASR | gold MT | V1-history MT | interleaved E2E | Phase3 replay |
|---|---:|---:|---:|---:|---:|
| clean | yes | yes | yes | yes | yes |
| noisy_content | yes | yes | yes | yes | yes |
| quarantine | no | yes | no | no | yes |

A real 32-record task-pool smoke over the ppg24 rollout passed.  Six
quarantined samples were excluded from streaming ASR and interleaved E2E, 49
V1-history incremental-MT requests were excluded, and all six samples still
contributed gold MT and both Phase3 replay families.

## 11. Phase-stratified all-family learning-canary audit (2026-08-22)

The original bounded learning canary used the first `N` blocks of the formal
schedule.  That was not a valid all-family learning check: the formal early
phase assigns zero probability to `incremental_mt_event`, so the 10/25/50/100
prefix runs contained zero independent incremental-MT blocks.  Their MT changes
came only from the MT term inside `interleaved_e2e_s2st`.

Commit `fdfc774` adds an opt-in phase-stratified canary.  It reuses complete
global-batch blocks from the immutable formal schedule, samples its
early/mid/steady phases in 10%/25%/65% proportions, preserves phase order and
fails unless all five families are present.  It does not alter the formal
3395-update curriculum or authorize continuing a canary optimizer state.

The 50-update all-family run completed normally:

```text
learning_canary_allfamily_50u_20260822T041918Z
50/50 updates, 0 skipped, 0 NaN
runtime: 19m37s
family blocks: ASR 14, incremental MT 7, E2E 14, quality replay 9,
               performance replay 6
phase blocks: early 5, mid 13, steady 32
```

Its independent incremental-MT loss was active and learned over the short run:
`mt_ce` moved from 2.5923 to 2.5051, `phase3_kl` from 0.4535 to 0.4046 and
`commit_consistency` from 0.4892 to 0.4658.  The fixed-selection gate remained
failed.  Gold-source mean target coverage improved slightly versus the old 50u
prefix (16.35% -> 16.74%), but free-source coverage was only 5.55% and ASR
retention worsened because the number of dedicated ASR blocks dropped from 24
to 14.

| Canary | CMN CER | ENG WER | ASR malformed | Gold MT coverage | Free MT coverage | Semantic malformed | Non-silent audio |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10u prefix | 19.25% | 36.77% | 6 | 14.17% | 6.33% | 47 | 8/8 |
| 25u prefix | 19.25% | 37.42% | 6 | 14.17% | 7.90% | 51 | 8/8 |
| 50u prefix | 19.72% | 43.23% | 4 | 16.35% | 5.32% | 50 | 8/8 |
| 100u prefix | 18.78% | 46.45% | 3 | 14.36% | 5.78% | 58 | 8/8 |
| 50u all-family | 23.00% | 53.55% | 9 | 16.74% | 5.55% | 54 | 8/8 |

The 100-update all-family run also completed normally:

```text
learning_canary_allfamily_100u_20260822T050920Z
100/100 updates, 0 skipped, 0 NaN
runtime: 37m17s
family blocks: ASR 28, incremental MT 16, E2E 28, quality replay 17,
               performance replay 11
phase blocks: early 10, mid 25, steady 65
frozen Stage A tensors: 254/254 exact bitwise match
```

At update 98 its independent incremental-MT terms were finite
(`mt_ce=2.3277`, `phase3_kl=0.4936`, `commit_consistency=0.4312`).  At update
100 its interleaved terms were also finite (`ASR CE=2.0227`, `MT CE=4.6319`,
`semantic CE=5.3684`).  The fixed 16-record gate is still required before any
formal authorization.

Immediately after the 100u checkpoint and frozen audit completed, the runtime
container lost all `/dev/nvidia*` device nodes.  The kernel driver modules
remain loaded, but NVML reports zero visible devices.  This is an external GPU
device-mount incident, not a training failure.  Export and the 8-GPU gate are
queued to run automatically after all eight GPUs become visible and idle.

## 12. Model-history semantic-boundary roll-in audit (2026-08-23)

The fixed 16-record, 384-semantic-token gate showed that the earlier gold-history
termination losses did not train the state actually encountered at inference.
The model can reach a semantic boundary through its own incorrect token history
and then keep generating until the hard limit.  Four isolated 100-update,
phase-stratified canaries therefore tested increasingly targeted exposure to
model-generated boundary histories.  Every run used the same five-family
Megatron schedule, eight GPUs, `MBS=2`, `GBS=128`, immutable task pools and fixed
gate selection.  Every run completed with zero skipped updates, zero NaN values,
normal checkpoint saving and a bitwise-exact audit of all 254 frozen Stage-A
tensors.

The successive implementation commits were:

1. `9967a6a`: replace eligible semantic boundary inputs with the model's
   semantic continuation prediction;
2. `832a655`: make selection sample-aware so that one packed sample receives at
   most one boundary replacement;
3. `a224d03`: independently normalize hard END CE and END-vs-semantic margin
   only over selected model-history END rows;
4. `f612355`: add a gold-history pre-END continue-vs-END margin to counter early
   termination.

The fixed-selection results are:

| Variant | Semantic malformed segments | Semantic coverage mean | Semantic coverage min | CMN ASR CER | ENG ASR WER |
|---|---:|---:|---:|---:|---:|
| best no-roll-in termination baseline | 27 | 0.9774 | 0.8189 | 0.2066 | 0.4710 |
| token-level model-history roll-in | 34 | 0.9095 | 0.5280 | 0.2066 | 0.4774 |
| sample-aware model-history roll-in | 39 | 0.9695 | 0.7559 | 0.2066 | 0.4839 |
| sample-aware roll-in + hard END loss | 31 | 0.8909 | 0.4658 | 0.2019 | 0.4839 |
| sample-aware + hard END + static continue margin | 28 | 0.9002 | 0.4534 | 0.2066 | 0.5161 |

Sample-aware selection repaired the principal defect of token-level roll-in:
multiple boundary corruptions no longer accumulated inside one trajectory, so
mean coverage recovered from 0.9095 to 0.9695.  It did not solve termination by
itself.  Independently normalized hard END supervision then reduced malformed
segments from 39 to 31, demonstrating that supervision under the generated
history can suppress 384-token continuation.  Its simultaneous coverage drop
to 0.8909 shows the opposite failure mode: the model learned to end too early.
The static gold-history continue margin reduced malformed segments further to
28 but did not recover free-running coverage.  In the final canary its training
diagnostics were `signed END margin=+0.1970`, `roll-in END CE=2.8708`,
`continue margin loss=4.5026`, and `sample roll-in rate=49.43%`.  The mismatch
between a positive supervised END margin and poor free-running coverage confirms
that fixed gold-history continuation rows do not represent the histories on
which premature END is selected at inference.

The latest gate remains failed and formal full-data training remains blocked.
Its gold-source MT quality is also still unusable (`cmn->eng BLEU=3.8614`,
`eng->cmn BLEU=0.00868`), while English ASR retention fails.  Boundary
calibration alone cannot repair those independent content-retention failures.

### 12.1 Required next experiment: symmetric model-history roll-in

Further tuning of static END/CONTINUE weights is not justified.  The next
canary must expose both decisions under model-generated histories while
retaining the one-replacement-per-sample safety invariant:

- **END candidate:** at a gold END boundary, the model-generated input is a
  legal semantic continuation; the selected row learns END using independently
  normalized END CE and END-vs-semantic margin.
- **CONTINUE candidate:** inside the final semantic tail before reference END,
  a row where the model incorrectly prefers END supplies a model-generated
  legal semantic alternative as input; the following selected row learns a
  semantic continuation using a mask-specific continue-vs-END margin.
- END and CONTINUE candidates compete inside each packed sample, and at most one
  candidate is selected.  Deterministic selection hashes the update, packed
  row, sample ordinal, candidate type and candidate position.
- Candidate-type selection is configurable.  The initial canary targets an
  approximately 50/50 END/CONTINUE split, total sample roll-in rate 0.5 with a
  25-update ramp, tail length 12, END margin 2.0 and CONTINUE margin 1.0.
- The static `semantic_continue_margin` weight is zero.  The new
  mask-specific roll-in END and CONTINUE margin weights start at 0.25 and 0.10.
- Diagnostics must separately report eligible and selected END/CONTINUE
  samples, their sample rates, and the mask-specific signed CONTINUE margin.

This experiment is still a canary, not a continuation checkpoint and not
authorization for the formal run.  It must pass the full test suite, an 8-GPU
structural/numerical smoke, frozen-parameter audit and the same fixed-16/384
free-running gate before any larger training decision.

### 12.2 Symmetric model-history result and causal supervision gap

The symmetric implementation was completed in commit `a18d2c9`.  Its complete
test suite passed (`107 passed`), and the latest eight-GPU structural smoke
activated both candidate types with zero skipped updates and zero NaN values.
The smoke and the learning canary both retained all frozen Stage-A parameters
bitwise exactly.

The 100-update canary was:

```text
learning_canary_allfamily_100u_semendmargin_symmetricmodelhistory_
  h0p25_c0p1_r0p5_t12_strat_20260823T175027Z
```

It ran from 2026-08-23 17:51:08 UTC to 18:32:01 UTC with the unchanged
phase-stratified five-family schedule, eight GPUs, `MBS=2`, `GBS=128`, a
25-update roll-in ramp, total roll-in target 0.5 and conditional CONTINUE ratio
0.5.  The final checkpoint is `iter_0000100`.  All 254 frozen Stage-A tensors,
732,131,842 bytes and the complete frozen tree SHA256 matched the V1 reference
exactly.  The GPU trace reached 100% utilization, 581.39 W peak power and
140,211 MiB peak allocated memory per sampled device; the lower time-average is
caused by the heterogeneous five-family update costs and their host-side
transitions rather than missing ranks.

The model-history diagnostics did not become symmetric:

| Semantic E2E update | END signed margin | CONTINUE signed margin | END margin loss | CONTINUE margin loss | Total sample roll-in rate |
|---:|---:|---:|---:|---:|---:|
| 3 | -4.6518 | -4.9992 | 6.8253 | 6.0377 | 0.0581 |
| 25 | -3.9695 | -5.8963 | 6.1155 | 7.0919 | 0.5044 |
| 52 | -1.3517 | -7.9150 | 3.7023 | 9.0204 | 0.5037 |
| 73 | -0.3431 | -8.5113 | 2.7814 | 9.5694 | 0.5018 |
| 100 | +0.2343 | -8.6880 | 2.4609 | 9.7573 | 0.4952 |

At update 100 the END and CONTINUE selected-sample rates were 0.2316 and
0.3374, with denominators 64.5 and 137.125 respectively.  CONTINUE was
therefore active and independently normalized; its failure is not a zero-mask
or insufficient-sampling bug.  The diagnostic population is model-selected
and changes as training changes, so its absolute trajectory is a moving
hard-negative statistic rather than a fixed-probe learning curve.  The
free-running gate is the decisive test.

The same immutable fixed-16 selection and 384-semantic-token cap produced:

| Variant | Semantic malformed | Semantic coverage mean | Semantic coverage min | CMN CER | ENG WER | Gold MT coverage | Free MT coverage | Non-silent audio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| best no-roll-in termination baseline | 27 | 0.9774 | 0.8189 | 0.2066 | 0.4710 | 0.1446 | 0.1077 | 8/8 |
| token-level model-history roll-in | 34 | 0.9095 | 0.5280 | 0.2066 | 0.4774 | 0.1446 | 0.1077 | 8/8 |
| sample-aware model-history roll-in | 39 | 0.9695 | 0.7559 | 0.2066 | 0.4839 | 0.1446 | 0.1116 | 8/8 |
| sample-aware + hard END | 31 | 0.8909 | 0.4658 | 0.2019 | 0.4839 | 0.1446 | 0.1077 | 8/8 |
| sample-aware + hard END + static continue | 28 | 0.9002 | 0.4534 | 0.2066 | 0.5161 | 0.1446 | 0.1077 | 8/8 |
| symmetric model-history END/CONTINUE | 29 | 0.9293 | 0.4348 | 0.2019 | 0.4774 | 0.1446 | 0.1058 | 8/8 |

The symmetric run recovered some mean semantic coverage versus the static
continue run (0.9002 -> 0.9293), but it did not beat the no-roll-in baseline,
did not reduce malformed segments and further reduced minimum coverage.  The
gate also failed English ASR retention (`WER=0.4774 > 0.353399`), MT target
coverage, both BLEU/chrF retention checks and both structure checks.  Gold-source
MT remained `3.8614/20.4776` BLEU/chrF for cmn-to-eng and
`0.00868/2.9596` for eng-to-cmn.  Formal training is therefore still explicitly
unauthorized.

The result exposes a causal supervision gap in the CONTINUE construction.  In
the shifted packer, `tokens = token_ids[:-1]` and
`labels = token_ids[1:]`.  A CONTINUE candidate at input position `p` is chosen
because the no-gradient forward at `logits[p-1]` prefers `END_SEMANTIC` over
every legal semantic token.  The current implementation then rolls a legal
model semantic token into input position `p`, places its CONTINUE mask at `p`
and applies the margin to `logits[p]` against `labels[p]`.  That trains the
token *after* the forced semantic continuation, but it does not directly apply
a gradient to the premature-END decision at `logits[p-1]` that made the sample
eligible.  It is useful downstream model-history exposure, but it is not a
complete correction of the failing decision.

The next isolated repair must keep the existing one-replacement-per-sample and
deterministic-selection invariants while separating two masks and targets:

1. `continue_decision_mask` at `p-1`, with an independently normalized margin
   that pushes a legal semantic decision above `END_SEMANTIC` at the exact row
   that selected END;
2. `continue_history_mask` at `p`, retaining a smaller independently normalized
   next-semantic-vs-END margin after the model-generated input has been rolled
   in;
3. explicit diagnostics for the fixed decision row and the downstream history
   row, so a changing hard-negative population cannot hide which term learned;
4. the same 100-update phase-stratified canary, frozen Stage-A bitwise audit and
   fixed-16/384 gate before considering any formal run.

This is an objective-alignment repair, not a request to increase training
length or start the 3395-update formal schedule.  Repeating the present loss
for more updates would strengthen END calibration without supervising the
decision row that must continue, and is not authorized by these results.

### 12.3 Exact CONTINUE-decision supervision result

Commit `ed167ba` implemented the two-row CONTINUE repair without changing any
immutable task pool, teacher cache, checkpoint or gate selection:

- `semantic_rollin_continue_decision_margin` is evaluated at `p-1`, the exact
  row where the no-gradient restricted choice selected END over all semantic
  tokens;
- the existing `semantic_rollin_continue_margin` remains at `p`, after the
  model-generated semantic alternative is rolled into the history;
- the two masks are independently normalized and separately logged;
- a packed-sample boundary guard rejects any CONTINUE decision row that would
  cross into the preceding sample.

The complete isolated suite passed (`108 passed`, two existing dependency
warnings).  The fresh eight-GPU interleaved smoke was:

```text
semantic_continue_decisionrow_smoke_20260823T185457Z
```

It completed one `MBS=2`, `GBS=128` update with zero skipped and zero NaN
iterations.  Both CONTINUE denominators were exactly 139.375.  Their initial
signed margins were distinguishable: decision row `-0.3814`, downstream
history row `-4.6135`.  The DCP checkpoint saved normally, and all 254 frozen
Stage-A tensors remained bitwise exact.

The 100-update canary was:

```text
learning_canary_allfamily_100u_semendmargin_decisionrow_
  h0p25_d0p25_c0p025_r0p5_t12_strat_20260823T185924Z
```

It used the same eight-GPU phase-stratified schedule, `MBS=2`, `GBS=128`, total
roll-in target 0.5, conditional CONTINUE ratio 0.5 and 25-update ramp.  The
roll-in END margin weight remained 0.25, the new decision-row weight was 0.25,
and the downstream history-row weight was reduced to 0.025.  It ran from
18:59:25 to 19:40:28 UTC, completed with zero skipped and zero NaN updates, and
again passed the 254-tensor frozen Stage-A audit.  The GPU trace reached 100%
utilization, 587.07 W peak power and 140,209 MiB peak memory.

| Semantic E2E update | END signed margin | CONTINUE decision margin | CONTINUE history margin | END eligible samples | CONTINUE eligible samples |
|---:|---:|---:|---:|---:|---:|
| 3 | -4.7133 | -0.3204 | -5.7410 | 375.75 | 250.13 |
| 25 | -3.9358 | -0.8080 | -5.8449 | 379.38 | 358.63 |
| 52 | -1.4107 | -1.6603 | -7.7752 | 355.00 | 388.00 |
| 73 | -0.4958 | -1.9276 | -8.1147 | 307.75 | 385.13 |
| 100 | -0.0897 | -2.0331 | -8.1998 | 309.38 | 406.38 |

The exact-row term is active and correctly aligned, but the joint objective
still moves toward END.  END eligibility decreased while CONTINUE eligibility
increased.  This is not explained by raw row counts alone because every
special term is independently normalized.  The effective structural
coefficient is nevertheless strongly asymmetric: gold semantic-END CE 0.5,
gold END margin 0.25 and roll-in END margin 0.25 contribute three END-directed
terms, while the two CONTINUE-directed terms contribute 0.25 and 0.025.  The
general boundary and semantic losses add further shared pressure around these
rows.

The fixed-16/384 result was:

| Variant | Semantic malformed | Semantic coverage mean | Semantic coverage min | CMN CER | ENG WER | Gold MT coverage | Free MT coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-roll-in termination baseline | 27 | 0.9774 | 0.8189 | 0.2066 | 0.4710 | 0.1446 | 0.1077 |
| symmetric history-only | 29 | 0.9293 | 0.4348 | 0.2019 | 0.4774 | 0.1446 | 0.1058 |
| exact decision + history | 33 | 1.0000 | 1.0000 | 0.2066 | 0.4968 | 0.1508 | 0.1077 |

Exact decision-row supervision made the semantic-coverage check pass for the
first time: mean and minimum coverage were both 1.0.  It did not make the
structure valid.  All 33 malformed segments were exactly the 33 events that
reached the 384-token hard limit; there were no other malformed lengths.  The
repair therefore prevented premature END but displaced the error to the
opposite boundary: failure to END after a complete fragment.  All semantic
tokens were legal, all eight S2S samples produced non-silent PCM and no source
or target rollback occurred.

Independent content-retention gates also remain failed.  English WER was
49.68% against a 35.34% limit.  Gold/free MT target coverage was only
15.08%/10.77% with a zero minimum.  Gold-source cmn-to-eng improved slightly to
`4.1010/20.6382` BLEU/chrF, while eng-to-cmn remained
`0.00868/2.9596`.  These metrics cannot authorize formal training even if the
semantic grammar is repaired.

The next semantic experiment must replace the stack of independently weighted
END and CONTINUE hinge terms with one explicitly balanced binary boundary
calibration objective.  For signed score
`z = logit(END_SEMANTIC) - max(logit(legal semantic))`, use equal class means:

```text
0.5 * mean softplus(margin - z)       on model-history END rows
+ 0.5 * mean softplus(margin + z)     on exact premature-END CONTINUE rows
```

The existing special END CE/margins and downstream history hinge must be zero
in that isolated canary so the binary objective is not silently counted four
times.  Ordinary token CE, Phase3/V1 retention losses and the five-family
schedule remain unchanged.  The canary must log both class counts and signed
scores, then repeat the same frozen audit and fixed-16/384 gate.  This is still
not authorization for the formal 3395-update run.

### 12.4 Class-balanced binary boundary calibration result

Commit `ed05bb6` implemented the isolated class-balanced calibration objective.
For the common restricted-choice score

```text
z = logit(END_SEMANTIC) - max(logit(legal semantic))
```

the implementation forms two independent distributed `LossTerm` objects and
then combines their global class means as

```text
0.5 * mean softplus(margin - z)       # model-history END rows
+ 0.5 * mean softplus(margin + z)     # exact CONTINUE decision rows
```

This preserves a strict 50/50 class contribution even when the selected END
and CONTINUE row counts differ.  The launcher fails closed if this objective is
enabled together with any of the duplicate special terms: gold semantic-END
CE, gold END margin, roll-in END CE/margin, decision/history CONTINUE hinges or
the static CONTINUE-tail hinge.  Ordinary ASR/MT/semantic CE, general boundary
and EOS CE, V1/Phase3 retention and the phase-stratified five-family schedule
were unchanged.  New diagnostics report both class counts, the common signed
score for each class and the balanced loss.

The complete experiment suite passed (`111 passed`, two existing dependency
warnings).  The fresh eight-GPU smoke was:

```text
semantic_boundary_binary_smoke_b0p5_m1p0_20260823T195827Z
```

It used binary weight `0.5`, symmetric logit margin `1.0`, one interleaved
`MBS=2`, `GBS=128` update and roll-in rate `1.0`.  END/CONTINUE denominators
were `253.625/139.375`; their common signed scores were `-5.7762/+0.3815`, and
the balanced loss was `4.3980`.  The update completed with zero skipped and
zero NaN iterations.  All GPUs reached 98--100% sampled utilization, and all
254 frozen Stage-A tensors remained bitwise exact.

The phase-stratified learning canary was:

```text
learning_canary_allfamily_100u_sembinary_b0p5_m1p0_r0p5_t12_strat_
  20260823T200426Z
```

It ran from 20:04:48 to 20:45:02 UTC on 2026-08-23 with eight H200 GPUs,
`MBS=2`, `GBS=128`, binary weight `0.5`, margin `1.0`, total roll-in target
`0.5`, conditional CONTINUE ratio `0.5`, tail 12 and a 25-update ramp.  The run
completed all 100 updates with zero skipped and zero NaN iterations.  Peak GPU
utilization was 100% on every device, peak power was 573.71 W and peak sampled
memory was 140,209 MiB.  The final `iter_0000100` checkpoint again matched all
254 frozen Stage-A tensors, 732,131,842 bytes and the reference frozen-tree
SHA256 bitwise exactly.  Its TensorBoard directory is exposed locally at
`http://127.0.0.1:6043` for this audit session.

| Semantic E2E update | END count | CONTINUE count | END score `z` | CONTINUE score `z` | Balanced loss | Total sample roll-in rate |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 15.125 | 7.250 | -4.9767 | +0.3909 | 4.0737 | 0.0588 |
| 25 | 126.750 | 69.375 | -5.3179 | +0.2397 | 4.1695 | 0.5063 |
| 52 | 133.250 | 63.750 | -4.1489 | +0.2954 | 3.5687 | 0.5050 |
| 73 | 122.375 | 62.375 | -4.0173 | +0.2344 | 3.4580 | 0.4810 |
| 100 | 129.125 | 77.250 | -3.4127 | +0.3048 | 3.2135 | 0.5081 |

The moving hard-negative training statistic improved without the catastrophic
CONTINUE collapse seen in the previous canary: END score moved from `-4.98` to
`-3.41`, while CONTINUE stayed near `+0.30`.  The previous diagnostic used the
opposite sign (`best semantic - END`) and fell to `-2.03`, equivalent to a
current-score `z` of `+2.03`; the balanced run avoided that worsening.
However, both classes still ended on the wrong side of the requested margin.
An END row needs `z >= +1`, while a CONTINUE row needs `z <= -1`; the final
means were `-3.41` and `+0.30`.  A falling balanced loss therefore did not
establish a correct restricted binary decision rule.

The checkpoint was exported to Hugging Face format, fingerprinted as ten files
and 1,067,360,389 bytes with SHA256
`84b77e2fdbb3f9646121e2b478ce212c83750b8c584fc7a1024c0984398d0367`, then
evaluated with the immutable 16-record selection and 384-semantic-token cap:

```text
free_running_gate_sembinary_b0p5_m1p0_fixed16_384_20260823T200426Z
```

| Variant | Semantic malformed | Coverage mean | Coverage min | CMN CER | ENG WER | Gold MT coverage | Free MT coverage | Non-silent audio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no-roll-in termination baseline | 27 | 0.9774 | 0.8189 | 0.2066 | 0.4710 | 0.1446 | 0.1077 | 8/8 |
| symmetric history-only | 29 | 0.9293 | 0.4348 | 0.2019 | 0.4774 | 0.1446 | 0.1058 | 8/8 |
| exact decision + history | 33 | 1.0000 | 1.0000 | 0.2066 | 0.4968 | 0.1508 | 0.1077 | 8/8 |
| class-balanced binary | 64 | 1.0000 | 1.0000 | 0.2019 | 0.4903 | 0.1530 | 0.1116 | 8/8 |

All 64 malformed segments were exactly 384 semantic tokens; there were no
malformed segments at any other length.  They occurred across all eight S2S
samples, with 4--10 capped events per sample.  Thus semantic tokens remained
legal, PCM remained non-silent and coverage stayed complete, but the model
failed to emit a natural END more often than the previous exact-decision run
(`33 -> 64` capped events).  The isolated binary hypothesis did not repair the
termination process.

Content retention also remains an independent blocker.  CMN CER passed at
20.19%, but English WER was 49.03% against a 35.34% limit.  Gold-source
cmn-to-eng reached `4.1399/21.7212` BLEU/chrF, while eng-to-cmn remained
`0.00868/2.9596`.  Gold/free target coverage was only 15.30%/11.16%, both with
zero minimum coverage.  The gate status is `failed` and
`formal_training_authorized=false`.

This result rejects class-weight imbalance as the sole semantic-boundary root
cause.  The binary term is applied only to the small, changing set of selected
model-history rows (about 129 END and 77 CONTINUE rows at update 100).  Removing
the broad gold END anchors made the isolated causal test clean, but also left
far less coverage of the many possible natural termination contexts.  Simply
raising the binary weight, restoring the old stack unchanged or running more
updates would repeat the same moving-selection problem and is not justified.

Any next semantic experiment should change supervision coverage rather than
only coefficients: build an immutable, class-matched boundary replay sidecar
containing fixed exact-decision END and CONTINUE rows across every semantic
fragment, include both gold-history and model-history contexts, and measure
the same fixed rows throughout training.  That design must retain a broad END
anchor without reintroducing the previous multi-term asymmetry.  Separately,
the English ASR and especially eng-to-cmn MT/data path must be repaired before
any formal full-data run can be authorized.  No 3395-update formal training was
started.
