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
--dist-ckpt-strictness raise_all
```

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
62 passed
```

This includes data schemas, rollout, teacher caches, packing, runtime audio,
five-family synchronization, flattened losses, distributed normalization,
checkpoint key checks and frozen/trainable partition checks.

## 9. Formal-training status and next gates

Formal training remains intentionally blocked.  The current gold gate still
has `formal_training_authorized=false`.  The following must complete before a
formal run can start:

1. finish train and valid V1 free-running rollout audits;
2. finish train and valid V1/Phase3 teacher caches;
3. build fresh formal train/valid task pools with the latest acoustic sidecar;
4. verify all active teacher denominators are nonzero over the validation interval;
5. run a 1--2 update 8-GPU structural/numerical smoke from V1;
6. run an all-family canary, free-running E-ASR/E-MT/E-S2S validation and frozen-parameter bitwise audit;
7. only then emit a gate with `formal_training_authorized=true` and start the three-coverage run.

The existing background rollout and teacher-cache waiter must not be stopped or
replaced by synthetic work.
