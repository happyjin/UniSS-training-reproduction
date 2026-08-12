# Generalize14 DAgger-prefix canary strict gate

## Verdict

Generalize14 completed all 200 Megatron iterations with zero skipped updates
and no non-finite loss.  It fixes the catastrophic no-WRITE behavior observed
in Generalize13: all eight strict runtime trials select a natural WRITE at
320--480 ms of source time, emit their first PCM by 599--896 ms wall-clock,
terminate with a natural EOS, and make no committed revision.

It nevertheless **fails the real streaming S2ST promotion gate**.  Generated
translations have only 0.065--0.20 character-sequence similarity to their
references, generated semantic units frequently collapse to repeated codes,
and every trial has RTF 1.11--1.44.  Generalize14 therefore demonstrates a
real latency-policy improvement, but not usable simultaneous translation.  It
must not be promoted to full15/full198.

## Reproducible inputs

- experiment: `uniss_phase3_runtime_parity_streaming_v2_generalize14_dagger_prefix_canary_v1`
- initialization: Generalize13 iteration 50
- training: 200 iterations, 8 GPUs, micro batch 2, global batch 128,
  sequence length 18000
- train packs: five canary trajectory packs, 128 formal sessions
- validation packs: two disjoint trajectory packs
- Phase3 replay: 10%
- trainable: Qwen LoRA, action/support/safe-commit heads and semantic
  microblock head
- frozen: Phase3 base, embedding/output matrix and causal frontend
- new supervision: scheduled text/semantic prefix roll-in,
  `runtime_prefix_recovery`, active grouped deadline survival
- strict runtime: natural action, text boundary, semantic CONTINUE/END and EOS;
  no forced WRITE, no oracle output length and no revision

TensorBoard remains available at:

```text
http://10.1.6.203:6087
```

## Teacher-forced held-out trajectory

| Iteration | Text loss | Text accuracy | Semantic loss | Semantic accuracy | Action accuracy | Predicted WRITE | Natural WRITE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 5.6917 | 19.70% | 7.7012 | 1.65% | 76.21% | 53.06% | 38.16% |
| 50 | **5.6768** | **20.39%** | **7.6532** | **1.69%** | 79.63% | 47.67% | 38.16% |
| 75 | 5.7940 | 17.76% | 7.7025 | 1.32% | **80.72%** | 46.87% | 38.16% |
| 100 | 5.9503 | 17.86% | 7.7377 | 1.26% | 78.60% | 47.94% | 38.16% |
| 125 | 6.1750 | 14.13% | 7.7934 | 1.12% | 79.40% | 46.47% | 38.16% |
| 150 | 6.3647 | 14.02% | 7.8260 | 1.23% | 79.21% | 46.20% | 38.16% |
| 175 | 6.5175 | 12.23% | 7.8320 | 1.11% | 78.68% | 46.72% | 38.16% |
| 200 | 6.5992 | 14.55% | 7.8336 | 1.11% | 78.72% | 46.52% | 38.16% |

Iteration 50 is the best held-out checkpoint.  At iteration 75 the schedule
has entered two-round roll-in and training prefix corruption reaches about
40.9%; at iteration 125 it reaches about 61.8%.  Action behavior remains
strong while text and semantic content deteriorate.  This is direct evidence
that the current independently sampled, fixed-length corruption is too severe
for the weak content heads.

## Strict real-PCM gate

| Iteration | Split | Sample | Generated text | Similarity | First WRITE source | First PCM wall | RTF | EOS/revision | Result |
|---:|---|---|---|---:|---:|---:|---:|---|---|
| 50 | held-out | `...0083` | 我能听到你的其他我听到你的声音。 | 0.1509 | 320 ms | 676 ms | 1.440 | natural / 0 | Fail |
| 50 | held-out | `...0261` | 我能听到你听到听起来 | 0.0645 | 320 ms | 612 ms | 1.205 | natural / 0 | Fail |
| 50 | train | `...0000` | 我能听到 | 0.1538 | 480 ms | 803 ms | 1.372 | natural / 0 | Fail |
| 50 | train | `...0002` | 我喜欢吃东西。 | 0.0930 | 320 ms | 613 ms | 1.352 | natural / 0 | Fail |
| 200 | held-out | `...0083` | 我听到起来在读书起来像个特别出色。 | 0.0741 | 320 ms | 901 ms | 1.290 | natural / 0 | Fail |
| 200 | held-out | `...0261` | 我为普通我 | 0.0769 | 320 ms | 599 ms | 1.164 | natural / 0 | Fail |
| 200 | train | `...0000` | 你好国。那家学校 | 0.2000 | 480 ms | 803 ms | 1.282 | natural / 0 | Fail |
| 200 | train | `...0002` | 我知道那听起来听起来像是一种的环境的变化。 | 0.1404 | 320 ms | 896 ms | 1.109 | natural / 0 | Fail |

The four non-overwriting evaluation roots are:

```text
reports/uniss_phase3_runtime_parity_streaming_v2/
  uniss_phase3_runtime_parity_streaming_v2_generalize14_dagger_prefix_canary_v1_held_out2_strict_v14_gate1/
  uniss_phase3_runtime_parity_streaming_v2_generalize14_dagger_prefix_canary_v1_train2_strict_v14_gate1/
  uniss_phase3_runtime_parity_streaming_v2_generalize14_dagger_prefix_canary_v1_held_out2_strict_v14_final1/
  uniss_phase3_runtime_parity_streaming_v2_generalize14_dagger_prefix_canary_v1_train2_strict_v14_final1/
```

Each sample directory contains source PCM, translated PCM, a stereo timeline
and the full event trace used for this conclusion.

## What Generalize14 proved

1. The active deadline objective plus model-prefix exposure is sufficient to
   turn the action policy from near-zero runtime WRITE into natural sub-second
   WRITE on both seen and held-out records.
2. This is not merely a lowered inference threshold: all decisions use the
   natural argmax action, and all samples reach a natural EOS.
3. The latency result alone is insufficient.  Text and semantic content are
   already wrong at the first few writes, and repeated semantic code `7645`
   becomes dominant in several final-checkpoint traces.
4. Per-tick generation and codec work takes longer than the 160 ms source
   cadence often enough to accumulate compute backlog, which explains the RTF
   failure despite a sub-second first PCM.

## Root cause and required next change

Generalize14 does not reproduce the actual runtime state distribution.  It
independently replaces fixed oracle text/semantic positions inside an
unchanged packed grammar.  Runtime instead creates a variable history from
its own action decision, emitted text length, semantic block length and prior
EOS/CONTINUE choices.  Consequently, the model is trained to repair random
token corruption but is evaluated on coherent, fully self-generated event
histories.  The action head learns the deadline; the content path does not
learn the true rollout.

Generalize15 must therefore use event-level runtime rollouts rather than a
higher corruption probability:

1. initialize from Generalize14 iteration 50, retaining its proven action
   behavior;
2. generate complete persistent-KV event prefixes with the exact inference
   grammar, including action, variable text, semantic length and EOS choices;
3. query the oracle continuation from each model-induced event state and
   optimize content/action correction only at those states;
4. preserve an explicit clean-oracle content anchor and semantic anti-collapse
   loss, and keep rollout probability bounded instead of rising to 50%;
5. profile or batch the per-tick text/semantic projections so the strict RTF
   gate can be below one without changing model decisions.

Promotion remains contingent on strict natural WRITE/PCM, translation,
playable non-collapsed PCM, natural EOS, zero revision and RTF below one.
