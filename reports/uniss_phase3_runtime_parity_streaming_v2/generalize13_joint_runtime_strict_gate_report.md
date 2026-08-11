# Generalize13 joint-runtime canary strict gate

## Verdict

Generalize13 `canary_v2` completed all 200 Megatron iterations with zero skipped
updates and no non-finite loss, but it **fails the real streaming S2ST gate on
both seen training records and held-out records**.  The run must not be promoted
to full15/full198.

The failure is stronger than ordinary held-out overfitting.  Under the exact
persistent-KV runtime, most samples never select a natural WRITE.  The only
seen-sample sub-second WRITE at iteration 200 emits just `你好，`; held-out
translation similarity is at most 0.0976.  A low RTF therefore does not
represent successful simultaneous translation.

## Reproducible inputs

- experiment: `uniss_phase3_runtime_parity_streaming_v2_generalize13_joint_runtime_canary_v2`
- base: completed Generalize12 iteration 200
- train packs: five canary trajectory packs, 128 formal sessions
- validation packs: two disjoint trajectory packs
- Phase3 replay: 10%
- trainable: Qwen LoRA, action/support/safe-commit heads, causal semantic
  microblock head
- frozen: Phase3 base, embedding/output matrix, causal frontend
- strict runtime: natural action, text boundary, semantic CONTINUE/END and EOS;
  no forced WRITE, no oracle output length, no revision

The first two training records were verified against `source_ids` in the
actual packed canary data.  They are `NCSSD_R_EN_0000000000` and
`NCSSD_R_EN_0000000002`; the held-out records are
`NCSSD_R_EN_0000000083` and `NCSSD_R_EN_0000000261`.

## Teacher-forced validation

| Iteration | Held-out text accuracy | Text loss | Microblock accuracy | END_CONTENT accuracy | Action accuracy |
|---:|---:|---:|---:|---:|---:|
| 25 | 17.49% | 5.7009 | 1.86% | 83.94% | 85.02% |
| 50 | **22.43%** | **5.6324** | 1.76% | **80.88%** | **85.57%** |
| 75 | 21.01% | 6.1457 | 1.61% | 75.90% | 83.99% |
| 150 | 18.86% | 7.5752 | 1.59% | 77.42% | 84.24% |
| 175 | 16.07% | 7.7359 | 1.59% | 78.74% | 84.35% |
| 200 | 16.76% | 7.8157 | 1.62% | 77.52% | 84.06% |

At iteration 200, training text accuracy reaches 76.62% and training
microblock accuracy reaches 10.24%, while held-out text loss has already risen
to 7.8157.  Iteration 50 is consequently the best teacher-forced held-out
checkpoint, but the strict PCM gate below shows that it is not a usable
runtime checkpoint.

Teacher-forced action statistics also expose why aggregate accuracy is not a
runtime guarantee.  Held-out labels contain 38.16% WRITE actions and the model
predicts 33.09% WRITE at iteration 200 with 84.06% aggregate accuracy, yet its
self-generated histories lead to zero WRITE on both held-out test records.

## Strict held-out real-PCM gate

| Iteration | Sample | Generated text | Similarity | First WRITE source | First PCM wall | RTF | Result |
|---:|---|---|---:|---:|---:|---:|---|
| 50 | `...0083` | 我希望能 | 0.0488 | 6560 ms | 6888 ms | 0.646 | Fail |
| 50 | `...0261` | *(empty)* | 0.0000 | none | none | 0.498 | Fail |
| 75 | `...0083` | 我真真的 | 0.0976 | 6560 ms | 6931 ms | 0.616 | Fail |
| 75 | `...0261` | *(empty)* | 0.0000 | none | none | 0.488 | Fail |
| 200 | `...0083` | *(empty)* | 0.0000 | none | none | 0.501 | Fail |
| 200 | `...0261` | *(empty)* | 0.0000 | none | none | 0.486 | Fail |

## Strict seen-training real-PCM gate

| Iteration | Sample | Generated text | Similarity | First WRITE source | First PCM wall | RTF | Result |
|---:|---|---|---:|---:|---:|---:|---|
| 50 | `...0000` | *(empty)* | 0.0000 | none | none | 0.536 | Fail |
| 50 | `...0002` | *(empty)* | 0.0000 | none | none | 0.547 | Fail |
| 75 | `...0000` | *(empty)* | 0.0000 | none | none | 0.565 | Fail |
| 75 | `...0002` | *(empty)* | 0.0000 | none | none | 0.502 | Fail |
| 200 | `...0000` | 你好， | 0.2400 | 640 ms | 1074 ms | 0.711 | Fail |
| 200 | `...0002` | *(empty)* | 0.0000 | none | none | 0.534 | Fail |

Artifacts containing source PCM, generated PCM/timelines, event traces and
machine-readable summaries are under the six non-overwriting directories:

```text
reports/uniss_phase3_runtime_parity_streaming_v2/
  uniss_phase3_runtime_parity_streaming_v2_generalize13_joint_runtime_canary_v2_{held_out2,train2}_strict_v13_gate1/
```

## Root cause

1. **Teacher-forcing exposure mismatch.**  The packed objective always
   conditions later ticks on oracle action/text/semantic history.  Runtime
   conditions on its own generated history.  The 84% teacher-forced action
   accuracy and near-zero runtime WRITE rate are direct evidence of this gap.
2. **No active latency objective.**  Generalize13 assigns zero weight to
   `deadline_survival`, even though the promotion gate requires a first WRITE
   before one second.  On held-out `...0083`, the oracle's first WRITE is at
   320 ms but the model waits until 6560 ms.
3. **Canary memorization rather than robust recovery.**  Training text accuracy
   rises from 17% to 77%, while held-out text loss starts worsening after
   iteration 50.  Continuing the same teacher-forced objective amplifies this
   divergence.
4. **Semantic exposure mismatch remains.**  The microblock head is trained on
   oracle prior semantic units inside and between blocks, but runtime feeds its
   own units back into Qwen.  Held-out microblock accuracy stays near 1.6%.

## Required Generalize14 change

Generalize14 should initialize from Generalize13 iteration 50 and perform
scheduled model-prefix training inside the isolated Megatron entrypoint:

1. run a no-gradient probe to predict text and causal semantic tokens;
2. replace a scheduled fraction of oracle generated-prefix inputs with those
   predictions while preserving sequence length and grammar;
3. run a second probe from the corrupted prefix, then optimize oracle next
   tokens/actions from that model-induced state (DAgger-style oracle
   correction);
4. add a separately reported recovery CE over corrupted-prefix positions;
5. activate grouped soft/hard deadline survival and retain Phase3 replay;
6. keep the base Phase3 weights, embeddings and causal frontend frozen.

Promotion still requires strict real-PCM success on both seen and held-out
records.  Lowering the runtime action threshold or extending the current run
alone is not an acceptable substitute for this training correction.
