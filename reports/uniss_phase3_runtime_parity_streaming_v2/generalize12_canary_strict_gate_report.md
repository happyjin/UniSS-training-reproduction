# Generalize12 causal microblock canary strict gate

## Verdict

Generalize12 completed all 200 canary iterations without a non-finite loss or
skipped update, but it **does not pass the streaming S2ST quality gate**.  It
improves semantic-unit diversity and preserves a sub-second first WRITE/PCM,
yet the runtime text path emits only short generic fragments even on examples
that were present in the canary training packs.

Consequently, this checkpoint must not be promoted to the full-15-shard run.
The next experiment must jointly adapt the runtime text/action path and the
causal semantic microblock path; adding epochs to the semantic-only head cannot
repair this failure.

## Training-time held-out validation

The two held-out packs were cyclically balanced across eight data-parallel
ranks.  The standalone PCM gate below remains unpadded and is the authoritative
result.

| Iteration | Semantic content loss | Token accuracy | First-slot accuracy | CONTINUE accuracy | Predicted unique fraction |
|---:|---:|---:|---:|---:|---:|
| 50 | 7.7696 | 1.758% | 4.785% | 79.078% | 50.20% |
| 75 | 7.6082 | 1.989% | 5.278% | 79.731% | 52.15% |
| 100 | 7.6173 | 1.864% | 4.715% | 79.957% | 52.25% |
| 125 | 7.6362 | 1.831% | 4.715% | 79.782% | 53.42% |
| 150 | 7.6570 | 1.936% | 4.584% | 80.219% | 54.88% |
| 175 | 7.6763 | 2.022% | 4.840% | 79.148% | 55.18% |
| 200 | 7.6874 | 1.805% | 4.559% | 79.279% | 55.96% |

The best validation content/first-slot point is iteration 75.  Later training
raises diversity but does not improve the semantic likelihood consistently,
which is a canary-overfitting signature.

## Strict held-out real-PCM results

All runs use natural WRITE, natural semantic CONTINUE/END, natural EOS, no
revision, no oracle output length, and a hard failure when a safety ceiling is
reached.

| Iteration | Sample | Generated / reference text similarity | First WRITE source time | First PCM wall time | RTF | Unique / total semantic units | Pass |
|---:|---|---:|---:|---:|---:|---:|---|
| 75 | `NCSSD_R_EN_0000000083` | 0.0976 | 320 ms | 649 ms | 1.350 | 49 / 380 | No |
| 75 | `NCSSD_R_EN_0000000261` | 0.0909 | 320 ms | 621 ms | 0.611 | 19 / 32 | No |
| 175 | `NCSSD_R_EN_0000000083` | 0.0816 | 320 ms | 635 ms | 1.239 | 119 / 328 | No |
| 175 | `NCSSD_R_EN_0000000261` | 0.0833 | 320 ms | 638 ms | 1.400 | 46 / 279 | No |
| 200 | `NCSSD_R_EN_0000000083` | 0.0909 | 320 ms | 640 ms | 1.299 | 168 / 379 | No |
| 200 | `NCSSD_R_EN_0000000261` | 0.0741 | 320 ms | 624 ms | 0.789 | 63 / 77 | No |

Artifacts, including source, translated PCM, timeline PCM and stereo files,
are under:

```text
reports/uniss_phase3_runtime_parity_streaming_v2/
  uniss_phase3_runtime_parity_streaming_v2_generalize12_microblock_canary_v1_held_out2_strict_v12_gate1/
```

## Seen-training-sample diagnostic

Iteration 200 was also evaluated on the first two formal training records,
both of which occur inside the five canary trajectory packs.

| Sample | Reference | Generated | Similarity | First PCM | RTF |
|---|---|---|---:|---:|---:|
| `NCSSD_R_EN_0000000000` | 你好，我在寻找一本关于正念的书。你能帮我吗？ | 我 | 0.0870 | 805 ms | 0.663 |
| `NCSSD_R_EN_0000000002` | 我对一份面向初学者的指南感兴趣，该指南能够解释基础知识并提供实用的练习。 | 我 | 0.0541 | 781 ms | 0.546 |

This rules out “wrong checkpoint selection” and “insufficient held-out data”
as the primary explanation.  Generalize12 freezes the Qwen LoRA, action head,
safe-commit head, frontend and text-generation path, and optimizes only the
semantic microblock head.  The trained head can diversify codec units, but it
has no gradient route capable of correcting the runtime text fragments or the
WRITE policy contexts that generate them.

## Required next experiment

Generalize13 must preserve the v12 causal semantic microblock decoder while
jointly training:

1. the existing Qwen LoRA branches on runtime-interleaved text deltas;
2. the natural action/support/safe-commit heads;
3. critical content/semantic/EOS boundaries; and
4. the semantic microblock content, final-length and CONTINUE/END heads.

Phase3 replay remains active as the quality anchor.  The chunk frontend and
base Phase3 weights remain frozen.  Promotion requires both seen-canary and
held-out text/PCM gates; a sub-second first PCM by itself is insufficient.
