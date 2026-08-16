# Stage A exact matching-sample Phase3 offline ASR anchor

## Outcome

The exact 334-sample Phase3 offline anchor completed successfully on eight
H200 GPUs.  All samples produced a non-empty transcription and reached the
first `END_CONTENT` token.  This closes the v1 audit item that the offline
anchor and Stage A diagnosis used different sample IDs.

The matching anchor is:

- Chinese: **6.4873% CER**;
- English: **8.5038% WER**.

Stage A v1 remains rejected.  Both its causal-full and event-streaming
free-running results exceed the registered relative degradation limit of 15%.
Stage B therefore remains blocked.

## Immutable inputs and outputs

- Phase3 model:
  `checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf`
- Stage A validation packs:
  `data/megatron/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_causal_asr/valid_packs_18k_v1.jsonl`
- Exact matching manifest:
  `data/processed/uniss_phase3_v4_quality_first_true_streaming_pilot15_v2/stage00_matching_offline/matching_stage_a_334.jsonl`
- Manifest SHA-256:
  `6e9ddad232be5cd8a5ed65ffb9e984f34a7c2f210d302e539182ba268ddc23c4`
- Evaluation output:
  `eval_outputs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v2/stage00_matching_offline/matching334_phase3_20260816T235024Z`
- Summary SHA-256:
  `6971bc938974402053d8f5a1f11ec84fe399bcf8d88a93075b10130c7762c8dc`

The manifest contains the exact rotated max-two acoustic selection used at
coverage epoch zero in the formal Stage A evaluation:

| Task | Chinese | English | Total |
|---|---:|---:|---:|
| streaming ASR | 114 | 129 | 243 |
| causal-full ASR | 36 | 55 | 91 |
| total | 150 | 184 | 334 |

The raw parquet transcript and Stage A canonical transcript differ in all 334
records because Stage A applies its own ASR canonicalization.  The audio ID,
parquet row, source language, and source-audio path are exact.  Scoring uses
the Stage A canonical reference, while the raw transcript is retained in the
manifest for provenance.  The Quality prompt ends before target transcript
tokens, so this reference choice does not change the model input.

## Decoder-equivalence smoke

Before the full run, sample `NCSSD_R_EN_0000000261` was decoded two ways:

1. the existing full Phase3 Quality path through translation and semantic
   generation;
2. the new ASR-only path stopping at the first `END_CONTENT`.

The ASR-only output had 13 tokens and the full output had 325 tokens.  Their
ASR token prefix and decoded transcription were exactly identical.  The smoke
artifact is:

`eval_outputs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v2/stage00_matching_offline/parity_20260816T234908Z/PARITY_RESULT.json`

Its SHA-256 is
`b7cdd72529d2132bf1db343309f173843395aabc9393b9963ee9ce7215e8f709`.
This validates the shorter evaluator without changing Phase3 ASR predictions.

## Exact Phase3 results

| Matching subset | Metric | Samples | Reference units | Errors | Error rate |
|---|---|---:|---:|---:|---:|
| causal-full IDs, Chinese | CER | 36 | 1,121 | 79 | 7.0473% |
| causal-full IDs, English | WER | 55 | 951 | 79 | 8.3070% |
| streaming IDs, Chinese | CER | 114 | 3,303 | 208 | 6.2973% |
| streaming IDs, English | WER | 129 | 2,471 | 212 | 8.5795% |
| all Chinese | CER | 150 | 4,424 | 287 | **6.4873%** |
| all English | WER | 184 | 3,422 | 291 | **8.5038%** |

Health checks:

- 334/334 unique IDs covered exactly once;
- 0 empty hypotheses;
- 0 first-`END_CONTENT` failures;
- mean ASR-prefix generation time: 0.4483 seconds/sample/GPU;
- Chinese is normalized to simplified Chinese and scored character by
  character; English is lower-cased and scored by word.

## Stage A v1 comparison on the same IDs

Stage A values below aggregate the registered 160/320/640/1280 ms diagnosis.
The Phase3 anchor has one prediction per ID; Stage A has four predictions per
ID.  References and unique sample identities are nevertheless identical.

| System/path | Chinese CER | English WER | Chinese relative degradation | English relative degradation |
|---|---:|---:|---:|---:|
| Phase3 offline matching anchor | **6.4873%** | **8.5038%** | baseline | baseline |
| Stage A v1 causal-full | 15.8787% | 12.7760% | +144.76% | +50.24% |
| Stage A v1 event-streaming | 21.0112% | 35.3399% | +223.88% | +315.58% |
| maximum allowed by gate | 7.4604% | 9.7794% | +15.00% | +15.00% |

Even the easier causal-full path is above the gate by 8.4182 Chinese CER
points and 2.9967 English WER points.  Event-streaming is above the gate by
13.5508 Chinese CER points and 25.5606 English WER points.  This rules out the
possibility that the v1 failure was caused merely by comparing unrelated
validation samples.

## Interpretation and next gate

The exact anchor strengthens the v1 root-cause diagnosis:

1. Phase3 itself performs substantially better on these same utterances, so
   the large error is introduced by the causal frontend/training/runtime path.
2. Stage A event-streaming is much worse than its own causal-full path,
   especially in English.  The incremental event history and free-running
   exposure mismatch remain primary suspects.
3. The v1 `offline_teacher_kl` denominator was zero for the complete run; the
   declared 0.20 same-prefix teacher constraint never trained the model.
4. The final checkpoint still requires cached/full, future-perturbation, and
   rollback gates.  These must be real checkpoint evaluations, not anchored
   zero training terms.

The next authorized action is Stage A v2 implementation: provide a real
same-prefix teacher posterior with a non-zero audited denominator and
fail-fast behavior, then add checkpoint-level causality/parity gates.  A new
Stage A training run is allowed only after an eight-GPU smoke passes.  Stage B
must not start unless the new Stage A checkpoint meets all content and
causality gates.

Generated at `2026-08-16T23:53:14Z`.
