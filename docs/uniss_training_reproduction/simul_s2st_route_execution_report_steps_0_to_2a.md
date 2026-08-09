# Simul-S2ST route — execution report for Steps 0, 1 and 2a

> Research report. All experiments live under `experiments/simul_s2st_route_v1/`; no existing
> training script, runtime or checkpoint was modified. Machine: 8× NVIDIA H200.
> Companion data reports are in `reports/simul_s2st_route_v1/`.

This executes the first three items of the plan in
[`simul_s2st_route_decision_and_recommendation.md`](./simul_s2st_route_decision_and_recommendation.md).
Two of the three came back differently from what that plan assumed, and both differences
change what should be built next.

---

## 0. What changed in the plan

| Step | Plan assumed | Measured | Consequence |
|---|---|---|---|
| 0 — RTF decomposition | Qwen AR decode dominates; if so, build the NAR head | Confirmed: AR decode 52.5% of wall clock, and a further 34.7% is the offline fallback that fires when no WRITE is accepted | NAR head confirmed as first priority, and the fallback is a second, cheaper target |
| 1 — V6 checkpoint recheck | Check whether BLEU fell along with agreement | Agreement fell ~5×; BLEU did not fall, and the final checkpoint slightly beats the teacher stream | The Stage B gate was measuring the wrong thing. V6 checkpoints are worth keeping |
| 2a — upsample ratio | Measure it; expect something smaller than the inherited 48 | 48 is **too small** — it silently drops 2.25% of training rows — and 99.9% coverage needs 96, at 4× the attention cost | The constant-ratio design itself is wrong; anchor the frame budget to duration instead |

---

## 1. Step 0 — where the streaming wall clock actually goes

Full data: [`step0_rtf_decomposition_v1.md`](../../reports/simul_s2st_route_v1/step0_rtf_decomposition_v1.md)
and [`step0b_qwen_forward_profile_v1.md`](../../reports/simul_s2st_route_v1/step0b_qwen_forward_profile_v1.md).

### 1.1 Method

A monkey-patching harness (`experiments/simul_s2st_route_v1/common/instrumentation.py`)
wraps the Stage09/10/11 runtime in a hierarchical wall-clock timer and restores every patched
attribute exactly on exit, so the shipped runtime is untouched. Two passes run over the same
16 samples (8 per direction, evenly spaced across the whole valid split): an unpatched
baseline for RTF, and an instrumented pass for the shares. Instrumentation overhead on total
wall clock was 0.5%.

### 1.2 Result

| Bucket | Share of streaming wall | RTF contribution |
|---|---:|---:|
| `qwen_ar_decode` | 52.5% | 3.322 |
| `offline_fallback` | 34.7% | 2.197 |
| `qwen_prefill_source` | 8.0% | 0.505 |
| `source_runtime` (mel + Emformer + CTC + bridge + policy) | **2.4%** | 0.152 |
| `qwen_prefill_wait` | 2.2% | 0.141 |
| `codec_stream_push` (BiCodec vocoder) | **0.1%** | 0.005 |

Pooled compute RTF is 6.30 on the baseline pass.

Two things stand out. The source frontend — the component that three stages of work were
spent on, and the component the Stage B gate was defending — accounts for **2.4%** of the
wall clock. And BiCodec vocoding, which the plan listed as a possible bottleneck worth an
incremental codec, accounts for **0.1%**. Neither is worth optimising.

The `offline_fallback` bucket is 34.7%: on 9 of 16 samples no WRITE was ever accepted, so the
engine fell back to generating the whole utterance offline at the end. That is not a latency
optimisation problem, it is a policy/quality problem, and it is why computed availability
latency (33.5 s) is so much worse than non-computational latency (4.6 s).

### 1.3 The 25 ms Qwen forward

`qwen_forward_ids` is called 10,752 times at 25.43 ms each. A separate microbenchmark
isolates what that 25 ms is:

- **It is not arithmetic.** A one-token forward costs 24.40 ms at batch 1 and 26.14 ms at
  batch 32 — a 32× increase in work for a 7% increase in time. The cost is fixed per-call
  launch overhead across 24 layers, not attention over the KV cache.
- **Merging the 48 LoRA adapters gives 1.31×** (24.40 → 18.63 ms) and is numerically an
  identity: argmax agreement 100.00% over the test batch.
- **The repetition penalty is a Python loop** issuing three CUDA operations per distinct
  generated token. It costs 4.60 ms per call in the pipeline — 8.0% of total wall clock. A
  vectorised replacement is bit-identical (max abs delta 0) and up to 98.8× faster at 512
  history tokens.

So there is roughly 1.35× available from two changes that alter no model output, and the
remaining AR cost is bounded below by the per-call launch overhead — which is exactly the
argument for replacing 50 sequential AR steps per second of audio with one NAR forward.

---

## 2. Step 1 — the Stage B gate was reading the wrong signal

Full data: [`step1_v6_bleu_recheck_v1.md`](../../reports/simul_s2st_route_v1/step1_v6_bleu_recheck_v1.md).

### 2.1 Method

For each joint-V6 checkpoint, run the V6 frontend (chunked WhisperVQ → STE bridge) to a
source GLM token stream, feed that stream to the **unchanged** Phase3 export
(`iter_0009075`), and score bidirectional Text-BLEU on 16 samples per direction. Agreement
against the manifest teacher stream is computed on the same samples, so the two quantities
are finally comparable.

Two controls make the comparison interpretable:

- **`manifest_teacher_glm`** — the manifest's own offline GLM stream, agreement 100% by
  construction. This is the ceiling.
- **`pretrained_frontend`** — the same frontend before any V6 training. This separates the
  cost of chunking from the cost of training.

Megatron `torch_dist` shards are read in a single process by staging tensors at the
checkpoint's own dtype; all 604 model tensors loaded for every checkpoint, with none missing
and none unused.

### 2.2 The backend really is fixed

| Checkpoint | Qwen tensors changed | Max abs delta |
|---|---:|---:|
| `stage_a_iter500` | 0 / 291 | 0 |
| `stage_b_iter250` | 178 / 291 | 9.5e-07 |
| `stage_b_iter5000` | 227 / 291 | 3.1e-05 |

Stage A left Qwen bit-identical, and Stage B's `LR_QWEN_MULT=0.001` moved it by at most
3e-05. Every difference below therefore comes from the source stream, not from backend drift.

### 2.3 Result

| Stream | Chunk | Agree EN→ZH | BLEU EN→ZH | Agree ZH→EN | BLEU ZH→EN |
|---|---|---:|---:|---:|---:|
| `manifest_teacher_glm` | offline | 100.00% | 44.10 | 100.00% | 38.13 |
| `pretrained_frontend` | offline | 25.48% | 44.40 | 19.30% | 31.80 |
| `stage_b_iter250` | offline | 26.75% | 41.04 | 21.23% | 32.77 |
| `stage_b_iter2500` | offline | 19.24% | 42.09 | 10.24% | 36.99 |
| `stage_b_iter5000` | offline | **13.27%** | **44.73** | **4.75%** | **39.89** |
| `pretrained_frontend` | 320 ms | 18.61% | 41.27 | 7.67% | 31.28 |
| `stage_b_iter5000` | 320 ms | 8.59% | 42.98 | 2.57% | 22.35 |

Three readings:

**Agreement collapsed and quality did not follow it.** Across Stage B the offline stream's
agreement falls about 5× in ZH→EN (21.23% → 4.75%), which is precisely what the safety gate
saw. Over the same checkpoints, offline BLEU rises: 32.77 → 39.89 in ZH→EN and 41.04 → 44.73
in EN→ZH. The final Stage B checkpoint slightly **exceeds** the 100%-agreement teacher stream
in both directions.

**Even an untrained frontend is fine.** `pretrained_frontend` at the offline chunk agrees
with the manifest teacher on only 25.5% / 19.3% of positions, yet scores 44.40 / 31.80 —
statistically indistinguishable from the teacher on EN→ZH. Two token streams that differ at
three quarters of all positions produce the same translations, because the Phase3 decoder
consumes the GLM **embedding geometry**, not exact codebook indices.

That also explains why the offline agreement was never near 100% even before any training:
the joint frontend's GPU mel path and its shorter padding differ from the original GLM4
tokenizer's 30-second-padded preprocessing, so it lands on different-but-equivalent codebook
neighbours. This reproduces the training-time `teacher_glm_agreement` (17–37%) reported in the
[Stage B failure analysis](./uniss_phase3_whisper_streamspeech_joint_v6_full198_stage_b_failure_analysis.md),
which confirms the probe is measuring the same quantity the gate was.

**The remaining cost is chunking, and Stage B made it lopsided.** Comparing the same
checkpoint at 320 ms versus offline is a clean paired contrast. `pretrained_frontend` loses
3.1 BLEU in EN→ZH and 0.5 in ZH→EN. `stage_b_iter5000` loses only 1.8 in EN→ZH but **17.5**
in ZH→EN. So Stage B slightly improved EN→ZH robustness to a truncated right context while
making ZH→EN drastically worse — unsurprising, since nothing in the objective rewarded
behaving well under a short chunk.

### 2.4 Honest limits

With 16 samples per direction, individual BLEU differences of two or three points are not
significant, and correlations over seven checkpoints are unstable (Pearson between agreement
and BLEU ranges from −0.88 to +0.57 depending on subset). The robust claim is not "lower
agreement is better." It is that agreement varies more than tenfold across these rows —
2.57% to 26.75% — while BLEU stays inside a band that contains the 100%-agreement teacher.
If agreement carried the signal the gate assumed, a tenfold swing would be visible. It is not.

The one place quality genuinely degrades is ZH→EN at the 320 ms chunk late in Stage B (22.35
at iteration 5000). That is worth watching, but it is a streaming-robustness failure, not the
agreement failure the gate reported.

### 2.5 Consequences

1. **Keep the V6 checkpoints.** `stage_b_iter5000` is the best offline source frontend
   measured here, better than the manifest teacher stream on this sample.
2. **Demote `teacher_glm_agreement` to a diagnostic.** It must not gate training. The gate
   should watch frozen-backend BLEU on a fixed probe set, which this harness now provides.
3. **Stop trying to raise agreement.** The ceiling audit already showed 320 ms caps out near
   0.5465; this shows the target was not worth reaching in the first place.

---

## 3. Step 2a — the NAR head's frame budget is parameterised wrongly

Full data: [`step2a_upsample_ratio_v1.md`](../../reports/simul_s2st_route_v1/step2a_upsample_ratio_v1.md).

### 3.1 Method

`NARBiCodecCTC` expands each target text token into `upsample_ratio` CTC frames with
`repeat_interleave`, then runs a causal Transformer over the expansion. The ratio therefore
controls both whether a CTC path exists and how expensive the head is (attention is quadratic
in `ratio × text_length`).

Feasibility uses the same rule the training loss uses — CTC needs one extra frame between
consecutive identical labels, so a row is feasible when
`ratio × text_length ≥ unit_length + adjacent_repeats`. A unit test pins that rule against
what `torch.nn.functional.ctc_loss` can actually align at the exact boundary.

Measured over 399,987 utterances drawn at evenly spaced byte offsets from the 66 GB
`joint_train.jsonl` plus all of `joint_valid.jsonl`.

### 3.2 The shipped ratio silently discards data

| Ratio | Feasible rows | Infeasible | Lattice occupancy | Relative attention cost |
|---:|---:|---:|---:|---:|
| 32 | 91.543% | 33,826 | 58.2% | 0.444× |
| **48 (current)** | **97.748%** | **9,007** | 41.3% | 1.000× |
| 64 | 98.973% | 4,107 | 31.7% | 1.778× |
| 96 | 99.573% | 1,707 | 21.5% | 4.000× |

At the shipped ratio 48, **2.25% of target utterances have no CTC path**. Those rows are
dropped from the `bicodec_ctc` loss by `ctc_normalized_loss` and counted only in the
`unit_infeasible` diagnostic — and they are systematically the long, fast-speech utterances
that matter most for streaming. Reaching 99.9% coverage on well-aligned rows requires ratio
96, which costs 4× the unit decoder's attention and leaves 78.5% of the lattice as padding.

Separately, 0.36% of rows are genuinely misaligned rather than merely long: their BiCodec
length sits exactly at the 3,000-token cap (60 s at 50 Hz) against a median of 11 text tokens.
Those are a data-cleaning item, not evidence that the ratio needs to be larger.

### 3.3 Text length is the wrong anchor

The reason no single ratio works is that frames-per-text-token is a wide distribution. The
alternative anchor — source audio duration — is far narrower:

| Anchor | Frames per anchor p50 | p95 | p99 | Coefficient of variation | p99/p50 |
|---|---:|---:|---:|---:|---:|
| target text tokens | 18.4 | 36.8 | 57.3 | 0.442 | **3.11×** |
| source audio seconds | 52.8 | 70.0 | 74.6 | **0.200** | **1.41×** |

This is expected once stated: BiCodec runs at a fixed 50 Hz, so the number of target frames
is a property of *time*, and the p50 of 52.8 frames per source second is essentially that
frame rate. Text token count, by contrast, varies with language and tokenizer — the median
EN→ZH target is 19 text tokens against 11 for ZH→EN.

A constant ratio must be sized for the tail, so the anchor's p99/p50 spread is directly the
padding factor the head pays on a typical utterance. Switching anchors takes that from 3.11×
to 1.41×, which is roughly `(3.11/1.41)² ≈ 4.9×` less attention work in the unit decoder,
before any other optimisation.

### 3.4 A second defect found while reading the head

`NARBiCodecCTC.forward` applies `src_key_padding_mask` and a causal mask to the unit decoder,
but calls `self.t2u_encoder(...)` with **neither**. The text-to-unit encoder is therefore
bidirectional over the full target text and attends to padding. For an offline head that is
merely wasteful; for the streaming head Step 2 is meant to produce, it is disqualifying,
because it lets every output frame see the entire future translation. This must be fixed
before any latency number from this head is meaningful.

---

## 4. Revised plan for Step 2

The plan's ordering survives — the NAR CTC head is still the right first build, and Step 0
strengthens that. What changes is the head's design:

1. **Anchor the frame budget to duration, not text tokens.** Set the CTC input length per
   utterance from elapsed source audio (≈75 frames/s covers p99) rather than a constant
   multiple of text length. This removes the infeasible-row class entirely and cuts the
   decoder's attention work by roughly 5×.
2. **Make the T2U encoder causal and padding-aware** before measuring anything.
3. **Fix the loss accounting.** `unit_infeasible` should be a hard failure in a streaming
   head's training, not a silent counter, now that we know it fires on 2.25% of rows.
4. **Take the free 1.35× first.** Merge the LoRA adapters and vectorise the repetition
   penalty — both verified numerically identical, both independent of the head.
5. **Gate on frozen-backend BLEU**, using the Step 1 harness, not on teacher agreement.

Step 3 (Λ-shaped KV cache + wait-k) is unchanged, but Step 0 adds a prerequisite: the
`offline_fallback` path fires on 9 of 16 samples and costs 34.7% of the wall clock. Until the
policy accepts WRITEs on most utterances, any Pareto curve drawn against LAAL will be
measuring the fallback rather than the streaming path.

---

## 5. Reproduction

```bash
# Step 0 — wall-clock decomposition and Qwen microbenchmark
bash experiments/simul_s2st_route_v1/step0_rtf_decomposition/run.sh

# Step 1 — frozen-Phase3 BLEU probe over joint-V6 checkpoints
bash experiments/simul_s2st_route_v1/step1_v6_bleu_recheck/run.sh

# Step 2a — upsample ratio and anchor measurement (CPU only)
bash experiments/simul_s2st_route_v1/step2_nar_ctc_head/run_measure_upsample_ratio.sh

# Tests
python experiments/simul_s2st_route_v1/step0_rtf_decomposition/tests/test_instrumentation.py
python experiments/simul_s2st_route_v1/step0_rtf_decomposition/tests/test_qwen_forward_profile.py
python experiments/simul_s2st_route_v1/step1_v6_bleu_recheck/tests/test_loader.py
python experiments/simul_s2st_route_v1/step2_nar_ctc_head/tests/test_measure_upsample_ratio.py
```
