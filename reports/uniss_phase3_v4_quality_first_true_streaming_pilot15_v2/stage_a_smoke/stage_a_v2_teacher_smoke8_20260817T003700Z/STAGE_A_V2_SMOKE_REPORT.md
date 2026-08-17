# Stage A v2 same-prefix teacher 8-GPU smoke report

## 1. Decision

**Implementation smoke: PASS. Quality/causality gate: NOT YET EVALUATED.**

This run proves that the repaired same-prefix Phase3 teacher cache is consumed
by the native Megatron Stage A objective, contributes a non-zero sparse
full-vocabulary KL term on every training step, saves a distributed checkpoint,
and strictly resumes model/optimizer/scheduler/RNG state. It does **not** prove
that the trained checkpoint preserves Phase3 ASR quality, is future-causal, or
has zero committed-prefix rollback. Stage B therefore remains blocked.

## 2. Immutable inputs and outputs

| Item | Value |
|---|---|
| Initial model | Phase3 v4 native checkpoint, iteration `9075` |
| Phase3 HF teacher | `checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf` |
| Train teacher cache | `data/processed/uniss_phase3_v4_quality_first_true_streaming_pilot15_v2/stage_a_teacher_cache_smoke/teacher_cache_packsource_smoke8_20260817T003100Z/train` |
| Validation teacher cache | same root, `valid` split |
| Megatron checkpoint | `checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v2/stage_a_smoke/stage_a_v2_teacher_smoke8_20260817T003700Z` |
| TensorBoard | `runs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v2/stage_a_smoke/stage_a_v2_teacher_smoke8_20260817T003700Z/tensorboard` |
| Training log | `logs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v2/stage_a_smoke/stage_a_v2_teacher_smoke8_20260817T003700Z/train.log` |
| Strict-resume log | `logs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v2/stage_a_strict_resume/stage_a_v2_strict_resume8_20260817T004800Z/train.log` |

All paths are isolated under the v2 namespace and do not overwrite v1 or any
earlier experiment.

## 3. Same-prefix teacher cache audit

The teacher sees exactly the prefix available at each streaming event. The
event-local BPE delta is aligned against the cumulative Phase3 Quality-ASR
tokenization. Incomparable BPE revisions are excluded. A position is retained
only when its reference token appears in the teacher top-32; the temperature
1.5 posterior is mixed with a 0.5 reference one-hot anchor.

| Split | Packs | Acoustic records | Candidate positions | Retained positions | Retention | Raw teacher top-1 accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Train smoke | 16 | 209 | 3,364 | 3,324 | 98.8109% | 23.6920% |
| Validation smoke | 4 | 4 | 52 | 51 | 98.0769% | 28.8462% |

The low raw top-1 accuracy is why the teacher is not used as a hard target.
The top-32/reference-inclusion filter plus 0.5 reference anchor prevents a
wrong teacher top-1 from directly opposing the original AR-ASR cross entropy.
Real dataset/collator reads additionally verified exact label equality, valid
probability sums, top-k width 32, and real PCM loading.

## 4. Training configuration

| Parameter | Smoke value |
|---|---:|
| GPUs | 8 |
| Sequence length | 4,096 |
| Micro batch | 1 |
| Global batch | 16 |
| Iterations | 32 |
| Train samples consumed | 512 |
| Maximum acoustics per pack | 1 |
| Cache coverage epochs | 32 |
| Checkpoint interval | 5 steps |
| Validation interval | 10 steps |

This deliberately small geometry validates wiring and checkpoint behavior. The
formal configuration remains sequence length 18,000, micro batch 1, global
batch 128, max acoustics 2, three coverage epochs, and 381 iterations.

## 5. Loss trajectory

| Iteration | AR-ASR | Source CTC | Offline teacher KL | Teacher tokens/batch | Grad norm |
|---:|---:|---:|---:|---:|---:|
| 1 | 8.486738 | 19.09538 | 5.838195 | 16.6875 | 243.870 |
| 8 | 4.929974 | 16.16265 | 7.184174 | 17.5000 | 142.916 |
| 16 | 2.997606 | 13.97245 | 5.815674 | 14.7500 | 61.161 |
| 24 | 2.455788 | 12.44303 | 4.836112 | 16.7500 | 81.179 |
| 32 | 2.327945 | 11.66168 | 4.456828 | 16.1875 | 37.124 |

Assertions from the complete 32-step trace:

- `offline_teacher_kl > 0` and `offline_teacher_kl_tokens > 0` on 32/32 steps;
- skipped iterations: 0;
- NaN iterations: 0;
- final distributed checkpoint at iteration 32 saved successfully.

The teacher KL is therefore genuinely active. Its non-monotonic early behavior
is expected in this very short curriculum smoke; the key result here is a
non-zero denominator and stable finite optimization, not convergence.

## 6. Final validation

| Metric | Iteration 32 validation |
|---|---:|
| AR-ASR | 2.375819 |
| Source CTC | 12.81975 |
| Offline teacher KL | 5.041720 |
| Teacher KL tokens/batch | 12.125 |
| Offline ASR replay | 0.3059871 |
| Phase3 replay | 4.183623 |
| Causal GLM agreement | 0.06438752 |
| Bridge residual RMS | 0.07247348 |

These objective values only establish numerical execution. In particular,
`causal_glm_agreement` is still low and is not an ASR quality score. Free-running
content, rollback, and causality must be measured by the external checkpoint
gates before formal training is authorized.

## 7. Strict checkpoint resume

The no-save resume probe used `--dist-ckpt-strictness raise_all` and loaded the
iteration-32 Stage A checkpoint, including optimizer, scheduler, and RNG state.
The resumed validation reproduced the complete iteration-32 metric vector,
including:

- AR-ASR `2.375819`;
- source CTC `12.81975`;
- offline teacher KL `5.041720`;
- teacher KL tokens `12.125`;
- Phase3 replay `4.183623`.

This also validates the v2 resume rule: the pristine Phase3 embedding
fingerprint is required only for initial Phase3-to-Stage-A initialization;
trained Stage A resumes are validated by strict distributed checkpoint keys.

## 8. GPU smoke profile

The 5-second `nvidia-smi` sampler observed all eight GPUs allocating model and
training tensors. Across samples with more than 1 GiB allocated, mean memory
was 27.9 GiB/GPU, mean utilization 17.0%, mean power 142.6 W, and the observed
maxima reached 100% utilization and 218.6 W. Compute-positive samples averaged
33.0 GiB, 27.4% utilization, and 153.2 W.

These averages include initialization, checkpoint saves every five steps,
validation, and teardown. They are also constrained by 4,096-token sequences,
GBS 16, and only 32 steps. They are not a meaningful target for the formal
18k/GBS-128 run and do not indicate that formal GPU tuning has failed.

## 9. Required next gate

Before generating the expensive formal teacher cache or starting the 381-step
formal run, the iteration-32 smoke checkpoint must pass all of the following:

1. trained frontend recomputed-full versus cached-prefix parity;
2. future-PCM perturbation invariance before the changed block;
3. committed-prefix rollback rate exactly zero;
4. matching-sample causal-full and streaming ASR comparison against the exact
   Phase3 anchor (Chinese CER 6.4873%, English WER 8.5038%).

Until those results exist and pass, the correct state is: **Stage A v2 code and
checkpoint mechanics validated; Stage A quality unresolved; Stage B blocked.**
