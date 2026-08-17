# Stage A v2 formal training failure report

## Decision

**This run is retained as a failed formal run and must not authorize Stage B.**

The job was intentionally stopped only after the iteration-100 distributed
checkpoint had been saved successfully.  The process itself was numerically
stable, but the validation CTC decoder had collapsed to all blank output and
therefore violated an explicit Stage A content gate.

## Immutable run identity

| Item | Value |
|---|---|
| Run ID | `stage_a_v2_teacher_formal8_20260817T012204Z` |
| Initial checkpoint | Phase3 v4 native iteration `9075` |
| GPUs | 8 x NVIDIA H200 |
| Sequence length | `18000` |
| Micro/global batch | `1 / 128` |
| Planned iterations | `381` |
| Completed checkpoint | `100` |
| Coverage schedule | three globally shuffled coverage epochs |
| Train packs | `16195` |
| Validation source rows | `167` |
| Teacher-cache train records | `94587` |
| Teacher-cache validation records | `334` |

Checkpoint root:

`checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v2/stage_a_formal/stage_a_v2_teacher_formal8_20260817T012204Z`

Training log:

`logs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v2/stage_a_formal/stage_a_v2_teacher_formal8_20260817T012204Z/train.log`

TensorBoard event directory:

`runs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v2/stage_a_formal/stage_a_v2_teacher_formal8_20260817T012204Z/tensorboard`

## Validation evidence

| Metric | Iteration 50 | Iteration 100 | Interpretation |
|---|---:|---:|---|
| AR-ASR | 2.217223 | 0.695638 | Conditional training objective improved |
| Source CTC | 5.556608 | 3.702559 | CTC scalar alone was misleading |
| Offline teacher KL | 7.599936 | 1.636372 | Same-prefix teacher supervision was active |
| Offline ASR replay | 0.280825 | 0.266169 | Offline ASR replay remained finite |
| Phase3 replay | 3.976165 | 3.904774 | Phase3 replay remained finite |
| CTC blank ratio | 0.910639 | **1.000000** | Hard failure: every validation frame selected blank |
| Causal GLM agreement | 0.138446 | **0.003432** | Pretrained code identity was almost entirely lost |
| Bridge residual RMS | 0.168211 | 0.151256 | Residual bridge remained bounded |
| Skipped / NaN iterations | 0 / 0 | 0 / 0 | No numerical failure |

The decisive observation is that lower CTC loss did not imply usable CTC
decoding.  By iteration 70, training-frame blank argmax had reached exactly
`1.0`; validation reached exactly `1.0` at iteration 100.  The final Stage A
gate requires zero all-blank rows, so continuing this configuration to 381
iterations would not be an acceptable success criterion.

## Root-cause diagnosis

1. The byte CTC head was randomly initialized and optimized only by sequence
   CTC.  On the much larger formal batches it found the dominant blank path
   before learning a stable byte-to-frame alignment.
2. The metric used for the hard gate is frame argmax blank rate, while the
   scalar CTC objective marginalizes over paths.  It is therefore possible for
   the scalar to fall while greedy decoding becomes all blank.
3. The top WhisperVQ layers were unfrozen after five percent of training.
   Their nearest-code identity subsequently fell from roughly 26 percent to
   below one percent because no differentiable codebook commitment term
   directly preserved the released source-GLM identity.
4. The 32-step implementation smoke did not expose this dynamic: it used
   sequence length 4096, global batch 16, one acoustic per pack, and a rapidly
   traversed curriculum.  Its checkpoint CTC blank ratio was only 0.0383.

## Authorized repair

The failed checkpoint is evidence only and is not a resume source.  The repair
must start a new isolated run from the original Phase3 iteration 9075 and add:

1. a negative initial blank bias for the fresh CTC head;
2. a short-lived monotonic byte-to-frame seed loss so the CTC head sees
   explicit non-blank locations before pure path marginalization dominates;
3. a differentiable per-utterance blank-posterior budget;
4. a codebook commitment loss against the immutable released source-GLM code;
5. delayed Whisper top/bottom/conv unfreezing;
6. a short formal-geometry canary whose validation blank rate must remain below
   the all-blank threshold before the 381-step replacement run is allowed.

The replacement run must use a new experiment namespace and must preserve
this failed run and both iteration-50 and iteration-100 checkpoints.

