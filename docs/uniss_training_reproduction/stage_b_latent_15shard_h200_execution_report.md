# Corrected Stage B Latent 15-Shard H200 Execution Report

## 1. Scope

This run is isolated from the historical CTC Stage B. It uses the completed
formal 15-shard Stage A manifests and writes to new `stage_b_latent_*` paths.
No historical checkpoint, script, TensorBoard directory, or result is
overwritten.

## 2. Corrected method

The historical 16,385-way GLM CTC head is replaced by:

```text
40 ms causal Emformer hidden
  -> fixed 2:1 pooling to 80 ms WhisperVQ rate
  -> 1,280-D latent regression
  -> frozen WhisperVQ codebook nearest-neighbor quantization
  -> original 16,384 GLM token space
```

The training loss is:

```text
1.0 * codebook latent L2
+ 0.5 * codebook-space cosine distillation
+ 0.3 * source CTC
+ 0.4 * target capacity
+ 0.2 * stability
+ 0.1 * chunk consistency
```

Repeated GLM tokens are retained and never passed through CTC collapse.

## 3. Stage A input

Formal Stage A completed after correcting its assembly invariant:

- A4/A5 input records: `1,500,000`
- A4/A5-passing records entering A6/A8: `1,496,943`
- A6/A8 formal accepted records: `1,338,712`
- deterministic train/valid manifests and offset indexes: complete

## 4. Verification completed

- unit tests for fixed-rate pooling, repeated-token retention, codebook
  quantization, all loss-head gradients, scripts, and Stage A assembly: pass;
- real 128-record one-GPU launcher smoke: pass;
- eight-GPU DDP/NCCL two-step smoke: pass;
- cache parity maximum absolute error: about `7.15e-7`;
- future perturbation maximum absolute error: `0`;
- smoke first-stable GLM: `320 ms`;
- smoke active RTF: about `0.019`;
- checkpoint, validation JSON, TensorBoard events, and GPU monitor: generated.

The two-step smoke has zero agreement by design and is used only as a
structural test. Formal quality is evaluated after training.

## 5. H200 throughput scan

All scans used the formal 768-hidden, 16-layer, 12-head, 3,072-FFN model on
eight H200 GPUs with BF16 and the same corrected losses.

| Per-rank batch | Global batch | Peak allocated / reserved | Typical SM utility | Typical power | Decision |
|---:|---:|---:|---:|---:|---|
| 64 | 512 | about 40 GiB allocated | high | lower than batch 128 | safe baseline |
| 128 | 1,024 | about 78 GiB allocated | 92%--100% | 490--549 W | selected |
| 192 | 1,536 | about 116 GiB allocated / up to 136 GiB reserved | 96%--100% | 510--568 W | rejected: insufficient OOM margin |

Increasing from 128 to 192 improved normal-step audio throughput by only about
3%--6%, while reducing free HBM to roughly 5 GiB on the fullest rank.
`batch=128` is therefore the highest safe long-run configuration.

The H200 power limit is 700 W, but this Emformer workload does not reach 700 W
even when reported SM utility is 100%. Artificial duplicate computation or an
unsafe batch is not used merely to raise the power reading.

## 6. Formal run configuration

```text
GPUs                    = 8
per-rank batch          = 128
global batch            = 1024
workers per rank        = 8
OMP/MKL/OpenBLAS threads= 4
max audio               = 8 seconds
steps                    = 50,000
learning rate            = 1e-4
precision                = BF16
consistency interval     = 4
master address           = 127.0.0.1
master port              = 29743
TensorBoard port         = 6057
```

The formal quality continuation gate is agreement `>= 0.70`; the final target
remains `>= 0.90`. Failure of the continuation gate prevents automatic Stage C
startup.

## 7. Formal result and gate decision

The eight-GPU run completed all `50,000` optimizer steps.  At global batch
`1,024`, this corresponds to about `38.63` passes over the `1,325,243`-record
training split, so the failed quality gate cannot reasonably be attributed to
too few optimizer steps.

The selected checkpoint is `best.pt` at step `49,000`:

| Metric | Result | Gate | Decision |
|---|---:|---:|---|
| position token agreement | `0.1931` | diagnostic | low |
| edit token agreement | `0.1607` | `>= 0.70` | fail |
| active RTF | `0.0987` | `<= 0.25` | pass |
| first stable GLM p50 / p95 | `480 / 480 ms` | `<= 700 / 1000 ms` | provisional pass |
| chunk polling invariance | `0.9887` | `>= 0.95` | pass |
| cache maximum absolute error | `3.81e-6` | `<= 1e-4` | pass |
| future perturbation maximum | `0` | `<= 1e-5` | pass |
| one-minute long-session active RTF | `0.0973` | bounded and `< 1` | pass |

`structural_pass=true` and `quality_pass=false`.  Stage C must therefore remain
blocked until a corrected Stage B passes the representation gate.

The training did learn a useful acoustic representation: validation latent L2
fell from `0.07434` at step 500 to `0.01404` at step 49,000, while the cosine
term fell from `0.08652` to `0.01501`.  Token agreement nevertheless plateaued
near `0.193` after roughly step 45,000.  This is convergence to the wrong
decision geometry, not divergence or an unfinished warmup.

## 8. Post-run diagnosis

### 8.1 Quantization geometry

A frozen-codebook audit found that the median squared distance per dimension
from a code to its nearest different code is about `0.01583`; the rough
midpoint decision-boundary MSE is only about `0.00396`.  The student's
validation latent MSE (`~0.014`) is therefore small as a regression loss but
still large enough to cross many Voronoi boundaries.

On an evenly distributed 32-record validation diagnostic:

| Teacher-code rank metric | Result |
|---|---:|
| top-1 | `0.1725` |
| top-5 | `0.4454` |
| top-10 | `0.5707` |
| top-100 | `0.8417` |
| median teacher-code rank | `7` |

The student often reaches the correct codebook neighbourhood but does not make
the correct code the nearest neighbour.  Plain L2 and cosine loss do not
directly optimize that boundary.

### 8.2 Full-context teacher versus causal student

The frozen WhisperVQ configuration has both
`encoder_causal_attention=false` and `quantize_causal_encoder=false`.  Its
final token for a time position can depend on substantially more future audio
than the student's 80 ms right context.  Requiring 90% exact agreement is only
valid if a prefix-teacher ceiling experiment first proves that 80 ms contains
enough information.

### 8.3 The current hidden loss is not teacher-hidden distillation

The current implementation reconstructs the target vector by embedding the
teacher token ID in the frozen VQ codebook.  Stage A does not cache the actual
pre-quantization WhisperVQ hidden state or its top-k code distances.  The loss
called `hidden_distill` is consequently another codebook-vector regression
term, not intermediate teacher representation distillation.

### 8.4 Stability supervision and latency accounting

The current stability target marks all tokens except the final 320 ms as
stable.  It does not verify persistence across `t`, `t+160 ms`, and
`t+320 ms` teacher prefixes.  In addition, validation appends right-context
audio before calling an inference API that does not separately mark the
lookahead-only region.  The reported 480 ms first-stable latency is therefore
provisional until output lengths exclude right-context-only frames and token
correctness is included in the commit criterion.

### 8.5 Reconstructed-audio domain boundary

UniST supplies BiCodec tokens rather than original waveforms.  Across the 30
formal Stage-A parts, the mean exact/edit agreement between released UniST GLM
tokens and frozen-WhisperVQ tokens re-encoded from reconstructed source audio
is only `0.40476`.  Current Stage B correctly trains against the re-encoded
teacher on the same waveform, so this is not the direct cause of its 0.193
score.  It is, however, a downstream Phase3 compatibility risk that must be
measured with frozen-Phase3 Text-BLEU and COMET before accepting the frontend.

### 8.6 Minor geometry findings

A 10,000-record audit confirms the nominal rate is correct (`12.575` teacher
tokens/s), adjacent repeats are rare (`2.55%`), and a constant one-position
shift does not explain the errors.  Student output is one token shorter on
about `19%` of audited utterances, confined to the final boundary; this should
be repaired but cannot explain the full quality gap.  The student frontend
also emits a torchaudio warning that at least one of 128 mel filters is empty
for the current 400-point FFT, so the causal mel implementation must be made
explicit and tested against the teacher feature geometry.

## 9. Ordered repair plan

### R0: make the measurements honest before retraining

1. Separate committed audio duration from right-context-only samples in the
   streaming inference API.
2. Report both self-stable and teacher-correct stable tokens; never count a
   right-context-only output token as committed.
3. Run frozen WhisperVQ prefix-ceiling tests at 80, 160, 320, and 640 ms
   lookahead, recording exact/edit agreement, top-k agreement, revision rate,
   and first correct stable latency.
4. Run the current student and both reconstructed/full teacher token streams
   through frozen Phase3 on a development subset to measure Text-BLEU/COMET
   sensitivity.

Decision rule:

| Prefix-teacher ceiling | Action |
|---|---|
| 80 ms agreement `>= 0.80` | retain 80 ms; repair the loss |
| 80 ms in `0.50--0.80` | sweep 160/320 ms while retaining a subsecond end-to-end target |
| 320 ms `< 0.70` | stop exact full-context imitation; use a streaming-WhisperVQ clone or a new causal target |

### R1: isolated Stage-A-v3 supervision

Do not rewrite the completed Stage A.  Build a versioned sidecar dataset that
caches the actual WhisperVQ pre-VQ hidden state, teacher token, top-32/64 hard
negative code IDs and distances, exact 80 ms frame boundaries, and persistence
labels derived from multiple prefixes.  Use batched GPU teacher inference plus
parallel CPU audio loading/index writing.  Store large hidden arrays in sharded
BF16 containers rather than embedding them in JSONL.

### R2: quantization-aware Stage-B-v2

Use an isolated checkpoint/run directory and train:

```text
1.0 * SmoothL1(student projected hidden, teacher pre-VQ hidden)
+ 0.5 * cosine hidden distillation
+ 1.0 * codebook-distance cross entropy
+ 0.5 * nearest-wrong-code margin
+ 0.1 * source CTC
+ 0.1 * target capacity
+ 0.1 * true stability
+ 0.05 * chunk consistency
```

The first 5k--10k steps train representation, codebook CE, and margin only;
auxiliary heads are then ramped in.  The existing checkpoint may initialize a
short low-LR diagnostic arm, but the final comparison must include a fresh
run with a reset optimizer.

### R3: minimal 15-shard experiment sequence

| Experiment | Purpose |
|---|---|
| `B2-E0` | current checkpoint under corrected validation |
| `B2-E1` | current data plus codebook CE/margin |
| `B2-E2` | true pre-VQ hidden plus hard-negative supervision |
| `B2-E3` | selected 80/160/320 ms lookahead after the ceiling audit |

Only the best diagnostic proceeds to a fresh eight-GPU 15-shard formal run.
Full198 expansion remains blocked until the 15-shard representation and
frozen-Phase3 downstream gates pass.

### R4: corrected acceptance gates

- causal recovery = student agreement / same-lookahead prefix-teacher ceiling;
- causal recovery `>= 0.90` and absolute edit agreement `>= 0.70`;
- frozen-Phase3 Text-BLEU drop `<= 2` and COMET drop `<= 0.03`;
- correctness-aware first-stable p50 `<= 400--600 ms`, p95 `<= 720--900 ms`;
- committed rollback `= 0`, cache parity `>= 99.9%`, active RTF p95 `< 0.25`.

Training longer, increasing the H200 power number, moving directly to
full198, or lowering the learning rate on the unchanged objective are not
accepted fixes because the v1 validation curve is already converged.

## 10. R0 causal-teacher ceiling result

The new prefix audit was run on 16 validation records sampled uniformly from
the indexed formal validation manifest.  For every 160 ms commit boundary, the
frozen WhisperVQ teacher was re-run with only the selected future lookahead.
Each newly committable token was compared once against the token produced by
the same frozen teacher with the complete utterance.  Full-waveform re-encoding
matched the cached Stage-A teacher exactly (`1.0` position and edit agreement),
which validates the audit implementation and cached labels.

| Lookahead | Immediate full-teacher agreement | Prefix edit agreement | Revision after 320 ms | Revision vs full | First correct-stable visible p50 / p95 |
|---:|---:|---:|---:|---:|---:|
| 80 ms | `0.2632` | `0.8061` | `0.6228` | `0.7368` | `2880 / 4800 ms` |
| 160 ms | `0.3933` | `0.8262` | `0.4791` | `0.6067` | `2960 / 4760 ms` |
| 320 ms | `0.5465` | `0.8542` | `0.3170` | `0.4535` | `2960 / 4760 ms` |
| 640 ms | `0.6814` | `0.8922` | `0.1830` | `0.3186` | `2960 / 4760 ms` |

The result is decisive: even 640 ms future context does not reach the 0.70
continuation gate, and a full-teacher-correct token does not become stable
until roughly 2.9 seconds at the median.  It is therefore mathematically
inconsistent to require an 80 ms-lookahead student to reproduce 90% of these
full-context token IDs while also claiming subsecond latency.

### 10.1 Route decision

The original `B2-E1` idea (add only margin/CE while retaining full-context
token labels) remains useful as a quantization ablation, but it cannot be the
main repair.  The ordered plan now takes the `<0.70 at 320 ms` branch:

1. preserve the completed latent v1 as the full-context-imitation baseline;
2. measure frozen Phase3 sensitivity to released, reconstructed-full, and
   prefix-causal GLM streams;
3. construct Stage-A-v3 sidecars containing true pre-VQ hidden states and
   prefix-causal targets at 80/160/320/640 ms;
4. train a chunk-causal WhisperVQ clone as the compatibility reference and a
   smaller quantization-aware Emformer student against the selected causal
   target;
5. select by downstream Phase3 quality and correctness-aware latency, not by
   unattainable full-context exact match alone.

For a target with lookahead `L`, report both:

```text
causal-target recovery = student agreement / causal-teacher self agreement
full-teacher recovery   = student agreement / measured full-teacher ceiling(L)
```

The first should exceed `0.90`.  The second cannot legitimately be required to
exceed the ceiling in the table.  Frozen-Phase3 Text-BLEU/COMET and real audio
quality become the final compatibility gates.

The machine-readable audit is stored in
`reports/simul_uniss_subsecond_v2/stage_b_teacher_prefix_ceiling_v1.json`.
