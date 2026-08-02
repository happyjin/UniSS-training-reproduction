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
