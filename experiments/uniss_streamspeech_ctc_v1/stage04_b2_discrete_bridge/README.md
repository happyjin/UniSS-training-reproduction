# Stage04: B2 discrete GLM bridge

The B2 bridge preserves the released Phase3 interface:

```text
40 ms causal encoder hidden
  -> 2:1 pooling (80 ms)
  -> trainable 768 -> 1280 projection
  -> frozen WhisperVQ codebook nearest neighbours
  -> GLM IDs / straight-through Phase3 GLM embeddings
```

Forward computation uses hard nearest-neighbour GLM IDs.  Backpropagation uses a
temperature-controlled distribution over the nearest codebook candidates and
the corresponding frozen Phase3 GLM embedding rows.  Thus the frozen Qwen model
can provide a downstream translation/semantic NLL without changing its token
vocabulary or overwriting the historical Phase3 checkpoint.

The bridge is initialized from the old Stage-B-v3 `glm_latent_head`, but its
selection criterion is Phase3 endpoint NLL/BLEU, not WhisperVQ token agreement.

`train_b2.py` evaluates and saves the untrained step-0 bridge before the first
optimizer update. Validation is rank-sharded across all eight GPUs and Phase3
NLL is weighted by the number of target tokens. This makes `initial.pt` a strict
baseline and prevents a later, degraded projection from replacing `best.pt`.

The formal eight-GPU launcher uses micro-batch 16 per rank (global batch 128).
An isolated two-step H200 smoke reached 100% SM utilization during compute while
remaining below 47 GB per GPU, leaving headroom for longer length buckets.
