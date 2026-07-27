# Simul-UniSS Stage4 end-to-end streaming evaluation v1

This experiment is isolated from the historical offline Phase2/Phase3 and
Stage3 action-only evaluations. It performs free-running Stage4 generation on
the full 7,965-record UniST dev split using GPU 0–3, then decodes real streaming
BiCodec waveforms and computes both streaming-specific and offline-compatible
quality metrics.

The main operating point is the Stage4 training point:

```text
chunk_ms=640, wait_k=2 pseudo boundary distribution
learned Stage4 WAIT/WRITE policy
greedy deterministic decoding
training context boundary=18000; native inference context=32768; no truncation
BiCodec left_context=50, holdback=5, overlap=80ms
```

The throughput runner intentionally keeps at most 512 active records per GPU.
A measured 1,024-record/GPU trial was slower and did not increase H200
utilization because this 0.5B model is dominated by short per-chunk action
decodes and host scheduling.  GPU power is therefore reported as a measured
result, never increased with duplicated or otherwise invalid computation.

Every sample records its maximum realized prompt length and whether free
running generation crossed the 18,000-token training boundary.  The 32,768
native Qwen context is only a crash-prevention envelope; crossing 18,000 remains
an explicit evaluation warning/failure characteristic rather than being hidden.

Run a new full evaluation:

```bash
experiments/evaluation/simul_uniss_stage4_streaming_v1/run_full_dev_4gpu.sh
```

Every invocation creates a new directory below:

```text
eval_outputs/simul_uniss_stage4_streaming_v1/
```

The full run reports action/prefix/token latency proxies, computation-aware
request and codec latency, playback gaps, boundary discontinuity, Text-BLEU,
Speech-BLEU, SLC, UTMOS, and AutoPCP. Source boundaries remain pseudo-aligned;
the report never labels them as real word-timestamp AL/LAAL/ATD.

The full runner also creates `latency_batch1/` for a separate 200-record,
four-GPU batch=1 latency audit. Those request/TTFT/ACT values are the primary
computation-aware latency results; the full-dev run remains the primary corpus
quality and throughput result.
