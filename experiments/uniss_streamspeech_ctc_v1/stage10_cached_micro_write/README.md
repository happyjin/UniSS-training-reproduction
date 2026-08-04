# Stage10: CTC-triggered Qwen KV-cache Micro-WRITE

Stage10 consumes Stage09 B1 speech embeddings exactly once. Each source chunk
is appended to the Qwen KV cache as:

```text
START_GLM -> continuous B1 embeddings -> END_GLM
```

Stage09's StreamSpeech CTC policy supplies WAIT or WRITE. WAIT is appended to
the cache without generation. WRITE performs greedy token-by-token generation
from the existing cache, so prior audio and prior output are not re-encoded.
Semantic anti-collapse and repetition controls are retained.

This differs from the original StreamSpeech decoder implementation because
UniSS retains its pretrained Phase3 streaming prompt and BiCodec semantic
tokens. The causal CTC scheduling principle and append-only commitment are the
same.

The smoke intentionally limits the number of Qwen WRITEs while proving that
multiple source chunks and generations extend one monotonic cache.

## Executed bilingual smoke

| Direction | Executed WRITEs | Valid WRITEs | Semantic tokens | Final cache/source B1 | First Qwen token |
|---|---:|---:|---:|---:|---:|
| EN→ZH | 2 | 1 | 121 | 363 / 75 | 24.9 ms |
| ZH→EN | 2 | 1 | 246 | 642 / 133 | 25.6 ms |

The cache path is healthy, but early text is incomplete or repetitive. Stage11
therefore decodes only structurally valid, non-collapsed semantic spans and
retains every rejected WRITE in the timeline for diagnosis.
