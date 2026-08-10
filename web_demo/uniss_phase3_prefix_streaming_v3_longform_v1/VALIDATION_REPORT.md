# Phase3 Prefix-Streaming V3 Five-Minute Bounded-Window Validation

Validation date: 2026-08-10 UTC

## Scope

This report validates that the isolated long-form demo can accept and finish a
real 300-second source recording without changing the historical short-form
demo, checkpoints, training code or evaluation outputs.

The validated model is:

```text
Phase3 full198 base: iter_0009075
Prefix-streaming V3 LoRA: iter_0008000
Direction: zh-en
Chunk update: 480 ms
```

This remains bounded-window pseudo-streaming. It is not a causal encoder-cache
implementation, and the current Gradio upload is available in full before the
server simulates incremental visibility.

## Input

```text
/opt/dlami/nvme/jasonleeeli/data/
  uniss_phase3_prefix_longform_v1_validation/
  chinese_singapore_vietnam_relations_300s.wav
```

Properties:

```text
duration: 300.000 seconds
sample rate: 16,000 Hz
channels: mono
subtype: PCM-16
```

## Validation history

### V1: non-empty output gate

The first 300-second run completed all 12 planned windows in 177.2 seconds
(`RTF=0.591`). File-level checks passed, but three windows produced only about
one second of speech for long target text. The underlying short-form events
showed semantic rejection due to low token diversity or a 25-token identical
run. Two additional windows reached the 160 target-text-token generation cap.

This proved that “non-empty audio” was not a sufficient long-form success gate.

### V2: strict text and semantic coverage gate

The production path now rejects and bisects a source window when either:

```text
committed_text_tokens >= 160
semantic_tokens / committed_text_tokens < 1.5
```

Recursive retry stops at the configured four-second minimum. Retry reasons and
depth are preserved in the final JSON.

## Final strict result

```text
result:
web_demo/uniss_phase3_prefix_streaming_v3_longform_v1/
  runtime_outputs/validation_zh300_480_v2/chunk_480ms/20260810/
  aab60e49e4204be987628a18cf9e2aba/result.json
```

| Metric | Result |
| --- | ---: |
| Source duration | 300.000 s |
| Planned bounded windows | 12 |
| Final successful windows | 18 |
| Failed windows | **0** |
| Successful retry descendants | 11 |
| Maximum final window | 27.940 s |
| Maximum target text tokens/window | 113 |
| Minimum semantic/text coverage | 1.887 |
| Translation text characters | 5,074 |
| Continuous target audio | 140.980 s |
| Global target timeline | 302.930 s |
| First target audio, source time | 25.720 s |
| Processing time | 309.719 s |
| Compute RTF | 1.032 |

Window-depth distribution:

```text
depth 0: 7 final windows
depth 1: 9 final windows
depth 2: 2 final windows
```

No OOM, NaN, empty final audio or unrecovered window failure occurred. GPU
memory stayed near 6.6 GiB throughout the run rather than growing with source
duration.

## Artifact assertions

All assertions below passed:

```text
source duration == 300.0 seconds
planned windows == 12
completed final windows == 18
failed windows == 0
maximum source window <= 30 seconds
maximum committed text tokens < 160
minimum semantic/text coverage >= 1.5
all output samples finite
translation waveform non-empty
timeline waveform non-empty
stereo left channel non-empty
stereo right channel non-empty
sample rate == 16 kHz
```

Artifacts:

```text
translation_continuous.wav                    4.4 MiB
translation_global_timeline.wav               9.3 MiB
stereo_left_source_right_translation.wav       19 MiB
source_16k.wav                                 9.2 MiB
result.json                                     40 KiB
```

## Interpretation and claim boundary

The five-minute engineering objective is satisfied: a real 300-second upload
finishes with bounded source windows, bounded GPU memory, complete auditable
artifacts and automatic recovery from target text/semantic truncation.

Latency is not yet competitive. Most source windows only emitted target speech
at their final flush, so the first target audio occurred after 25.72 seconds.
The quality gate also raised compute RTF from 0.591 to 1.032. Therefore the
validated claim is:

> Five-minute bounded-window pseudo-streaming inference is stable and complete
> under the implemented structural gates.

The following claim is not supported:

> Low-latency causal or real-time simultaneous S2ST has been achieved.

Reducing the 25-second first-audio delay requires a less conservative policy or
a causal cached source encoder plus an append-only cached target protocol; it
cannot be obtained merely by changing the upload duration limit.

## Gradio transport validation

The isolated Gradio service was launched on GPU 1 and port 7867. The public
page returned HTTP 200. A short API smoke returned all eight declared outputs:

```text
translation text
continuous target WAV
global target timeline WAV
stereo WAV
result JSON
status Markdown
100% progress
window audit table
```

The exact 300-second WAV was then submitted through the running Gradio
`/phase3_bounded_longform` endpoint, rather than invoking the engine directly.
The HTTP request completed successfully:

| Metric | Gradio result |
| --- | ---: |
| Outputs returned | 8 / 8 |
| Source duration | 300.000 s |
| Planned / final windows | 12 / 18 |
| Failed windows | **0** |
| Minimum semantic/text coverage | 1.887 |
| Maximum target text tokens/window | 113 |
| Continuous target audio | 140.980 s |
| Processing time | 306.391 s |
| Compute RTF | 1.021 |
| Progress | 100% |

Server-side Gradio result:

```text
web_demo/uniss_phase3_prefix_streaming_v3_longform_v1/
  runtime_outputs/chunk_480ms/20260810/
  6d78ebfbfdd24f8383d6a4d230629f04/result.json
```

This second run verifies the browser/server transport boundary, Gradio queue,
five-minute upload normalization, generator output contract and downloadable
artifact handoff in addition to the model engine itself.
