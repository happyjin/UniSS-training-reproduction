# Validation record

## CPU/module validation

Run:

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-offline-demo/bin/python \
  -m unittest discover -s web_demo/streaming_s2st_r2_v1/tests -v
```

The tests cover frozen model assets, audio validation/resampling, session
isolation, Stage4 prompt/action/write parsing, timeline rendering and public
access metadata.

## GPU upload smoke

Validated on physical GPU1 with R2 step 300 and the existing source sample
`magicdata_0000000001` (5.46 s, Chinese to English):

```text
translation = I want to search for text messages in Baidu.
translation audio = 5.40 s
server inference after load = 7.41 s
forced actions = 0
structural recoveries = 0
max prompt tokens = 421
CUDA OOM = 0
```

The smoke output is under the new directory's gitignored `runtime_outputs/` and
does not overwrite formal evaluation results.

## Public Gradio upload smoke

The no-login `https://*.gradio.live` endpoint was tested from a fresh
`gradio_client.Client` without credentials. It returned:

```text
HTTP page health = 200
translation = I want to search for text messages in Baidu.
HLS playlist = accessible
timeline WAV = HTTP 200
aligned stereo WAV = HTTP 200
WAIT/WRITE HTML = contains WRITE
auth metadata = public_no_login, username/password null
```

## Microphone-prefix engine smoke

The live engine was exercised with the same 5.46 s waveform as one simulated
browser stream followed by a separate final flush:

```text
prefix updates before final = 8
First WRITE = 4480 ms
First playable audio = 4480 ms
translation = I like it. Search for text messages in Baidu.
forced actions = 1 (final flush)
structural recoveries = 0
continuous/timeline/stereo audio = non-empty
```

The cumulative-prefix result is intentionally different from upload replay and
demonstrates the documented pseudo-streaming quality trade-off. The Gradio
5.49.1 Python client does not understand the browser streaming-input SSE status
`process_streaming`; therefore microphone UI verification must use a real
browser, while the server-side engine and output audio are validated directly.

## Gradio streaming audio dependency

Gradio streaming audio uses both `ffmpeg` and `ffprobe`. Private wrappers in
`bin/` run the already recovered FFmpeg 8 pair with only its recovered Conda
libraries. The Python/PyTorch process does not inherit that library path, and
the implementation installs no system packages or modifies the old demo.

## Semantic collapse/noise regression (2026-07-30)

A real public upload produced intelligible R2 text but a buzzing waveform. The
saved trace proved that this was model semantic collapse rather than a browser
or FFmpeg failure:

```text
source duration = 5.16 s
old target duration = 32.94 s
old semantic count / unique = 1647 / 13
old maximum identical run = 1634
old waveform RMS / ZCR = 0.0053 / 0.647
```

The fix adds a semantic-only logits processor, blocks a token after six
identical BiCodec IDs, guards low-diversity sliding windows and rejects any
remaining collapse before waveform decoding. The same exact saved source then
produced:

```text
target duration = 6.40 s
semantic count = 320
per-WRITE unique ratio = 0.875, 1.000, 0.992
maximum identical run = 6
waveform RMS / peak / ZCR = 0.0734 / 0.7012 / 0.209
fallback used = false
Whisper-large-v3 ASR = Xi Shashuo said I enjoy being single, but if I encounter
someone suitable for me, I would also consider it a day.
```

The separately tested Phase3 safety fallback produced 271 semantic tokens,
maximum run 3, unique ratio 0.970 and waveform RMS 0.0727. It remains lazy-loaded
and is used only if the R2 quality gate rejects a WRITE or no safe streaming
audio is available.
