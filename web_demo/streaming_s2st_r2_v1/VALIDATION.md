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
