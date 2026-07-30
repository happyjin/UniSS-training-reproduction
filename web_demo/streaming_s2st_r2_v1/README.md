# UniSS R2 Streaming S2ST public demo

This directory is isolated from `web_demo/offline_s2st_phase3_v1/` and
`web_demo/web_demo.py`. It never writes into training checkpoints, datasets or
historical evaluation outputs.

## Modes

- Upload replay: full input is encoded once, then R2 freely predicts WAIT/WRITE
  over 640 ms-equivalent GLM chunks. This is the closest web path to the audited
  Stage4 evaluation, but it is source-side pseudo-streaming.
- Microphone: cumulative WhisperVQ prefix re-encoding plus stable-prefix commit,
  R2 WAIT/WRITE and streaming BiCodec. This is online pseudo-streaming, not a
  causal Whisper encoder.

Every WRITE passes a semantic anti-collapse processor and a post-generation
quality gate before BiCodec decoding. Repeated-token collapse is never sent to
the browser. If streaming semantic still fails the gate, the request is
completed with the frozen full198 Phase3 Quality model and the UI/JSON records
that fallback explicitly.

Both upload and microphone-completion tabs expose the aligned stereo result as
an in-page audio player, not only as a download. With headphones, the left
channel is the original source and the right channel is the translated speech
placed at its real WAIT/WRITE timeline; silence on the right is the audible
translation delay.

## Launch

```bash
web_demo/streaming_s2st_r2_v1/setup_environment.sh
web_demo/streaming_s2st_r2_v1/launch_public_tmux.sh
web_demo/streaming_s2st_r2_v1/status.sh
```

Defaults:

```text
tmux = uniss_streaming_r2_demo
GPU = physical GPU1
port = 7862
auth = none
public URL = automatically generated https://*.gradio.live
```

The URL is temporary and usually changes after restart. Anyone holding it can
use the service, so concurrency is one, the queue is bounded to four and audio
duration/size are limited.

Stop only this demo with:

```bash
web_demo/streaming_s2st_r2_v1/stop.sh
```

Runtime results stay below `runtime_outputs/` for 24 hours by default. Logs,
URLs and access metadata are gitignored.
