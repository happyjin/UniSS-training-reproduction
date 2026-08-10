# Phase3 prefix-streaming v3 stereo public demo

This demo is isolated from every historical offline/streaming Gradio directory.
It loads the selected `iter_0008000` LoRA adapter over the frozen full198 Phase3
HF base and exposes 320/480/640 ms cumulative-prefix inference.

The downloadable stereo artifact always uses:

```text
left channel  = original source waveform
right channel = translated waveform placed on the measured WAIT/WRITE timeline
```

Launch without login/authentication:

```bash
web_demo/uniss_phase3_prefix_streaming_v3_stereo_v1/launch_public_tmux.sh
web_demo/uniss_phase3_prefix_streaming_v3_stereo_v1/status.sh
```

The generated `https://*.gradio.live` URL is temporary and changes after a
restart.  Stop only this service with `stop.sh`.

