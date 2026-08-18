# UniSS Subsecond E2 Gradio diagnostic

This directory is isolated from the historical offline and Stage7A demos.  It
loads the Stage-B causal student and replays uploaded/microphone audio through
real incremental PCM, causal log-Mel, and cached Emformer inference.

It reports frontend First GLM / fixed-wait-k First WRITE NCA and CA latency,
source-CTC text, active RTF, and per-chunk timing.  It intentionally does not
claim end-to-end translated audio: Qwen micro-WRITE and Streaming BiCodec are
future E4/E5 stages.

```bash
web_demo/subsecond_e2_v1/launch_public_tmux.sh
web_demo/subsecond_e2_v1/status.sh
web_demo/subsecond_e2_v1/stop.sh
```
