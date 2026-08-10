# Phase3 prefix-streaming v3 five-minute bounded-window demo

This directory is isolated from every historical UniSS web demo. It reuses the
audited full198 Phase3 base and selected prefix-streaming V3 `iter_0008000`
adapter without changing either checkpoint or the existing short-audio demo.

## Runtime contract

- accepted duration: 0.5–305 seconds (the extra five seconds tolerate container
  timestamp/codec rounding around a five-minute recording);
- source planning: silence-seeking, non-overlapping 18–30 second windows;
- per-window inference: the unchanged 320/480/640 ms cumulative-prefix engine;
- recovery: a failed source window is recursively bisected to four seconds;
- global state: completed translation text, monotonic target-audio cursor and
  stereo source timeline;
- output: continuous target audio, WAIT-aware target timeline, left-source /
  right-translation stereo and a complete JSON audit.

This is **bounded-window pseudo-streaming**, not a causal encoder-cache claim.
The Gradio upload or recording is complete before server inference starts.

## Public launch

```bash
web_demo/uniss_phase3_prefix_streaming_v3_longform_v1/launch_public_tmux.sh
web_demo/uniss_phase3_prefix_streaming_v3_longform_v1/status.sh
```

Defaults:

```text
tmux = uniss_phase3_prefix_longform_v1_demo
GPU  = 0
port = 7867
auth = public, no login
```

The temporary `https://*.gradio.live` URL changes after a watchdog restart.
Stop only this demo with `stop.sh`.

## Tests

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-offline-demo/bin/python -m unittest -v \
  web_demo.uniss_phase3_prefix_streaming_v3_longform_v1.tests.test_windowing \
  web_demo.uniss_phase3_prefix_streaming_v3_longform_v1.tests.test_engine_fake \
  web_demo.uniss_phase3_prefix_streaming_v3_longform_v1.tests.test_app_contract
```

