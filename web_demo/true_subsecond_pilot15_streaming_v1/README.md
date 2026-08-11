# Pilot15 iter_0000350 true-input-streaming Gradio

This directory is isolated from every historical UniSS demo and experiment.
It loads the frozen Phase3-v4 HF export plus LoRA/objective tensors exported
from `uniss_true_subsecond_pilot15_epoch1_v3/iter_0000350`.

## Runtime contract

- The model receives only PCM samples whose source clock has elapsed.
- WhisperVQ uses the training cache geometry: 160 ms causal blocks and 80 ms
  bounded right context.
- The frontend keeps at most 4.8 seconds of waveform and emits append-only VQ
  codes. It never edits codes already sent to Qwen.
- Qwen source context uses an append-only KV cache. Each policy/WRITE decision
  is evaluated on a copied branch, so response tokens never corrupt the source
  cache.
- BiCodec uses append-only semantic history with bounded decode left context.
- A 305-second upload is processed by the same state machine. It is not split
  into independent 18–30 second offline windows.

The Gradio upload and ordinary microphone components prepare a file before the
callback starts. The callback then performs strict source-clock replay. This is
true input-causal model inference, but not yet browser WebRTC packet streaming.

Whisper feature normalization is computed from each visible prefix. Re-encoding
an old prefix after more past audio arrives can therefore produce a different
counterfactual VQ code. The runtime audits this as
`committed_revision_violations` but never rewinds a committed code. This is a
known train/runtime mismatch in the pilot checkpoint, not future-audio access.

## Reused environment

No new environment is required. The launcher deliberately reuses the previous
Phase3-v4 Gradio environment:

```text
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-offline-demo
```

## Launch

```bash
web_demo/true_subsecond_pilot15_streaming_v1/launch_public_tmux.sh
web_demo/true_subsecond_pilot15_streaming_v1/status.sh
```

Default local port: `7868`. Default GPU: `0`. The public Gradio URL is written
to `public_url.txt` and `access_info.json`.
