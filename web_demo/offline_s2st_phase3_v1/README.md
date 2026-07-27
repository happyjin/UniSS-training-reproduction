# UniSS Phase3 Quality-only Offline S2ST Web Demo

This application is isolated from the historical `web_demo/web_demo.py`, all
training scripts, checkpoints, and evaluation outputs. It loads only:

```text
checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf
```

The UI displays, for every recorded/uploaded sentence:

1. input audio;
2. Phase3's own source transcription;
3. Phase3 target translation;
4. playable/downloadable translated speech.

The fixed mode is `Quality`. Performance and external-ASR fallback are not
exposed because the requested transcription must come from the same UniSS
Quality inference.

## Setup

```bash
web_demo/offline_s2st_phase3_v1/setup_environment.sh
```

The isolated environment is created below the user NVMe root:

```text
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-offline-demo
```

## Public launch

The launcher refuses to take a busy GPU and creates password-protected Gradio
share access:

```bash
UNISS_DEMO_GPU=0 web_demo/offline_s2st_phase3_v1/launch_public_tmux.sh
```

It prints the generated username/password. Wait for model initialization and
then inspect the actual public URL:

```bash
web_demo/offline_s2st_phase3_v1/status.sh
```

The public URL is also written to `public_url.txt`. A Gradio share URL is
temporary and may change after restart. A fixed permanent URL requires a user
domain, DNS, TLS, and a reverse proxy.

Authenticated public smoke test:

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-offline-demo/bin/python \
  web_demo/offline_s2st_phase3_v1/smoke_public.py \
  --url "$(cat web_demo/offline_s2st_phase3_v1/public_url.txt)" \
  --username uniss \
  --password '<password printed by launch_public_tmux.sh>' \
  --audio /path/to/input.wav \
  --direction '中文 → 英文'
```

## Tests

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-offline-demo/bin/python \
  -m unittest discover -s web_demo/offline_s2st_phase3_v1/tests -v
```

Runtime audio and credentials are ignored by Git and expire from the output
store after 24 hours by default.
