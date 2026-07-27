# Simul-UniSS Stage6 end-to-end streaming evaluation v1

This isolated experiment evaluates the completed full198 Stage6 iteration 1189
checkpoint with the same frozen protocol used by Stage4:

```text
chunk_ms=640, wait_k=2, max_phrase_tokens=16
greedy deterministic decoding, repetition_penalty=1.1
training context boundary=18000, native inference context=32768
streaming BiCodec left_context=50, holdback=5, overlap=80ms
```

The Stage6 HF export, output root, reports, logs, tmux sessions, and audio files
are all separate from Stage3, Stage4, and offline Phase2/Phase3 artifacts.  A
runner refuses to reuse an existing output directory unless `RESUME=1` is set.

The requested GPU handoff is:

- UniST test: GPU 0-3, start immediately.
- UniST dev: GPU 4-7, start only after the active Stage4 test writes its top-level
  `COMPLETE` marker and all four GPUs pass two consecutive idle checks.

Launch both the immediate test and queued dev jobs:

```bash
experiments/evaluation/simul_uniss_stage6_streaming_v1/launch_test_now_dev_when_free_tmux.sh
```

The full runs compute streaming action/latency/continuity metrics, Text-BLEU,
Speech-BLEU, SLC, UTMOS, AutoPCP, offline Phase3 deltas, GPU utilization/power,
and a separate 200-record batch-one latency audit.
