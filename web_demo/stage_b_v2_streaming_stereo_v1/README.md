# Student-v2 causal streaming stereo demo

This isolated public demo replaces the old cumulative WhisperVQ source
frontend with the Stage-B-v2 prefix-80 causal Student while retaining the
audited Stage7A Reward-v2 R2 WAIT/WRITE controller and Streaming BiCodec.

- Student native input: 160 ms PCM chunks with 80 ms right context.
- R2 policy cadence: 640 ms, unchanged from the previous public demo.
- Upload and microphone modes both export aligned stereo WAV: source on the
  left channel and translated speech on the right at its real WRITE time.
- The page is public and has no login. Runtime artifacts are isolated and
  expire independently from historical evaluation outputs.

The Student frontend passes causality, cache parity and RTF gates but its
target agreement is only about 29.3%. The reported 320/480 ms correct-stable
p50/p95 is conditional on the 31.25% of validation samples whose first stable
token is also correct. This service is an auditable listening prototype, not a
claim that the formal quality gate has passed.

The result panel distinguishes source-timeline NCA from server-side CA
estimates. The historical `first_audio_ms` field is only the point where audio
is placed on the WAIT/WRITE timeline; it excludes autoregressive WRITE and
codec compute and must not be described as wall-clock first playable audio.

The launcher reuses the recovered historical demo environment at
`/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-offline-demo`.  Runtime caches,
temporary files and any Gradio/Hugging Face downloads are explicitly confined
under `/opt/dlami/nvme/jasonleeeli/`; no package installation is performed by
the launcher.

Launch:

```bash
web_demo/stage_b_v2_streaming_stereo_v1/launch_public_tmux.sh
web_demo/stage_b_v2_streaming_stereo_v1/status.sh
```
