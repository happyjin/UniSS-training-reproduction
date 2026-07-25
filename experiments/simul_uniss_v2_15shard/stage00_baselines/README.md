# Stage 0 — reconstruction and prefix baseline

Stage 0 does not train a model, so it may use one GPU while the training stages
remain eight-GPU jobs.

- `prepare_audio.sh`: reconstructs an isolated 1000-record source/target audio
  manifest for the audio student and BiCodec refinement.
- `run_prefix_baseline.sh`: evaluates prefix re-encoding on the first 100
  reconstructed records without modifying the original v1 outputs.

