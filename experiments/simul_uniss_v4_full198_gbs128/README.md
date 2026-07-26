# Simul-UniSS v4 full198 — micro batch 2 / global batch 128

This is an isolated restart of the full198 streaming Qwen stages. It reuses the
completed v3 full198 packed data and the non-simultaneous Phase3 iteration-9075
checkpoint read-only. It does not load or overwrite the stopped v3 Stage 3 run.

## Isolated outputs

- Checkpoints: `checkpoints/simul_uniss_v4_full198_gbs128/`
- Runs/TensorBoard: `runs/simul_uniss_v4_full198_gbs128/`
- Logs: `logs/simul_uniss_v4_full198_gbs128/`
- TensorBoard: port `6018`

The effective batch geometry is eight data-parallel ranks, micro batch 2, global
batch 128, hence eight gradient-accumulation micro-steps per optimizer update.
The generated schedule preserves one packed epoch for Stage 3/4 and one quarter
epoch for Stage 6.

```bash
experiments/simul_uniss_v4_full198_gbs128/data_preparation/generate_schedule.sh
experiments/simul_uniss_v4_full198_gbs128/orchestration/run_shuffle_smoke_8gpu.sh
experiments/simul_uniss_v4_full198_gbs128/orchestration/start_tensorboard.sh
experiments/simul_uniss_v4_full198_gbs128/orchestration/launch_qwen_pipeline_tmux.sh
```

The old v3 TensorBoard remains on port 6017. This experiment uses port 6018.
