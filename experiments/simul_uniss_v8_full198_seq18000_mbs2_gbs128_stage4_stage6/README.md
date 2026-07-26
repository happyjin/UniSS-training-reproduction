# Simul-UniSS v8 full198 sequence-18000 Stage4/Stage6

This isolated experiment continues the completed v7 sequence-18,000 Stage3
checkpoint through the two other Megatron/Qwen stages that support fixed-length
packing:

1. Stage4 phrase-level interleaved S2ST SFT.
2. Stage6 low-learning-rate interleaved refinement.

Stage1/2 are not included because they train the streaming audio/token student
and CTC heads with dynamic PyTorch batches rather than Megatron packed Qwen
sequences. Stage5, Stage7, and Stage8 likewise do not use `--seq-length`.

The prepared full198 samples are read-only. Interleaved training and validation
data are repacked to 18,000 in a new namespace with explicit accounting for
represented and dropped-overlong samples. Stage4 starts only after the v7
Stage3 checkpoint reaches iteration 4753. Stage6 starts only after Stage4 has
completed and passed checkpoint/log verification.

- Sequence length: `18000`
- GPUs: `8 x H200`
- Micro batch: `2`
- Global batch: `128`
- Stage4 epochs: `1.0`
- Stage6 epochs: `0.25`
- Data: `data/megatron/simul_uniss_v8_full198_seq18000_mbs2_gbs128_stage4_stage6/`
- Checkpoints: `checkpoints/simul_uniss_v8_full198_seq18000_mbs2_gbs128_stage4_stage6/`
- Logs: `logs/simul_uniss_v8_full198_seq18000_mbs2_gbs128_stage4_stage6/`
- TensorBoard: port `6022`

Commands:

```bash
experiments/simul_uniss_v8_full198_seq18000_mbs2_gbs128_stage4_stage6/data_preparation/launch_tmux.sh
experiments/simul_uniss_v8_full198_seq18000_mbs2_gbs128_stage4_stage6/orchestration/start_tensorboard.sh
experiments/simul_uniss_v8_full198_seq18000_mbs2_gbs128_stage4_stage6/orchestration/launch_pipeline_tmux.sh
```

The pipeline launcher refuses to start until both the 18k interleaved data and
the final v7 Stage3 checkpoint are complete. Existing outputs are never
overwritten; `--recover-completed` only accepts stages whose final iteration,
eight distributed checkpoint shards, validation log, and zero NaN/skipped
status can all be verified.
