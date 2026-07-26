# Simul-UniSS v7 full198 sequence-18000 utilization comparison

This isolated Stage3 experiment repacks the existing full198 prepared action
samples to fixed length 18,000 and trains on eight H200 GPUs with micro batch 2,
global batch 128, matching the completed non-simultaneous Phase3 batch shape.
It uses generated sidecar byte offsets, not the legacy full-file startup scan.
Model weights initialize from the completed non-simultaneous UniSS Phase3
checkpoint, while iteration, optimizer, scheduler, data position, and RNG all
start fresh. No v5/v6 training state is resumed.

The 4096 data, v1-v6 checkpoints, logs, and TensorBoard roots remain untouched.
Only action data is repacked; unused interleaved data is not duplicated.

- Data: `data/megatron/simul_uniss_v7_full198_seq18000_mbs2_gbs128_stage3/`
- Checkpoints: `checkpoints/simul_uniss_v7_full198_seq18000_mbs2_gbs128_stage3/`
- Logs: `logs/simul_uniss_v7_full198_seq18000_mbs2_gbs128_stage3/`
- TensorBoard: port `6021`
