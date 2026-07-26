# Simul-UniSS v5 full198 Stage3 utilization run

This isolated Stage3-only experiment uses eight H200 GPUs with micro batch 4,
global batch 128, and four gradient-accumulation micro-steps per optimizer
update. It reuses full198 data and the completed non-simultaneous Phase3 anchor
read-only. The stopped v4 experiment remains untouched on TensorBoard port 6018.

- Checkpoints: `checkpoints/simul_uniss_v5_full198_mbs4_gbs128_stage3/`
- Runs/TensorBoard: `runs/simul_uniss_v5_full198_mbs4_gbs128_stage3/`
- Logs: `logs/simul_uniss_v5_full198_mbs4_gbs128_stage3/`
- TensorBoard: port `6019`

Training, TensorBoard, and memory logging intervals are 10 rather than 1 to
avoid per-iteration host synchronization. Historical configs retain the old
default because the shared launcher only changes behavior when the new config
variables are explicitly set.
