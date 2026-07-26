# Simul-UniSS v6 Phase3-index comparison

This isolated eight-H200 Stage3 run continues the complete v5 iteration-500
checkpoint while reproducing the old non-simultaneous Phase3 JSONL indexing
behavior. `phase3-scan` bypasses the 23 MB pre-generated offset sidecar, scans
the 241 GB packed action JSONL at startup in every data-parallel process, and
stores the resulting offsets in Python lists. Steady-state sample reads remain
`seek(offset) + readline`, exactly as in v5.

All model, batch, shuffle, data, and optimization settings remain unchanged:
micro batch 4, global batch 128, eight GPUs, cyclic full-data shuffle, and the
same 22,652-iteration schedule. Optimizer, scheduler, RNG, and iteration state
resume from v5 rather than restarting.

- v5 input checkpoint (read-only):
  `checkpoints/simul_uniss_v5_full198_mbs4_gbs128_stage3/stage03_action_sft/`
- v6 checkpoints:
  `checkpoints/simul_uniss_v6_full198_phase3_index_scan_stage3/`
- v6 logs: `logs/simul_uniss_v6_full198_phase3_index_scan_stage3/`
- v6 TensorBoard: port `6020`
