# Stage A V9 8-GPU smoke report

## Outcome

PASS. The isolated V9 Megatron path loaded the immutable Phase3 checkpoint,
built the exact globally shuffled prefix dataset, completed both requested
updates, ran validation, and wrote the final distributed checkpoint.

## Run identity

- Run: `stage_a_v9_bridgefreeze_smoke8_20260817T120800Z`
- Code commit: `e107f43`
- Initial checkpoint:
  `checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4`
- GPUs: 8 H200, data parallel size 8
- Sequence length: 18,000
- Micro/global batch: 1 / 128
- Updates/samples: 2 / 256
- Shuffle seed: 20260816
- Source packs: 16,195
- Complete three-epoch schedule: 48,768 padded samples

## Runtime checks

- Megatron distributed initialization: PASS
- Phase3 model handoff: PASS
- Exact prefix schedule construction: PASS
- Training updates: 2/2
- Skipped updates: 0
- NaN updates: 0
- Final checkpoint save: PASS
- Final validation: PASS

## Final validation snapshot

| Metric | Value |
|---|---:|
| AR-ASR | 9.045088 |
| source CTC | 17.36146 |
| blank argmax ratio | 0.000000 |
| blank posterior | 0.000544 |
| causal-code agreement | 0.154319 |
| teacher-code cosine | 0.956141 |
| adapter RMS | 0.000550 |
| curriculum progress | 1.000000 |

These two updates validate execution only. They are not a quality gate and do
not authorize formal training or Stage B.

## GPU observation

The five-second monitor observed 100% peak GPU utilization and 372.54 W peak
power during the very short run. Its whole-run active-sample averages (44.09%
and 193.67 W) include checkpoint loading, validation, checkpoint saving, and
process teardown, so they are not representative of steady-state canary
training. The 255-update canary is the first meaningful sustained-utilization
measurement.
