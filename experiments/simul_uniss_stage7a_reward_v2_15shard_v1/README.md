# Simul-UniSS Stage7A Reward-v2 15-shard experiments

This namespace is isolated from the completed Stage7A v1 E0–E3 runs. It uses
the same 15-shard action data and the same H200 dynamic batch profile while
changing only explicitly selected Reward-v2 behavior.

GPU allocation:

```text
GPU 0,1: R0 E3-v1 WRITE-logit-bias full-dev action sweep
GPU 2,3: R1 rebalanced action costs + coverage/final-flush
GPU 4,5: R2 R1 + explicit first-WRITE/mean-write/write-area latency deltas
GPU 6,7: R3 R2 + direction-balanced replay and adaptive KL
```

All training runs use group size 8, 1,000 steps, the original 15-shard
bootstrap, 524,288 dynamic padded tokens/GPU, and at most 1,024 samples/GPU.
The old Stage7A scripts continue to default to Reward v1.

After R0 selects a dev WRITE bias and R1–R3 finish training/export, every
session automatically continues into an isolated 7,965-sample free-running dev
evaluation on the same two GPUs. The evaluation includes BiCodec decode,
Text/Speech BLEU, SLC, UTMOS, AutoPCP, streaming latency, and GPU monitoring.

Run smoke tests before the formal launch:

```bash
experiments/simul_uniss_stage7a_reward_v2_15shard_v1/common/run_train_2gpu.sh r1 --smoke
experiments/simul_uniss_stage7a_reward_v2_15shard_v1/common/run_train_2gpu.sh r2 --smoke
experiments/simul_uniss_stage7a_reward_v2_15shard_v1/common/run_train_2gpu.sh r3 --smoke
```

Launch the four isolated experiments:

```bash
experiments/simul_uniss_stage7a_reward_v2_15shard_v1/tensorboard/start.sh
experiments/simul_uniss_stage7a_reward_v2_15shard_v1/orchestration/launch_all_tmux.sh
```

TensorBoard defaults to `http://127.0.0.1:6040` and can be viewed remotely
through an SSH tunnel. No script reuses a completed v1 output directory.

After all four full-dev evaluations finish, the comparison watcher writes:

```text
eval_outputs/simul_uniss_stage7a_reward_v2_15shard_v1/full_dev_e2e_v1/
├── comparison.json
└── reward_v2_four_way_full_dev_report.md
```
