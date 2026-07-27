# Simul-UniSS Stage7A 15-shard validation

This experiment is isolated from all existing Stage3/4/6 assets. It compares:

1. `e0_baselines`: frozen Stage6 action logits and fixed wait-k policies;
2. `e1_continued_sft`: Stage6-initialized binary action head trained with CE;
3. `e2_grpo_g4`: the same head trained with group-relative rewards, group 4;
4. `e3_grpo_g8`: the same head trained with group-relative rewards, group 8.

The action head is initialized from the exact WAIT/WRITE rows of the Stage6 LM
head. The Stage6 Qwen backbone, text generation, semantic generation, and
BiCodec remain frozen. This is an action-policy proof, not full-Qwen token GRPO.

Run unit tests and a short two-GPU smoke before launching all experiments:

```bash
experiments/simul_uniss_stage7a_15shard_v1/orchestration/run_smoke_all.sh
experiments/simul_uniss_stage7a_15shard_v1/orchestration/launch_all_tmux.sh
```

The un-packed action samples are short (median about 387 tokens, p95 about 602),
so fixed `seq_length=13000/18000` is not appropriate. GPU saturation is tuned
with dynamic `MAX_BATCH_TOKENS_PER_GPU` and `MAX_BATCH_SIZE_PER_GPU` instead.
The H200 profile selected 524,288 padded tokens and at most 1,024 samples per
GPU. During steady training this reached 100% utilization, about 631-686 W of
the 700 W power limit, and about 38-40 GiB resident memory per GPU.

The first implementation deliberately trains only the binary action head on
the frozen Stage6 hidden states. Its GRPO reward is the pseudo-alignment action,
timing, final-flush, and structure proxy. It does not update Stage6 text or
semantic generation weights, and it must not be reported as semantic-token or
full-Qwen GRPO. End-to-end free-running generation is a separate evaluation
step after choosing each experiment's best dev checkpoint.

After E1-E3 complete, the post-training watcher exports each best action head
as an untied HF/vLLM model and evaluates the fixed dev/test action schedules:

```bash
tmux new-session -d -s simul_stage7a_post_eval \
  experiments/simul_uniss_stage7a_15shard_v1/post_training/wait_export_and_evaluate.sh
```

The exporter keeps the Stage6 input embeddings unchanged and changes only the
WAIT/WRITE rows of a newly untied output LM head. Consequently the existing
Stage4/6 free-running generator can consume the export without changing its
default code path.

The public Phase3 demo must remain stopped while GPU0 participates in E0 latency
and throughput measurements.

## Full four-way test evaluation

The isolated full free-running test workflow is under `test_evaluation/`. It
assigns exactly two H200 GPUs to each experiment:

```text
GPU 0,1: E0 frozen Stage6
GPU 2,3: E1 continued SFT best
GPU 4,5: E2 GRPO G4 best
GPU 6,7: E3 GRPO G8 best
```

It runs the same 23,369 test schedules through greedy streaming generation,
BiCodec audio decode, Text/Speech BLEU, SLC, UTMOS, AutoPCP, streaming latency,
offline Phase3 comparison, and a 200-sample batch-one latency audit. Run:

```bash
experiments/simul_uniss_stage7a_15shard_v1/test_evaluation/launch_all_tmux.sh
tmux new-session -d -s simul_stage7a_test_compare \
  experiments/simul_uniss_stage7a_15shard_v1/test_evaluation/wait_and_compare.sh
```

The H200 throughput profile uses 1,024 active records/rank, 524,288 batched
tokens, and 94% vLLM memory budget. GPU utilization and power are reported but
never inflated with dummy computation or invalid padding.
