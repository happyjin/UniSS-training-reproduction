# Stage-B-v3 balanced hidden repair

This experiment is isolated from Stage-B-v1/v2 and never rewrites their data,
checkpoints, logs or TensorBoard runs.

## Motivation

Stage-B-v2 recovered only about 29.3% of exact prefix-causal target tokens.
The prefix fine-tuning sidecar had no pre-VQ hidden target, replaced rather
than mixed the broad clone objective, and selected the first contiguous 100k
records instead of a direction-balanced subset.

Stage-B-v3 therefore changes only the remaining representation supervision:

1. deterministically select 50k `eng->cmn` plus 50k `cmn->eng` records;
2. run the released WhisperVQ on exact `160 ms + 80 ms` visible prefixes;
3. export token, real pre-VQ hidden, stability and codebook-neighbour targets;
4. interleave each exact-prefix record with the immutable v2 streaming-clone
   record for the same source sample;
5. initialize from the retained Stage-B-v2 prefix checkpoint and fine-tune;
6. rank checkpoints by direction- and supervision-balanced agreement before
   frozen-Phase3 BLEU selection.

## Paths

Configuration:

```text
configs/experiments/simul_uniss_subsecond_v3/
stage_b_v3_balanced_hidden_15shard_v1.env
```

All new runtime artifacts use `simul_uniss_subsecond_v3` roots under `data`,
`checkpoints`, `runs`, `logs` and `reports`.

## Commands

```bash
bash scripts/simul_uniss_subsecond_v3/prepare_stage_b_v3_data.sh
bash scripts/simul_uniss_subsecond_v3/train_stage_b_v3.sh
bash scripts/simul_uniss_subsecond_v3/start_tensorboard.sh
```

Formal training is blocked until the balanced prefix-hidden sidecar and mixed
manifest markers are complete. The current public listening demo may remain
resident on physical GPU0 because H200 memory is sufficient; GPU monitoring is
recorded separately and any OOM requires stopping only that demo before a clean
restart of this new experiment.
