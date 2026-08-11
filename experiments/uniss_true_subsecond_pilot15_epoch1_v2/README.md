# UniSS true-subsecond pilot15 epoch1 v2

This directory is an isolated repair of the failed v1 50-step pilot. It does
not overwrite historical Phase1/2/3, full198, or pilot15-v1 artifacts.

The frozen raw-data scope is UniST train shards `0..14`. The repaired cache
contains deterministic `320/480/640/800 ms` observations, one middle/late
observation when physically valid, deduplicated Phase3 teacher views, a strict
semantic cursor, observed committed history, and a deadline mask for short
utterances. A forced hard-deadline WRITE has no hard action/text/semantic CE.

The formal cache starts at batch 160. A measured H200 probe processed 160 rows
at 22.77 rows/s; batch 128 reached 100% utility, about 584 W and 102 GiB in the
active phase. The launcher retries unfinished shards at 144/128/96/64 only on
an actual CUDA OOM.

Eight workers use atomic per-shard claims instead of a static rank-to-shard
mapping. Faster GPUs therefore take the next unfinished shard, while a shard is
still written by exactly one process and becomes visible only after its atomic
`PART_COMPLETE.json` marker.

Main commands:

```bash
bash experiments/uniss_true_subsecond_pilot15_epoch1_v2/scripts/run_cache_smoke_1gpu.sh
bash experiments/uniss_true_subsecond_pilot15_epoch1_v2/scripts/launch_pipeline_tmux.sh
```

Formal outputs are under the experiment-specific `data/processed`,
`data/megatron`, `logs`, `reports`, `runs`, and `checkpoints` paths declared in
`config.env`.

The pipeline refuses to train unless the cache audit passes. It then packs
complete sessions atomically, creates a uniformly sampled replay-offset subset,
uses the previously committed strict global shuffle, performs a Phase3 handoff
through iteration 15, strictly resumes optimizer/RNG/sampler state, and finishes
one complete trajectory-coverage epoch on all eight GPUs.
