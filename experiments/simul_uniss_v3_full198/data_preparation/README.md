# Full198 data preparation

This pipeline prepares all UniST train shards `00000` through `00197` without
writing into the v1/v2 data directories.

1. `prepare_shards.sh` creates one atomic, checksummed schedule/sample part per
   parquet. Completed parts are verified and skipped on restart. Rows with
   missing/invalid token streams are skipped only in this full-data experiment
   and counted by reason; the shared historical default remains fail-fast.
2. `pack_shards.sh` independently creates interleaved and action-only packed
   parts. The large temporary action sample is deleted after its packed part is
   published.
3. `assemble_full198.sh` validates all 198 markers, concatenates parts in shard
   order, writes the aggregate manifest, evaluates a deterministic schedule
   subset, generates compact byte-offset sidecars, and calculates exact Stage
   3/4/6 iteration counts.

The default worker counts are eight schedule workers and four pack workers.
They are CPU/storage settings, not Megatron data-loader workers.

```bash
experiments/simul_uniss_v3_full198/data_preparation/run_full_preparation.sh --dry-run
experiments/simul_uniss_v3_full198/data_preparation/launch_tmux.sh
```

No formal training launcher accepts the data until `FULL_DATA_READY.json` and
`training_schedule.env` both exist.
