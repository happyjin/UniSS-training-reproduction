# Stage 0 — full-data stratified baselines

The audio reconstruction command passes all 198 parquet paths and caps each
shard independently. The default produces five records per shard (990 total),
instead of taking every record from shard 00000 as the earlier bootstrap did.
The prefix baseline remains a one-GPU evaluation and writes only to the v3
namespace.
