# Phase3 event-rollout joint SFT: fixed UniST 15-shard

This isolated experiment reuses the tested exact-runtime implementation from
`uniss_phase3_event_rollout_joint_full198_v1`, but its formal data scope is
strictly UniST train shards `00000` through `00014`.

Key invariants:

- only Phase3 v4 `iter_0009075` may initialize a fresh run;
- all trajectory packs live in a multi-file prefix-sum global namespace;
- every coverage epoch uses a deterministic full `randperm` over complete pack IDs;
- session-internal 160 ms event order is immutable;
- Phase3 replay is fixed to the same 15 shards and targets 35%;
- Megatron, 8 H200, BF16, Flash Attention, MBS 2, GBS 128, sequence 18000;
- first formal run is exactly one complete coverage epoch (717 iterations);
- self-resume requires distributed-checkpoint strictness `raise_all`.

Formal artifact preparation refuses to run until the full data audit reports
`status=pass`.  Timing provenance is recorded as forced/oracle pseudo timing,
not observed natural exact READ/WRITE timing.

The isolated smoke wrapper exposes one real pack from each of the 32 train
parts and one from each of the 4 validation parts.  It uses three tiny coverage
epochs so `SMOKE_EXIT_INTERVAL=1` can verify interruption and strict self-resume
without treating a synthetic dataset as evidence.
