# UniSS Phase3 true-subsecond full198 v1

This directory is an isolated implementation of
`docs/uniss_training_reproduction/uniss_phase3_true_subsecond_deadline_streaming_full198_implementation_plan.md`.

It never writes into historical Phase1/2/3, Student, StreamSpeech, GRPO, or
Gradio experiment paths. The formal run is initialized from the exported
Phase3 v4 checkpoint and is orchestrated by the repository's Megatron runtime.

Formal invariants:

- all 198 UniST training shards are indexed;
- every accepted row materializes Quality replay, Performance replay, early
  trajectory, and middle/late trajectory tasks;
- final train iterations are `ceil(packed_count / 128)`;
- 8 GPUs, micro batch 2, global batch 128, sequence length 18000;
- no core CTC objective;
- launchers refuse to overwrite an existing run unless `RESUME=1` and a
  checkpoint tracker exists.

All generated data, caches, logs, runs, and checkpoints remain under
`/opt/dlami/nvme/jasonleeeli`.

## Data preparation status and cache smoke

The full direction index and deterministic trajectory plan contain 198/198
shards, 19,281,676 accepted rows, and 38,563,352 trajectory points. Formal
cache generation is intentionally gated on a real-model smoke test.

The cache stores bounded-causal WhisperVQ token IDs rather than permanent
1280-dimensional pre-VQ hidden states. Training restores quantized hidden
states with the frozen WhisperVQ codebook. Cache references use disjoint
namespaces:

```text
bundle.npz::causal:<batch-row>
bundle.npz::teacher:<request-index>
```

Create an isolated two-row index and run the real GPU smoke without modifying
formal data:

```bash
source experiments/uniss_phase3_true_subsecond_deadline_full198_v1/config.env
SMOKE_ROOT="$DATA_ROOT/smoke/cache_2row_v3"

"$PYTHON" -m \
  experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.build_cache_smoke_index \
  --source-root "$INDEX_ROOT" \
  --output-root "$SMOKE_ROOT/index" \
  --shard 0 \
  --limit 2

CUDA_VISIBLE_DEVICES=0 \
TMPDIR=/opt/dlami/nvme/jasonleeeli/tmp \
HF_HOME=/opt/dlami/nvme/jasonleeeli/hf_cache \
"$PYTHON" -m \
  experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.build_trajectory_cache \
  --raw-unist-dir "$RAW_UNIST_DIR" \
  --index-root "$SMOKE_ROOT/index" \
  --output-root "$SMOKE_ROOT/cache" \
  --phase3-model "$PHASE3_MODEL" \
  --whispervq-model "$REPO_ROOT/pretrained_models/UniSS/glm4_tokenizer" \
  --bicodec-checkpoint "$REPO_ROOT/pretrained_models/UniSS/bicodec/BiCodec" \
  --rank 0 --world-size 1 --limit-shards 1 --batch-size 2
```

The validated v3 smoke produces four checksum-valid trajectories with causal
row references `0,0,1,1`, teacher request references `0,4,8,12`, and distinct
causal lengths of 53/112 tokens for the two variable-duration source rows.

Trajectory task materialization is also isolated per shard. It preserves the
ordinary Phase3 next-token tensors and adds `token_roles` plus one compact
sidecar per packed boundary. Roles distinguish action, text, AR semantic, and
boundary losses so the trainer can normalize each objective independently.
Deadline-forced WRITE samples carry only the action hard label; anticipated
content remains teacher-top-k soft supervision and is never written as a hard
future reference.

```bash
PART="$PACKED_ROOT/parts/part-000"
"$PYTHON" -m \
  experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.pack_trajectory_cache \
  --cache-part "$CACHE_ROOT/part-000" \
  --raw-parquet "$RAW_UNIST_DIR/train-00000.parquet" \
  --output "$PART/packed_trajectory.jsonl" \
  --marker "$PART/PACK_COMPLETE.json" \
  --seq-length 18000

"$PYTHON" -m \
  experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.assemble_trajectory_packs \
  --parts-root "$PACKED_ROOT/parts" \
  --output "$PACKED_ROOT/packed_trajectory.jsonl" \
  --offsets "$PACKED_ROOT/packed_trajectory.offsets.u64" \
  --marker "$PACKED_ROOT/ASSEMBLY_COMPLETE.json" \
  --shard-count 198 \
  --seq-length 18000
```

Assembly writes an immutable concatenated JSONL, a uint64 byte-offset sidecar,
and `packed_trajectory.jsonl.count`. Formal `TRAIN_ITERS` is derived only after
this real count and the Phase3 replay count are frozen.

After the 1024-row sustained benchmark passes, launch the resumable formal
cache on all eight GPUs and inspect it without attaching to the worker shell:

```bash
bash experiments/uniss_phase3_true_subsecond_deadline_full198_v1/scripts/launch_cache_full198_tmux.sh
bash experiments/uniss_phase3_true_subsecond_deadline_full198_v1/scripts/status.sh
```

The formal cache uses batch 64 (the measured throughput optimum on this H200
host), one disjoint shard stream per GPU rank, ten-second telemetry, atomic
part markers, and a final 198/198 validation summary.
