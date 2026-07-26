# UniSS full198 Phase2/Phase3 evaluation

This directory contains the isolated, non-overwriting execution entrypoints for
the evaluation plan in:

```text
docs/uniss_training_reproduction/uniss_evaluation_dataset_and_audio_generation_plan.md
```

Historical 13-shard scripts and output directories are not modified.

## Frozen checkpoints at initialization

```text
Phase2: checkpoints/uniss_qwen0p5b_phase2_unist198_from_phase1_fast_decay_v4/iter_0015381
Phase3: checkpoints/uniss_qwen0p5b_phase3_unist198_after_phase2_v4/iter_0009075
```

Every export command reads the tracker again and writes the exact iteration into
the export directory name. An existing export/output directory is rejected.

## Commands

Prepare deterministic manifests:

```bash
experiments/evaluation/uniss_full198_phase2_phase3/prepare_manifests.sh
```

Export exact checkpoints:

```bash
experiments/evaluation/uniss_full198_phase2_phase3/export_exact.sh phase2
experiments/evaluation/uniss_full198_phase2_phase3/export_exact.sh phase3
```

Run the two-checkpoint HF smoke matrix after a GPU is free:

```bash
EVAL_CUDA_VISIBLE_DEVICES=0 \
experiments/evaluation/uniss_full198_phase2_phase3/run_hf_matrix.sh smoke
```

Run the 50-record listening matrix:

```bash
EVAL_CUDA_VISIBLE_DEVICES=0 \
experiments/evaluation/uniss_full198_phase2_phase3/run_hf_matrix.sh listen
```

The HF path is intentionally limited to smoke/listening evaluation. Full dev
and test generation use the separate vLLM runner after its isolated runtime is
validated.
