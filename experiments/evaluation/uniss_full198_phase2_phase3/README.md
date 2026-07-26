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
the export directory name. An existing export/output directory is rejected. The
conversion first writes to a process-specific `.partial` directory and removes
it automatically on failure before atomically publishing the verified export.

Megatron/Triton import requires a visible CUDA driver even though checkpoint
weights are initialized and converted on CPU. Do not run export with an empty
`CUDA_VISIBLE_DEVICES`; exposing one GPU is sufficient and does not place the
checkpoint weights on that GPU.

The logical UniSS tokenizer contains 180,407 tokens. Megatron pads the embedding
matrix to 180,480 rows (73 dummy rows), so a valid exported checkpoint has
`tokenizer_size=180407` and `config.vocab_size=180480`. This matches the existing
13-shard exports.

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

The isolated full-evaluation environment is:

```text
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval
```

The official repository tested vLLM 0.8.0. Its pinned `xgrammar==0.1.16`
artifact is no longer available from the active package index, so this run uses
the API-compatible `vllm==0.8.5.post1` and records that version in every vLLM
`run_config.json`.

Recreate the isolated environment and download metric models with:

```bash
experiments/evaluation/uniss_full198_phase2_phase3/setup_eval_environment.sh
experiments/evaluation/uniss_full198_phase2_phase3/prepare_metric_models.sh
```

After metric-model preparation succeeds and while any existing training job is
still allowed to finish normally, start the complete non-overwriting pipeline:

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)" \
tmux new-session -d -s uniss_full198_evaluation \
  "cd ${PWD} && RUN_ID=${RUN_ID} experiments/evaluation/uniss_full198_phase2_phase3/run_complete_evaluation.sh"
```

The pipeline waits until all local GPU compute processes have exited. It then
runs the fixed Phase2/Phase3 HF smoke matrix, validates every objective metric
on the smoke audio, repeats the same smoke set through the vLLM and BiCodec
pipeline, and only then runs the 50-record listening matrix, full UniST dev,
full UniST test, objective metrics, and the aggregate report. Dev is completed
before test. Full generation and objective metrics are resumable; an
incomplete HF smoke/listening directory is preserved and requires a new
`RUN_ID`. Set `PREFLIGHT_ONLY=1` to validate and freeze inputs without starting
GPU work.

Common Voice v4 Chinese source speech is license-gated and is not available on
this host. The pipeline copies the audited CVSS-T blocked-state manifest into
the final report; it does not mislabel UniST results as CVSS-T paper results.

The final report is written to both:

```text
report/aggregate_report.md
report/phase2_phase3_detailed_evaluation_report.md
```

It contains Phase2-vs-Phase3 Q/P and direction-specific deltas, generation and
failure audits, artifact paths, the exact UniSS paper CVSS-T Table 1 baseline
values, and the paper's 0.5B efficiency reference. Direct paper deltas are
enabled only when the local run is detected as full CVSS-T test evaluation;
UniST dev/test results are explicitly marked as cross-dataset and are never
ranked against CVSS-T numbers.
