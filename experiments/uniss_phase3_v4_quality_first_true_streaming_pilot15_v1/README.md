# Phase3 v4 quality-first true-streaming pilot15

This directory implements the staged experiment specified in
`docs/uniss_training_reproduction/uniss_phase3_v4_quality_first_true_streaming_asr_mt_tts_training_plan.md`.

The experiment is intentionally isolated. It does not modify historical
training, evaluation, demo, checkpoint, log, or data directories. Every stage
writes to the experiment-specific roots declared in `experiment.env`, refuses
ambiguous existing output, and has its own result report.

## Stage order

1. `stage00_baseline`: causal WhisperVQ frontend and Phase3 artifact audit;
2. `stage_a_streaming_asr`: chunk-causal acoustic encoder plus incremental ASR;
3. `stage_b_incremental_mt`: committed-source incremental translation;
4. `stage_c_fragment_tts`: aligned semantic continuation TTS;
5. `stage_d_frozen_e2e`: frozen stateful end-to-end evaluation.

Stage 00 must write `GATE_PASSED.json` before Stage A is allowed to prepare
training data. The first Stage 00 implementation deliberately writes the
narrower `FRONTEND_GATE_PASSED.json`; it cannot falsely authorize Stage A
until the remaining Phase3 and BiCodec audits are complete.

## Stage 00 quick start

```bash
bash experiments/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/scripts/run_stage00_cpu_tests.sh
bash experiments/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/scripts/launch_stage00_frontend_tmux.sh
bash experiments/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/scripts/launch_stage00_offline_baseline_tmux.sh
bash experiments/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/scripts/status.sh
```

The GPU launcher stops only the explicitly named synthetic load session
`uniss_gpu_load_60`. It refuses to kill unknown GPU processes.

The completed Stage 00 report is
`reports/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage00_baseline/STAGE00_RESULT_REPORT.md`.
The gate is conditional: Qwen cached validation/runtime must use FP32 eager
attention until a separate BF16 parity gate passes.

## Stage A data gate

Stage A uses a label-independent fixed UTF-8 byte vocabulary (256 labels plus
blank) for the auxiliary source CTC head. This is the audited fallback after a
strict train-only Qwen map exposed rare validation OOVs. The Phase3 AR-ASR path
still uses the original Qwen tokenizer. Train and validation labels cannot
change the CTC inventory, and every record must pass UTF-8 round-trip plus
20-ms-frame CTC feasibility gates.

```bash
bash experiments/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/scripts/run_stage_a_cpu_tests.sh
bash experiments/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/scripts/launch_stage_a_ctc_maps_tmux.sh
bash experiments/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/scripts/prepare_stage_a_inputs.sh
bash experiments/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/scripts/launch_stage_a_data_audit_tmux.sh
```

Every command refuses to overwrite an existing map, snapshot, or audit run.

## Stage A formal result

The native-Megatron formal run `stage_a_formal8_20260816T224100Z` completed
381/381 updates and three strict global-shuffle coverage epochs without skipped
or non-finite iterations. Its full free-running quality gate failed, so Stage B
is intentionally blocked and no `SELECTED_CHECKPOINT.json` was created.

The immutable decision artifacts are:

- `reports/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_formal/stage_a_formal8_20260816T224100Z/STAGE_A_RESULT_REPORT.md`;
- `reports/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_formal/stage_a_formal8_20260816T224100Z/STAGE_A_FINAL_SUMMARY.json`;
- `reports/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_formal/stage_a_formal8_20260816T224100Z/GATE_FAILED.json`.
