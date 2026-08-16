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
