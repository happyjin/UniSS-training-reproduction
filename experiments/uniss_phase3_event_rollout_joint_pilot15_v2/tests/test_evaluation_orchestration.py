from __future__ import annotations

from pathlib import Path


def test_eight_gpu_orchestration_is_complete_and_non_overwriting() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "evaluation" / "run_checkpoint_evaluation_8gpu.sh").read_text(
        encoding="utf-8"
    )
    assert '[[ ! -e "${OUTPUT_ROOT}" ]]' in source
    assert "GPU_LIST must contain exactly 8 GPU IDs" in source
    assert "checkpoint_export" in source
    assert source.index("checkpoint_export") < source.index("run_eight_shards valid")
    assert "run_eight_shards valid" in source
    assert "run_eight_shards train" in source
    assert 'VALID_SAMPLES_PER_DIRECTION="${VALID_SAMPLES_PER_DIRECTION:-}"' in source
    assert "--expected-manifest" in source
    assert "fused_cached fused_uncached unfused_cached unfused_uncached" in source
    assert 'KV_CACHE="${parity_cache[${index}]}"' in source
    assert "compare_runtime_parity" in source
    assert '"${OUTPUT_ROOT}/complete.json"' in source
    assert 'REPO_ROOT="$(cd "${EVAL_DIR}/../../.." && pwd)"' in source


def test_quality_orchestration_runs_all_hard_gate_metrics() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "evaluation" / "run_quality_metrics_8gpu.sh").read_text(
        encoding="utf-8"
    )
    for module in (
        "evaluation.asr_transcribe",
        "evaluation.text_metrics",
        "evaluation.slc_metrics",
        "evaluation.utmos_metrics",
        "evaluation.autopcp_metrics",
        "speaker_similarity",
        "build_prefix_asr_manifest",
        "score_prefix_asr",
    ):
        assert module in source
    assert "GPU_LIST must contain exactly 8 GPU IDs" in source
    assert 'AUTOPCP_ENCODER="${AUTOPCP_ENCODER:-${USER_ROOT}/evaluation_models/wav2vec2-large-xlsr-53}"' in source
    assert '--encoder-model "${AUTOPCP_ENCODER}"' in source
    assert "complete.json" in source
    assert 'REPO_ROOT="$(cd "${EVAL_DIR}/../../.." && pwd)"' in source


def test_phase3_retention_is_paired_and_eight_gpu() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "evaluation" / "run_phase3_retention_8gpu.sh").read_text(
        encoding="utf-8"
    )
    assert "GPU_LIST must contain exactly 8 GPU IDs" in source
    assert "evaluate_phase3_retention" in source
    assert "merge_phase3_retention" in source
    assert '"${OUTPUT_ROOT}/complete.json"' in source
    assert "--samples-per-direction" in source

    metrics = (
        root / "evaluation" / "run_phase3_retention_metrics_8gpu.sh"
    ).read_text(encoding="utf-8")
    assert "evaluation.text_metrics" in metrics
    assert "evaluation.slc_metrics" in metrics
    assert "run_objective_metrics.sh" in metrics
    assert "AUTOPCP_ENCODER" in metrics
    assert "complete.json" in metrics
    assert 'REPO_ROOT="$(cd "${EVAL_DIR}/../../.." && pwd)"' in source
    assert 'REPO_ROOT="$(cd "${EVAL_DIR}/../../.." && pwd)"' in metrics


def test_final_selector_requires_all_runtime_quality_and_retention_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "evaluation" / "select_final_checkpoint.py").read_text(
        encoding="utf-8"
    )
    for evidence in (
        "train_aggregate",
        "valid_aggregate",
        "useful_audio.json",
        "parity",
        "speaker_similarity.json",
        "phase3_retention",
        "no_checkpoint_passed",
    ):
        assert evidence in source


def test_post_training_pipeline_waits_then_probes_and_full_evaluates() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "evaluation" / "run_post_training_pipeline.sh").read_text(
        encoding="utf-8"
    )
    assert "after training is done" in source
    assert '[[ -e "${OUTPUT_ROOT}" && "${RESUME}" != 1 ]]' in source
    assert 'RESUME="${RESUME:-0}"' in source
    assert "runtime_complete" in source
    assert "retention_complete" in source
    assert "Cannot safely resume incomplete runtime root" in source
    assert "summarize_validation" in source
    assert "shortlist_checkpoints" in source
    assert "run_checkpoint_evaluation_8gpu.sh" in source
    assert "run_quality_metrics_8gpu.sh" in source
    assert "run_phase3_retention_8gpu.sh" in source
    assert "run_phase3_retention_metrics_8gpu.sh" in source
    assert source.count("select_final_checkpoint") == 2
    assert "VALID_SAMPLES_PER_DIRECTION=" in source
    assert 'REPO_ROOT="$(cd "${EVAL_DIR}/../../.." && pwd)"' in source
