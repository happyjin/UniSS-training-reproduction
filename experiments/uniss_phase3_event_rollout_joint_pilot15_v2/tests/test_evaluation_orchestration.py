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
    assert "compare_runtime_parity" in source


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
    assert "complete.json" in source
