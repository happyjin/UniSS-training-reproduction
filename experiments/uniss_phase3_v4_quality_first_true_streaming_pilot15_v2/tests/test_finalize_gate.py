from copy import deepcopy

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.finalize_gate import (
    GROUPS,
    finalize,
)


def _row(task: str, language: str, index: int) -> dict:
    return {
        "task": task,
        "language": language,
        "sample_id": f"{task}-{language}-{index}",
        "ar_free_running": {
            "errors": 0,
            "reference_units": 10,
            "text": "ok",
            "all_events_reached_stop": True,
            "events": [
                {"content_tokens": [1]},
                {"content_tokens": [2]},
            ],
        },
        "ctc": {"collapsed_nonblank_tokens": 1},
        "committed_rollback": {"rollback_count": 0},
        "cached_recomputed_parity": {
            "hidden": {"allclose": True},
            "tokens": {"exact": True},
            "bridge_residual": {"allclose": True},
            "free_generation_exact": True,
        },
    }


def _fixtures():
    rows = []
    offline_groups = {}
    counts = {
        "streaming_asr:cmn": 114,
        "streaming_asr:eng": 129,
        "causal_full_asr:cmn": 36,
        "causal_full_asr:eng": 55,
    }
    for group in GROUPS:
        task, language = group.split(":")
        rows.extend(_row(task, language, index) for index in range(counts[group]))
        offline_groups[group] = {
            "metric": "cer" if language == "cmn" else "wer",
            "samples": counts[group],
            "error_rate": 0.1,
        }
    diagnosis = {
        "schema_version": "uniss_quality_first_stage_a_checkpoint_diagnosis_v2",
        "checkpoint": "/checkpoint",
        "samples": rows,
    }
    offline = {
        "schema_version": "uniss_quality_first_stage_a_matching_offline_asr_v1",
        "records": 334,
        "unique_ids": 334,
        "metrics_by_task_language": offline_groups,
    }
    frontend = {
        "schema_version": "uniss_quality_first_stage_a_checkpoint_frontend_passed_v2",
        "passed": True,
    }
    return diagnosis, offline, frontend


def test_finalize_gate_passes_exact_healthy_coverage() -> None:
    diagnosis, offline, frontend = _fixtures()
    result = finalize(diagnosis, offline, frontend, relative_degradation_limit=0.15)
    assert result["passed"] is True
    assert result["stage_b_authorized"] is True
    assert result["coverage"]["unique_ids"] == 334


def test_finalize_gate_blocks_rollback() -> None:
    diagnosis, offline, frontend = _fixtures()
    broken = deepcopy(diagnosis)
    broken["samples"][0]["committed_rollback"]["rollback_count"] = 1
    result = finalize(broken, offline, frontend, relative_degradation_limit=0.15)
    assert result["passed"] is False
    assert "committed_rollback_zero" in result["failed_checks"]
