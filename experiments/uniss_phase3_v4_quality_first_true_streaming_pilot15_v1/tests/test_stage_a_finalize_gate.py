from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.finalize_gate import (
    finalize,
)


def row(task: str, language: str, chunk_ms: int, errors: int) -> dict[str, object]:
    events = [
        {"content_tokens": [1], "reached_stop": True},
        {"content_tokens": [2], "reached_stop": True},
    ]
    return {
        "task": task,
        "language": language,
        "chunk_ms": chunk_ms,
        "ar_free_running": {
            "errors": errors,
            "reference_units": 10,
            "text": "ok",
            "all_events_reached_stop": True,
            "events": events,
            "write_structure_rate": 1.0,
        },
        "ar_teacher_forced": {"correct_tokens": 9, "target_tokens": 10},
        "ctc": {
            "input_frames": 10,
            "raw_nonblank_frames": 2,
            "collapsed_nonblank_tokens": 1,
        },
    }


def test_finalize_rejects_streaming_content_above_offline_gate() -> None:
    diagnosis = {
        "checkpoint": "/checkpoint/iter_0000381",
        "summary": {"samples": 4},
        "samples": [
            row("streaming_asr", "eng", 160, 3),
            row("streaming_asr", "cmn", 160, 2),
            row("causal_full_asr", "eng", 160, 1),
            row("causal_full_asr", "cmn", 160, 1),
        ],
    }
    baseline = {
        "quality_asr_error": {
            "eng->cmn": {
                "edits": 5,
                "reference_units": 100,
                "error_rate": 0.05,
                "metric": "WER",
                "samples": 10,
            },
            "cmn->eng": {
                "edits": 5,
                "reference_units": 100,
                "error_rate": 0.05,
                "metric": "CER",
                "samples": 10,
            },
        }
    }
    summary, gate = finalize(diagnosis, baseline, relative_degradation_limit=0.15)
    assert gate["passed"] is False
    assert gate["blocked_next_stage"] == "stage_b_incremental_mt"
    assert "streaming_eng_error_exceeds_offline_plus_15%" in gate["failed_checks"]
    assert "streaming_cmn_error_exceeds_offline_plus_15%" in gate["failed_checks"]
    assert summary["streaming_event_health"]["prefinal_content_rate"] == 1.0
    assert summary["by_task_language_chunk"]["streaming_asr:eng:160"][
        "error_rate"
    ] == 0.3
