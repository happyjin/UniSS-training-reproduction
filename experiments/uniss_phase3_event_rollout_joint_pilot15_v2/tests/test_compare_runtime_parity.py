from __future__ import annotations

from experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.compare_runtime_parity import (
    compare,
)


def _summary(text_id: int = 7):
    return {
        "samples": [
            {
                "sample_id": "sample",
                "generated_text": "hello",
                "natural_writes": 1,
                "natural_eos": True,
                "forced_writes": 0,
                "events": [
                    {
                        "event_index": 0,
                        "source_end_ms": 160,
                        "source_finished": False,
                        "new_source_codes": 2,
                        "action": "WRITE",
                        "text_ids": [text_id],
                        "semantic_codes": [3, 4],
                        "continuation_choice": None,
                        "compute_ms": 10.0,
                    }
                ],
            }
        ]
    }


def test_runtime_parity_ignores_expected_timing_differences() -> None:
    first = _summary()
    second = _summary()
    second["samples"][0]["events"][0]["compute_ms"] = 99.0
    report = compare({"fused_cached": first, "unfused_uncached": second})
    assert report["passed"] is True


def test_runtime_parity_detects_committed_token_difference() -> None:
    report = compare({"a": _summary(), "b": _summary(text_id=8)})
    assert report["passed"] is False
    assert report["failures"][0]["reason"] == "semantic_output_mismatch"
