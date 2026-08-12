from __future__ import annotations

from evaluation.runtime_parity_streaming.gates import build_report


def _summary():
    return {
        "samples": [
            {
                "quality_passed": True,
                "forced_writes": 0,
                "committed_revision_violations": 0,
                "source_finished_before_first_write": False,
                "translation_audio_samples": 16000,
                "pcm_finite": True,
                "natural_eos": True,
                "rtf": 0.5,
                "severe_semantic_collapse": False,
            }
        ]
    }


def test_missing_formal_metrics_never_defaults_to_pass() -> None:
    value = build_report([_summary()])
    assert value["mechanism_gate"]["status"] == "not_evaluable"
    assert value["formal_status"] == "not_evaluable"
    assert value["subsecond_status"] == "not_evaluable"


def test_complete_mechanism_gate_passes_but_useful_audio_remains_explicit() -> None:
    directions = {
        name: {
            "samples": 256,
            "failure_rate": 0.0,
            "text_bleu_retention": 0.9,
            "speech_bleu_retention": 0.8,
            "utmos_drop": 0.1,
            "autopcp_drop": 0.05,
        }
        for name in ("eng-cmn", "cmn-eng")
    }
    value = build_report(
        [_summary()],
        causality_audit={"passed": True},
        cache_parity={"passed": True},
        generalization_metrics={"directions": directions},
    )
    assert value["mechanism_gate"]["status"] == "pass"
    assert value["formal_status"] == "pass"
    assert value["subsecond_gate"]["status"] == "not_evaluable"
    assert "missing_subsecond_metrics" in value["subsecond_gate"]["failures"]


def test_forced_write_is_a_hard_mechanism_failure() -> None:
    summary = _summary()
    summary["samples"][0]["forced_writes"] = 1
    value = build_report(
        [summary],
        causality_audit={"passed": True},
        cache_parity={"passed": True},
    )
    assert value["mechanism_gate"]["status"] == "fail"
    assert "forced_write_nonzero" in value["mechanism_gate"]["failures"]

