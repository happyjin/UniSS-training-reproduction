from __future__ import annotations

from copy import deepcopy

from experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.select_final_checkpoint import (
    GateThresholds,
    select,
)


def _metric(value: float, field: str) -> dict:
    return {"groups": {"exact_runtime:eng->cmn": {"sample_count": 4, field: value}}}


def _retention_metric(base: float, adapted: float, field: str) -> dict:
    return {
        "groups": {
            "phase3_v4:eng->cmn": {"sample_count": 4, field: base},
            "streaming_adapter:eng->cmn": {"sample_count": 4, field: adapted},
        }
    }


def _runtime(*, useful: bool = True) -> dict:
    group = {
        "samples": 8,
        "natural_write_sample_rate": 0.875 if useful else 0.0,
        "all_wait_rate": 0.125 if useful else 1.0,
        "post_source_eos_first_write_rate": 0.125,
        "first_write_premature_rate": 0.125,
        "forced_writes": 0,
        "committed_revision_violations": 0,
        "playable_pcm_rate": 1.0,
        "finite_pcm_rate": 1.0,
        "severe_semantic_collapse_rate": 0.0,
        "natural_eos_rate": 0.875,
    }
    return {"coverage": {"complete": True}, "groups": {"all": group}}


def _candidate(iteration: int = 350, p50: float = 800.0) -> dict:
    quality = {
        "text_bleu.json": _metric(30.0, "score"),
        "speech_bleu.json": _metric(25.0, "score"),
        "slc.json": _metric(0.8, "slc_0_4"),
        "utmos.json": _metric(3.5, "mean"),
        "autopcp.json": _metric(3.0, "mean"),
        "speaker_similarity.json": _metric(0.75, "mean"),
    }
    retention_quality = {
        "text_bleu.json": _retention_metric(40.0, 36.0, "score"),
        "speech_bleu.json": _retention_metric(35.0, 30.0, "score"),
        "slc.json": _retention_metric(0.9, 0.8, "slc_0_4"),
        "utmos.json": _retention_metric(3.8, 3.5, "mean"),
        "autopcp.json": _retention_metric(3.5, 3.0, "mean"),
    }
    return {
        "iteration": iteration,
        "evaluation_root": f"/eval/{iteration}",
        "retention_root": f"/retention/{iteration}",
        "train_runtime": _runtime(),
        "valid_runtime": _runtime(),
        "useful_audio": {
            "groups": {
                "all": {
                    "useful_audio_recall": 0.75,
                    "first_useful_audio_wall_ms": {"p50": p50, "p90": 950.0, "p95": 980.0},
                }
            }
        },
        "parity": {"passed": True, "failure_count": 0},
        "quality": quality,
        "retention": {
            "paired_complete": True,
            "groups": {
                "streaming_adapter": {
                    "generated_text_rate": 1.0,
                    "semantic_output_rate": 1.0,
                    "playable_audio_rate": 1.0,
                    "finite_audio_rate": 1.0,
                    "non_silent_audio_rate": 1.0,
                }
            },
        },
        "retention_quality": retention_quality,
    }


def test_selector_chooses_fastest_checkpoint_only_after_all_gates_pass() -> None:
    report = select(
        [_candidate(300, 900.0), _candidate(350, 700.0)], GateThresholds()
    )
    assert report["selection_status"] == "selected"
    assert report["selected_iteration"] == 350
    assert report["selection_order"] == [350, 300]


def test_selector_rejects_arbitrary_pcm_without_useful_audio() -> None:
    candidate = _candidate()
    candidate["useful_audio"]["groups"]["all"]["useful_audio_recall"] = 0.0
    candidate["useful_audio"]["groups"]["all"]["first_useful_audio_wall_ms"]["p50"] = None
    report = select([candidate], GateThresholds())
    assert report["selection_status"] == "no_checkpoint_passed"
    reasons = report["candidates"][0]["rejection_reasons"]
    assert "valid:useful_audio_recall_below_threshold" in reasons
    assert "valid:first_useful_audio_p50_not_subsecond" in reasons


def test_selector_rejects_forced_write_all_wait_collapse_and_parity_failure() -> None:
    candidate = _candidate()
    valid = candidate["valid_runtime"]["groups"]["all"]
    valid["natural_write_sample_rate"] = 0.0
    valid["all_wait_rate"] = 1.0
    valid["forced_writes"] = 1
    valid["severe_semantic_collapse_rate"] = 1.0
    candidate["parity"]["passed"] = False
    report = select([candidate], GateThresholds())
    reasons = report["candidates"][0]["rejection_reasons"]
    assert "valid:natural_write_rate_below_threshold" in reasons
    assert "valid:all_wait_rate_above_threshold" in reasons
    assert "valid:forced_write_nonzero" in reasons
    assert "valid:semantic_or_audio_collapse" in reasons
    assert "runtime_parity_failed_or_missing" in reasons


def test_selector_rejects_missing_quality_and_phase3_retention_collapse() -> None:
    candidate = _candidate()
    del candidate["quality"]["speech_bleu.json"]
    candidate["retention"]["groups"]["streaming_adapter"]["playable_audio_rate"] = 0.25
    candidate["retention_quality"]["text_bleu.json"] = _retention_metric(40.0, 5.0, "score")
    report = select([candidate], GateThresholds())
    reasons = report["candidates"][0]["rejection_reasons"]
    assert "missing_or_nonfinite:valid_quality.speech_bleu" in reasons
    assert "phase3_retention:playable_audio_rate_below_threshold" in reasons
    assert "phase3_retention:text_bleu_ratio_below_threshold" in reasons


def test_selector_does_not_mutate_candidate_evidence() -> None:
    candidate = _candidate()
    original = deepcopy(candidate)
    select([candidate], GateThresholds())
    assert candidate == original
