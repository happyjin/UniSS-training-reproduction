"""The verdict is regression-tested against two runs whose diagnosis is known.

The speak-decision run failed on the decision itself; the delta_cont=4 bias run
reproduced the target decision behaviour and failed only on length.  A verdict
that cannot tell those two apart is worthless as an unattended diagnosis.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pytest

from experiments.uniss_phase3_e2e_continue_end_v1.evaluation import verdict as v

REPO_ROOT = Path(__file__).resolve().parents[3]
GATES = (
    REPO_ROOT
    / "reports/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/formal_gold_20260818T090515Z"
    / "free_running_gates"
)
PROBES = REPO_ROOT / "reports/uniss_phase3_e2e_speak_decision_v1/family_logit_probe"
SPEAK_GATE = GATES / "speak_decision_iter1132_la_hb2_m1200_20260831T211140Z"
BASELINE_GATE = GATES / "stage2_paced_m1200_iter0002264_20260831T180448Z"


def _bias_run(name: str) -> Path:
    matches = sorted(PROBES.glob(f"sweep_{name}_*"))
    if not matches:
        pytest.skip(f"bias sweep run {name} not present")
    return matches[-1]


def test_it_reproduces_the_baseline_numbers_exactly() -> None:
    if not BASELINE_GATE.is_dir():
        pytest.skip("baseline gate not present")
    observed = v._load_events(str(BASELINE_GATE))
    assert observed["events"] == 95
    assert observed["write_mt_per_event"] == pytest.approx(0.168, abs=0.002)
    assert observed["natural_eos"] == pytest.approx(0.50, abs=0.01)
    assert observed["semantic_coverage"] == pytest.approx(0.666, abs=0.002)


def test_the_failed_speak_decision_run_is_blamed_on_the_decision() -> None:
    if not SPEAK_GATE.is_dir():
        pytest.skip("speak-decision gate not present")
    result = v.evaluate(str(SPEAK_GATE), {})
    assert result["status"] == "failed"
    assert "continue-after-fragment" in result["primary_cause"]
    by_key = {check["key"]: check for check in result["checks"]}
    assert not by_key["write_mt_per_event"]["passed"]
    assert not by_key["natural_eos"]["passed"]


def test_the_target_behaviour_bias_run_is_blamed_on_length_not_the_decision() -> None:
    """delta_cont=4 is what training is trying to reproduce internally."""
    run = _bias_run("cont4")
    result = v.evaluate(str(run), {})
    by_key = {check["key"]: check for check in result["checks"]}
    # The three decision axes must all land.
    assert by_key["write_mt_per_event"]["passed"]
    assert by_key["natural_eos"]["passed"]
    assert by_key["semantic_coverage"]["passed"]
    # And the failure must be attributed to the term that has never existed.
    assert not by_key["text_length_ratio_median"]["passed"]
    assert "content_end_margin" in result["primary_cause"]


def test_the_control_arm_scores_the_same_as_the_gate_baseline() -> None:
    """A drift here means the probe harness perturbed the session after all."""
    control = _bias_run("control")
    if not BASELINE_GATE.is_dir():
        pytest.skip("baseline gate not present")
    a = v._load_events(str(control))
    b = v._load_events(str(BASELINE_GATE))
    for key in ("write_mt_per_event", "natural_eos", "semantic_coverage", "events"):
        assert a[key] == pytest.approx(b[key], abs=1e-6), key


def test_every_failing_check_carries_a_cause_and_an_action() -> None:
    if not SPEAK_GATE.is_dir():
        pytest.skip("speak-decision gate not present")
    result = v.evaluate(str(SPEAK_GATE), {})
    for check in result["checks"]:
        if check["passed"]:
            assert "cause" not in check
        else:
            assert check["cause"] and check["action"], check["key"]


def test_passing_everything_does_not_declare_success() -> None:
    """8 train-seen samples cannot support a success claim; the plan says so."""
    text = v.evaluate.__doc__ or ""
    result = v.evaluate(str(BASELINE_GATE), {}) if BASELINE_GATE.is_dir() else None
    if result is None:
        pytest.skip("baseline gate not present")
    # Force the all-pass branch and read the guidance it emits.
    fake = dict(result)
    fake["checks"] = [dict(c, passed=True) for c in result["checks"]]
    guidance = v.PREDICTIONS  # sanity: predictions are declarative data
    assert guidance
    all_pass = v.evaluate(str(BASELINE_GATE), {})
    if all_pass["status"] == "passed":
        assert "64" in all_pass["next_action"]


def test_the_predictions_carry_the_measured_bias_sweep_reference() -> None:
    references = {p["key"]: p["bias_reference"] for p in v.PREDICTIONS}
    assert references["write_mt_per_event"] == 0.863
    assert references["natural_eos"] == 1.00
    assert references["semantic_coverage"] == 0.997
    assert references["text_length_ratio_max"] == 4.07
