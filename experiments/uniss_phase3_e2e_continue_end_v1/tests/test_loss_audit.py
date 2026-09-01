"""The audit must catch the failure mode that went unnoticed for 717 updates."""
from __future__ import annotations

from pathlib import Path

import pytest

from experiments.uniss_phase3_e2e_continue_end_v1.evaluation import loss_audit as la

REPO_ROOT = Path(__file__).resolve().parents[3]
LOGS = REPO_ROOT / "logs/uniss_phase3_e2e_continue_end_v1"


def _synthetic(path: Path, *, dead_weight: float, dead_rows: float) -> Path:
    """Two iterations, one term alive and one term weighted but never firing."""
    lines = [
        "  e2e_asr_weight ....... 1.0\n",
        f"  e2e_ghost_weight  {dead_weight}\n",
        '{"objective_extension": {"repetition_penalty": 0.3, "repetition_window": 8.0}}\n',
    ]
    for iteration, value in ((1, 2.0), (2, 1.0)):
        lines.append(
            f" iteration {iteration}/  10 | loss/asr_ce: {value} | "
            f"denominator/asr_ce: 1000.0 | loss/ghost: 0.0 | "
            f"denominator/ghost: {dead_rows} |\n"
        )
    path.write_text("".join(lines))
    return path


def test_a_weighted_term_that_never_fires_is_reported_as_dead(tmp_path: Path) -> None:
    log = _synthetic(tmp_path / "dead.log", dead_weight=0.25, dead_rows=0.0)
    result = la.audit(str(log))
    assert result["status"] == "failed"
    assert result["dead_with_weight"] == ["ghost"]


def test_a_zero_weight_term_that_never_fires_is_not_an_error(tmp_path: Path) -> None:
    """speaker_continuity is deliberately off; it must not raise an alarm."""
    log = _synthetic(tmp_path / "monitor.log", dead_weight=0.0, dead_rows=0.0)
    result = la.audit(str(log))
    assert result["status"] == "passed"
    assert result["dead_with_weight"] == []


def test_a_weighted_term_with_too_few_rows_is_flagged_as_negligible(tmp_path: Path) -> None:
    log = _synthetic(tmp_path / "thin.log", dead_weight=0.25, dead_rows=23.0)
    result = la.audit(str(log))
    assert result["status"] == "passed", "negligible is a warning, not a failure"
    assert [item["name"] for item in result["negligible_terms"]] == ["ghost"]


def test_it_parses_long_argument_names_that_get_no_dot_padding(tmp_path: Path) -> None:
    """Megatron pads short names with dots and long ones with spaces."""
    log = tmp_path / "pad.log"
    log.write_text(
        "  e2e_semantic_rollin_continue_decision_margin_weight  0.25\n"
        "  e2e_asr_weight ................ 1.0\n"
        " iteration 1/ 10 | loss/asr_ce: 1.0 | denominator/asr_ce: 5.0 |\n"
    )
    weights = la.parse_weights(str(log))
    assert weights["semantic_rollin_continue_decision_margin"] == 0.25
    assert weights["asr_ce"] == 1.0


def test_aggregates_are_not_mistaken_for_dead_losses(tmp_path: Path) -> None:
    """boundary_eos is a weighted combination; its denominator is zero by design."""
    log = tmp_path / "agg.log"
    log.write_text(
        "  e2e_boundary_eos_weight ....... 0.1\n"
        " iteration 1/ 10 | loss/boundary_eos: 0.5 | denominator/boundary_eos: 0.0 | "
        "loss/boundary_ce: 0.4 | denominator/boundary_ce: 900.0 |\n"
    )
    result = la.audit(str(log))
    assert result["status"] == "passed"
    kinds = {e["name"]: e["kind"] for e in result["entries"]}
    assert kinds["boundary_eos"] == "aggregate"
    assert kinds["boundary_ce"] == "component"


def test_the_live_run_has_no_dead_weighted_term() -> None:
    logs = [p for p in sorted(LOGS.glob("continue_end_m3_*.log")) if p.name.count(".") == 1 and "chain" not in p.name and "launcher" not in p.name]
    if not logs:
        pytest.skip("no continue-end run log present")
    result = la.audit(str(logs[-1]))
    assert result["dead_with_weight"] == [], result["dead_with_weight"]
    # The three new terms must all be alive and weighted.
    kinds = {e["name"]: e["kind"] for e in result["entries"]}
    for name in ("continue_after_fragment", "content_end_margin", "repetition_penalty"):
        assert kinds.get(name) == "active", (name, kinds.get(name))


def test_rising_is_reported_rather_than_treated_as_a_failure() -> None:
    """Terms in tension trade against each other; the audit surfaces it."""
    logs = [p for p in sorted(LOGS.glob("continue_end_m3_*.log")) if p.name.count(".") == 1 and "chain" not in p.name and "launcher" not in p.name]
    if not logs:
        pytest.skip("no continue-end run log present")
    result = la.audit(str(logs[-1]))
    assert result["counts"]["rising"] > 0
    assert result["status"] == "passed"
