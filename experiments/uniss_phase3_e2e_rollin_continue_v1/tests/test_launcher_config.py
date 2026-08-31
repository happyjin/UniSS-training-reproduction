"""Static guards on the roll-in continuation launcher.

These run on CPU and protect the two mistakes that are expensive to discover
after a seven hour training run: a roll-in weight paired with a zero roll-in
rate, which is a silently dead loss, and drift away from the established
launcher's data, geometry and audits.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
OWN = REPO / "experiments/uniss_phase3_e2e_rollin_continue_v1/scripts/run_8gpu.sh"
ESTABLISHED = (
    REPO
    / "experiments/uniss_phase3_v4_e2e_simuls2st_pilot15_v1"
    / "end_margin_epoch23_v1/run_8gpu.sh"
)


def _text() -> str:
    return OWN.read_text(encoding="utf-8")


def _default(name: str) -> str:
    match = re.search(rf"^{name}=\$\{{{name}:-([^}}]*)\}}", _text(), re.MULTILINE)
    assert match, f"{name} has no defaulted assignment"
    return match.group(1)


def test_the_launcher_exists_and_parses_as_bash() -> None:
    import subprocess

    assert OWN.is_file()
    subprocess.run(["bash", "-n", str(OWN)], check=True)


def test_it_delegates_to_the_established_megatron_launcher() -> None:
    text = _text()
    assert 'scripts/run_e2e_megatron.sh' in text
    assert "uniss_phase3_v4_e2e_simuls2st_pilot15_v1" in text


def test_it_writes_only_into_its_own_namespace() -> None:
    text = _text()
    assert "OWN_NAME=uniss_phase3_e2e_rollin_continue_v1" in text
    # Only the definitions matter; the env pass-through lines are indented.
    for line in text.splitlines():
        if line.startswith(("RUN_SAVE_DIR=", "RUN_TENSORBOARD_DIR=", "RUN_LOG=", "RUN_REPORT_ROOT=")):
            assert "${OWN_NAME}" in line, line


def test_the_parent_is_the_epoch3_checkpoint() -> None:
    assert _default("PARENT_RUN_ID") == "endmargin_epoch23_15shard_20260824T190227Z"
    assert _default("PARENT_ITER") == "0002264"


def test_data_geometry_and_seed_match_the_established_run() -> None:
    text, established = _text(), ESTABLISHED.read_text(encoding="utf-8")
    assert _default("TASK_POOL_RUN_ID") == "task_pool_formal_p4_20260820T154500Z"
    assert _default("RUN_SEED") == "20260819"
    for setting in ("RUN_NPROC=8", "RUN_MBS=2", "RUN_GBS=128", "RUN_NUM_WORKERS=0"):
        assert setting in text, setting
        assert setting in established, setting


def test_the_teacher_forced_end_weights_are_unchanged() -> None:
    assert _default("SEMANTIC_END_WEIGHT") == "0.5"
    assert _default("SEMANTIC_END_MARGIN_WEIGHT") == "0.25"
    assert _default("SEMANTIC_END_LOGIT_MARGIN") == "2.0"


def test_the_missing_decision_supervision_is_actually_opened() -> None:
    assert float(_default("ROLLIN_CONTINUE_DECISION_MARGIN_WEIGHT")) > 0.0
    assert float(_default("ROLLIN_END_WEIGHT")) > 0.0


def test_boundary_binary_is_off_because_it_replaces_the_margin_family() -> None:
    """The trainer refuses both; the binary term is an alternative, not an add-on."""

    assert float(_default("BOUNDARY_BINARY_WEIGHT")) == 0.0
    text = _text()
    assert "semantic_boundary_binary replaces the END/CONTINUE margin family" in text
    assert "margin_family_positive" in text and "binary_positive" in text


def test_the_trainer_really_enforces_that_exclusion() -> None:
    """Guard the premise behind the pre-flight check."""

    trainer = (
        REPO
        / "experiments/uniss_phase3_v4_e2e_simuls2st_pilot15_v1"
        / "training/pretrain_e2e_megatron.py"
    ).read_text(encoding="utf-8")
    assert "balanced semantic boundary calibration requires duplicate special" in trainer
    assert "active_duplicates" in trainer


def test_the_roll_in_rate_is_positive_so_the_losses_are_not_dead() -> None:
    """A roll-in weight with rate 0 selects an empty mask and stays at 0.0."""

    assert float(_default("BOUNDARY_ROLLIN_RATE")) > 0.0
    assert int(_default("BOUNDARY_ROLLIN_RAMP_UPDATES")) > 0


def test_the_launcher_refuses_a_weight_without_a_rate() -> None:
    text = _text()
    assert "the loss would be identically zero" in text
    assert "roll_in_enabled" in text and "rate_positive" in text


def test_prefix_corruption_stays_off_because_it_is_mutually_exclusive() -> None:
    assert float(_default("PREFIX_CORRUPTION_RATE")) == 0.0
    assert "RUN_SEMANTIC_CONTINUE_MARGIN_WEIGHT=0.0" in _text()
    assert "RUN_CONTENT_END_WEIGHT=0.0" in _text()


def test_the_established_run_really_left_these_terms_at_zero() -> None:
    """Guard the premise: this continuation is only interesting if they were 0."""

    established = ESTABLISHED.read_text(encoding="utf-8")
    for name in (
        "RUN_SEMANTIC_ROLLIN_END_WEIGHT",
        "RUN_SEMANTIC_ROLLIN_CONTINUE_DECISION_MARGIN_WEIGHT",
        "RUN_SEMANTIC_BOUNDARY_BINARY_WEIGHT",
        "RUN_SEMANTIC_BOUNDARY_ROLLIN_RATE",
    ):
        assert f"{name}=0.0" in established, name


def test_it_keeps_the_gpu_lock_and_the_overwrite_guards() -> None:
    text = _text()
    assert "flock -n 9" in text
    assert "GPUs are busy; refusing to interfere with PIDs" in text
    assert "refusing to overwrite output" in text
    assert "audit_frozen_stage_a" in text
