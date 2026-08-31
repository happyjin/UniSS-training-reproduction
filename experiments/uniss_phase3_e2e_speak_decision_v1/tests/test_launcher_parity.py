"""The vendored megatron launcher must stay a two-line copy of the established one."""

from __future__ import annotations

import difflib
import re
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "experiments/uniss_phase3_v4_e2e_simuls2st_pilot15_v1"
ESTABLISHED = BASE / "scripts/run_e2e_megatron.sh"
VENDORED = (
    REPO
    / "experiments/uniss_phase3_e2e_speak_decision_v1/scripts/run_e2e_megatron_speak.sh"
)
LAUNCHER = (
    REPO / "experiments/uniss_phase3_e2e_speak_decision_v1/scripts/run_8gpu.sh"
)


def _diff() -> tuple[list[str], list[str]]:
    left = ESTABLISHED.read_text(encoding="utf-8").splitlines()
    right = VENDORED.read_text(encoding="utf-8").splitlines()
    removed, added = [], []
    for line in difflib.unified_diff(left, right, lineterm="", n=0):
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("-"):
            removed.append(line[1:])
        elif line.startswith("+"):
            added.append(line[1:])
    return removed, added


def test_every_script_parses() -> None:
    for script in (VENDORED, LAUNCHER):
        assert script.is_file(), script
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_only_the_entrypoint_and_experiment_dir_differ() -> None:
    removed, added = _diff()
    assert len(removed) == 2, removed
    assert len(added) == 3, added
    assert any("pretrain_e2e_megatron.py" in line for line in removed)
    assert any("pretrain_speak_decision_megatron.py" in line for line in added)
    assert any("EXPERIMENT_DIR=" in line for line in removed)


def test_the_vendored_copy_still_sources_the_established_environment() -> None:
    text = VENDORED.read_text(encoding="utf-8")
    assert 'source "${EXPERIMENT_DIR}/experiment.env"' in text
    assert "uniss_phase3_v4_e2e_simuls2st_pilot15_v1" in text


def test_the_vendored_copy_keeps_the_established_geometry_and_guards() -> None:
    text = VENDORED.read_text(encoding="utf-8")
    for token in ("--nproc_per_node", "--master_port", "--sft", "torchrun"):
        assert token in text, token


def test_the_launcher_uses_the_vendored_copy_not_the_established_one() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "run_e2e_megatron_speak.sh" in text
    assert 'BASE_EXPERIMENT}/scripts/run_e2e_megatron.sh' not in text


def _default(name: str) -> str:
    match = re.search(
        rf"^{name}=\$\{{{name}:-([^}}]*)\}}",
        LAUNCHER.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match, f"{name} has no defaulted assignment"
    return match.group(1)


def test_the_measured_decisionrow_configuration_is_preserved() -> None:
    """S0.2 measured this exact set as the best; the run must not perturb it."""

    assert _default("SEMANTIC_END_WEIGHT") == "0.5"
    assert _default("SEMANTIC_END_MARGIN_WEIGHT") == "0.25"
    assert _default("SEMANTIC_END_LOGIT_MARGIN") == "2.0"
    assert _default("ROLLIN_END_WEIGHT") == "0.25"
    assert _default("ROLLIN_CONTINUE_DECISION_MARGIN_WEIGHT") == "0.25"
    assert _default("ROLLIN_CONTINUE_DECISION_LOGIT_MARGIN") == "1.0"
    assert _default("BOUNDARY_ROLLIN_RATE") == "0.5"


def test_the_two_new_weights_are_actually_on() -> None:
    assert float(_default("SPEAK_DECISION_WEIGHT")) > 0.0
    assert float(_default("REPETITION_WEIGHT")) > 0.0
    assert int(_default("REPETITION_WINDOW")) >= 1


def test_boundary_binary_stays_off_because_it_replaces_the_margin_family() -> None:
    assert float(_default("BOUNDARY_BINARY_WEIGHT")) == 0.0


def test_the_parent_is_the_epoch3_checkpoint() -> None:
    assert _default("PARENT_RUN_ID") == "endmargin_epoch23_15shard_20260824T190227Z"
    assert _default("PARENT_ITER") == "0002264"


def test_outputs_land_in_this_experiments_namespace() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "OWN_NAME=uniss_phase3_e2e_speak_decision_v1" in text
    for line in text.splitlines():
        if line.startswith(("RUN_SAVE_DIR=", "RUN_TENSORBOARD_DIR=", "RUN_LOG=", "RUN_REPORT_ROOT=")):
            assert "${OWN_NAME}" in line, line
