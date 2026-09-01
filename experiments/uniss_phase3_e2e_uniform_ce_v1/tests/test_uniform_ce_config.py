"""One weight changes, one family retires, and nothing else moves.

Three runs have now moved the inference-time speak decision monotonically the
wrong way -- -2.88, -3.75, -4.97 -- so this experiment retires the margin family
and instead stops starving the decision tokens: boundary is 32.8% of supervised
tokens in the interleaved family and receives 4.7% of the gradient at
boundary_eos 0.10, a value no script in this repository has ever changed.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
OWN = REPO_ROOT / "experiments/uniss_phase3_e2e_uniform_ce_v1/scripts"
BASE = REPO_ROOT / "experiments/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/scripts"
LAUNCHER = OWN / "run_8gpu.sh"
ENTRYPOINT = OWN / "run_e2e_megatron_uniform.sh"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_both_scripts_parse() -> None:
    for path in (LAUNCHER, ENTRYPOINT):
        assert subprocess.run(["bash", "-n", str(path)]).returncode == 0, path


def test_the_entrypoint_differs_from_the_established_one_in_three_places() -> None:
    """Redirect the experiment directory, then declare and pass one argument.

    The redirect is required because this copy sits in a sibling experiment that
    has no experiment.env of its own; everything else -- the environment, the
    trainer, every path -- still resolves to the established experiment.
    """
    established = BASE / "run_e2e_megatron.sh"
    diff = subprocess.run(
        ["diff", str(established), str(ENTRYPOINT)], capture_output=True, text=True
    )
    hunks = [line for line in diff.stdout.splitlines() if re.match(r"^\d", line)]
    assert len(hunks) == 3, diff.stdout
    added = [line for line in diff.stdout.splitlines() if line.startswith("> ")]
    assert any("RUN_BOUNDARY_EOS_WEIGHT=" in line for line in added)
    assert any("--e2e-boundary-eos-weight" in line for line in added)
    assert any("uniss_phase3_v4_e2e_simuls2st_pilot15_v1" in line for line in added)


def test_boundary_eos_is_raised_to_one() -> None:
    assert "BOUNDARY_EOS_WEIGHT=${BOUNDARY_EOS_WEIGHT:-1.0}" in read(LAUNCHER)
    assert 'RUN_BOUNDARY_EOS_WEIGHT="${BOUNDARY_EOS_WEIGHT}"' in read(LAUNCHER)


def test_every_margin_and_rollin_weight_is_zero() -> None:
    text = read(LAUNCHER)
    for name in (
        "SEMANTIC_END_WEIGHT",
        "SEMANTIC_END_MARGIN_WEIGHT",
        "ROLLIN_END_WEIGHT",
        "ROLLIN_END_MARGIN_WEIGHT",
        "ROLLIN_CONTINUE_DECISION_MARGIN_WEIGHT",
        "ROLLIN_CONTINUE_MARGIN_WEIGHT",
        "CONTINUE_MARGIN_WEIGHT",
        "BOUNDARY_BINARY_WEIGHT",
        "BOUNDARY_ROLLIN_RATE",
        "PREFIX_CORRUPTION_RATE",
    ):
        assert f"{name}=${{{name}:-0.0}}" in text or f"{name}=${{{name}:-0}}" in text, name


def test_no_objective_extension_is_installed() -> None:
    """This run uses the established trainer unchanged -- no new Python at all."""
    text = read(LAUNCHER)
    for token in (
        "UNISS_E2E_",
        "objective_ext",
        "pretrain_speak_decision",
        "pretrain_continue_end",
        "continue_after_fragment",
        "repetition_penalty",
    ):
        assert token not in text, token
    assert "run_e2e_megatron_uniform.sh" in text
    entry = read(ENTRYPOINT)
    assert "training/pretrain_e2e_megatron.py" in entry, "must use the established trainer"


def test_the_distillation_anchors_are_kept() -> None:
    """Both KLs measurably fall during training; this run changes one thing."""
    text = read(LAUNCHER)
    for name in ("V1_ASR_KL", "PHASE3_KL", "REPLAY", "COMMIT"):
        assert f"{name}_WEIGHT=" not in text, (
            f"{name} must stay at the trainer's established default, not be overridden"
        )


def test_the_parent_is_iter_0002264() -> None:
    """-2.88 is the reference point every measurement in this lineage is against."""
    text = read(LAUNCHER)
    assert "PARENT_RUN_ID=${PARENT_RUN_ID:-endmargin_epoch23_15shard_20260824T190227Z}" in text
    assert "PARENT_ITER=${PARENT_ITER:-0002264}" in text


def test_dataloader_workers_are_not_zero() -> None:
    """The dataset reopens the file inside __getitem__, so workers are safe.

    Every continuation launcher in this lineage hardcoded 0 while the
    established default is 8, and the parent run ran at 47% mean GPU
    utilisation with 46% of samples at 0%.
    """
    text = read(LAUNCHER)
    assert "RUN_NUM_WORKERS=0" not in text
    assert 'RUN_NUM_WORKERS="${NUM_WORKERS:-8}"' in text


def test_the_dataset_holds_no_file_handle_across_getitem() -> None:
    """The precondition for the previous test, checked in the source."""
    source = (
        REPO_ROOT
        / "experiments/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/training/runtime_dataset.py"
    ).read_text(encoding="utf-8")
    body = source[source.index("def __getitem__") :]
    body = body[: body.index("\n    def ", 1)] if "\n    def " in body[1:] else body
    assert "with self.path.open(" in body, "workers are only safe if the file is reopened"


def test_geometry_and_data_are_unchanged_from_the_parent_run() -> None:
    """Same pool, same batch geometry, same seed: the result must be comparable."""
    text = read(LAUNCHER)
    assert "TASK_POOL_RUN_ID=${TASK_POOL_RUN_ID:-task_pool_formal_p4_20260820T154500Z}" in text
    assert "RUN_MBS=2" in text and "RUN_GBS=128" in text
    assert "COVERAGE_EPOCHS=${COVERAGE_EPOCHS:-1}" in text
