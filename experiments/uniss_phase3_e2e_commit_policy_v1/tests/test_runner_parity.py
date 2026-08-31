"""The isolated gate runner must stay a near-verbatim copy of the established one.

If the E2E gate script gains a guard, a fingerprint check or an extra argument,
this test fails and the copy has to be refreshed.  Without it the two runners
would silently drift and the comparison would stop being apples-to-apples.
"""

from __future__ import annotations

import difflib
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
ESTABLISHED = (
    REPO
    / "experiments/uniss_phase3_v4_e2e_simuls2st_pilot15_v1"
    / "scripts/run_free_running_gate_8gpu.sh"
)
ISOLATED = (
    REPO
    / "experiments/uniss_phase3_e2e_commit_policy_v1"
    / "scripts/run_gate_local_agreement_8gpu.sh"
)
WORKER_MODULE = (
    "experiments.uniss_phase3_e2e_commit_policy_v1"
    ".evaluation.run_worker_local_agreement"
)


def _changed_lines() -> tuple[list[str], list[str]]:
    left = ESTABLISHED.read_text(encoding="utf-8").splitlines()
    right = ISOLATED.read_text(encoding="utf-8").splitlines()
    removed = [
        line[1:]
        for line in difflib.unified_diff(left, right, lineterm="", n=0)
        if line.startswith("-") and not line.startswith("---")
    ]
    added = [
        line[1:]
        for line in difflib.unified_diff(left, right, lineterm="", n=0)
        if line.startswith("+") and not line.startswith("+++")
    ]
    return removed, added


def test_both_runners_exist() -> None:
    assert ESTABLISHED.is_file()
    assert ISOLATED.is_file()


def test_only_the_worker_module_and_experiment_dir_differ() -> None:
    removed, added = _changed_lines()
    assert len(removed) == 2, removed
    assert len(added) == 3, added
    assert any("run_worker" in line for line in removed)
    assert any(WORKER_MODULE in line for line in added)
    assert any("EXPERIMENT_DIR" in line for line in removed)


def test_the_isolated_runner_invokes_the_patched_worker() -> None:
    text = ISOLATED.read_text(encoding="utf-8")
    assert WORKER_MODULE in text
    assert "evaluation.run_worker \\" not in text


def test_the_isolated_runner_keeps_the_overwrite_and_gpu_guards() -> None:
    text = ISOLATED.read_text(encoding="utf-8")
    assert "refusing to overwrite free-running gate run" in text
    assert "GPUs are busy; refusing to interfere with PIDs" in text
    assert "formal free-running gate requires eight GPU workers" in text
    assert "malformed candidate HF fingerprint" in text
