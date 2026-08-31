"""The post-training measurement chain must measure and never train.

The plan rule these tests enforce: a failed S2 gate is recorded as a wall, not
retried with another weight setting.  A chain that could launch training would
turn an automatic measurement into an automatic weight scan.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CHAIN = REPO_ROOT / "experiments/uniss_phase3_e2e_speak_decision_v1/scripts/wait_then_export_and_gate.sh"
BASELINE_GATE = (
    REPO_ROOT
    / "reports/uniss_phase3_v4_e2e_simuls2st_pilot15_v1/formal_gold_20260818T090515Z"
    / "free_running_gates/stage2_paced_m1200_iter0002264_20260831T180448Z"
)


@pytest.fixture(scope="module")
def chain_text() -> str:
    return CHAIN.read_text()


def _run(env_overrides: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(env_overrides)
    return subprocess.run(
        ["bash", str(CHAIN)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_the_chain_parses() -> None:
    assert subprocess.run(["bash", "-n", str(CHAIN)]).returncode == 0


def test_the_chain_never_invokes_a_training_entrypoint(chain_text: str) -> None:
    forbidden = ("torchrun", "pretrain_", "run_e2e_megatron", "run_8gpu.sh")
    for token in forbidden:
        assert token not in chain_text, f"measurement chain must not reference {token}"


def test_the_chain_reuses_the_established_gate_runner_and_converter(chain_text: str) -> None:
    assert "uniss_phase3_e2e_commit_policy_v1" in chain_text
    assert "run_gate_local_agreement_8gpu.sh" in chain_text
    assert "scripts/convert_uniss_checkpoint.sh" in chain_text
    # It must not reimplement safetensors surgery or a bespoke commit policy.
    assert "safetensors.torch" not in chain_text
    assert "StablePrefixCommitter" not in chain_text


def test_the_gate_configuration_matches_the_baseline_that_measured_iter_0002264(
    chain_text: str,
) -> None:
    """Same commit policy, same pacing, same cap: otherwise the delta is confounded."""
    assert "UNISS_E2E_MT_HOLDBACK=\"${HOLDBACK}\"" in chain_text
    assert "HOLDBACK=${HOLDBACK:-2}" in chain_text
    assert "PACE_MARGIN_MS=${PACE_MARGIN_MS:-1200}" in chain_text
    assert "UNISS_E2E_SEMANTIC_PACE=1" in chain_text
    assert "MAX_S2S_SEMANTIC_TOKENS=${MAX_S2S_SEMANTIC_TOKENS:-384}" in chain_text
    # The inference-side speak gate was falsified in S0.1; it must stay off so the
    # trained decision is what is being measured.
    assert "UNISS_E2E_CONTENT_GATED_SPEAK" not in chain_text
    assert "UNISS_E2E_EAGER_SPEAK" not in chain_text


def test_the_baseline_selection_is_copied_rather_than_regenerated(chain_text: str) -> None:
    assert 'cp "${BASELINE_GATE}/SELECTION.json" "${GATE_ROOT}/SELECTION.json"' in chain_text
    assert "selection_seed" not in chain_text, "must not resample the selection"


def test_the_baseline_gate_default_still_exists_on_disk() -> None:
    assert (BASELINE_GATE / "SELECTION.json").is_file()
    assert (BASELINE_GATE / "E2E_FREE_RUNNING_GATE.json").is_file()


def test_it_requires_both_run_ids() -> None:
    assert "TRAIN_RUN_ID is required" in _run({}).stderr
    got = _run({"TRAIN_RUN_ID": "probe"}).stderr
    assert "GATE_RUN_ID is required" in got


def test_it_refuses_to_overwrite_an_existing_gate_root() -> None:
    got = _run({"TRAIN_RUN_ID": "probe", "GATE_RUN_ID": BASELINE_GATE.name})
    assert got.returncode == 3
    assert "refusing to overwrite" in got.stderr


def test_it_rejects_a_polling_interval_that_would_hammer_nvidia_smi() -> None:
    got = _run(
        {
            "TRAIN_RUN_ID": "probe",
            "GATE_RUN_ID": "test_chain_guard_poll",
            "POLL_SECONDS": "5",
        }
    )
    assert got.returncode == 2
    assert "POLL_SECONDS must be an integer of at least 10" in got.stderr


def test_it_gives_up_instead_of_gating_a_run_that_never_completed(tmp_path: Path) -> None:
    gate_run_id = "test_chain_guard_timeout"
    log = REPO_ROOT / "logs/uniss_phase3_e2e_speak_decision_v1" / f"{gate_run_id}.chain.log"
    log.unlink(missing_ok=True)
    try:
        got = _run(
            {
                "TRAIN_RUN_ID": "no_such_training_run",
                "GATE_RUN_ID": gate_run_id,
                "MAX_WAIT_SECONDS": "0",
                "POLL_SECONDS": "10",
            }
        )
        assert got.returncode == 4
        assert "did not report complete" in got.stdout + got.stderr
    finally:
        log.unlink(missing_ok=True)


def test_it_refuses_to_gate_a_training_run_that_reported_a_non_complete_status(
    tmp_path: Path,
) -> None:
    """The status check is the reason a crashed run does not silently get gated."""
    import json

    run_id = "test_chain_guard_status"
    report_root = REPO_ROOT / "reports/uniss_phase3_e2e_speak_decision_v1" / run_id
    summary = report_root / "SPEAK_DECISION_RUN.json"
    gate_run_id = "test_chain_guard_status_gate"
    log = REPO_ROOT / "logs/uniss_phase3_e2e_speak_decision_v1" / f"{gate_run_id}.chain.log"
    report_root.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps({"status": "failed", "final_checkpoint": "/dev/null"}))
    log.unlink(missing_ok=True)
    try:
        got = _run({"TRAIN_RUN_ID": run_id, "GATE_RUN_ID": gate_run_id})
        assert got.returncode != 0
        assert "refusing to gate" in got.stdout + got.stderr
    finally:
        log.unlink(missing_ok=True)
        summary.unlink(missing_ok=True)
        try:
            report_root.rmdir()
        except OSError:
            pass
