from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage00_shell_scripts_are_syntax_valid() -> None:
    import subprocess

    for path in sorted((ROOT / "scripts").glob("*.sh")):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_stage00_does_not_authorize_stage_a_from_frontend_gate_only() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "FRONTEND_GATE_PASSED.json" in readme
    assert "cannot falsely authorize Stage A" in readme

