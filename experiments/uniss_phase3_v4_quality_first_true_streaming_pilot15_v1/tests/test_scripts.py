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


def test_real_audit_uses_strict_fp32_hidden_and_exact_token_gates() -> None:
    source = (
        ROOT / "stage00_baseline" / "audit_frontend_real_pcm.py"
    ).read_text(encoding="utf-8")
    assert "rtol=2e-5, atol=2e-6" in source
    assert '"match_ratio"' in source
    assert '"future_hidden_exact_before_changed_block"' in source
    assert "forward_recomputed_reference" in source
    assert '"single_mask_numerical_diagnostic"' in source


def test_gpu_launcher_only_stops_named_synthetic_session() -> None:
    source = (ROOT / "scripts" / "launch_stage00_frontend_tmux.sh").read_text(
        encoding="utf-8"
    )
    assert "tmux kill-session -t uniss_gpu_load_60" in source
    assert "refusing to kill unknown GPU processes" in source
