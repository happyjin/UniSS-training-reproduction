from __future__ import annotations

import json
import tempfile
from argparse import Namespace
from pathlib import Path

import pytest

from experiments.uniss_phase3_event_rollout_joint_pilot15_v1.data.build_training_manifest import (
    _require_data_audit,
)


def _audit(path: Path, status: str, *, complete: bool) -> Path:
    gates = {
        "fixed_shards_only": True,
        "deterministic_split": True,
        "train_valid_intersection_zero": True,
        "phase3_replay_fixed15_only": True,
        "complete_160ms_sessions": complete,
        "gap_free_text_and_semantic_coverage": complete,
        "sessions_never_cross_packs": complete,
        "runtime_parsers_accept_all_packs": complete,
    }
    path.write_text(json.dumps({"status": status, "gates": gates}), encoding="utf-8")
    return path


def test_formal_manifest_requires_full_data_audit() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        sampled = _audit(root / "sampled.json", "sampled_pass", complete=False)
        with pytest.raises(ValueError, match="formal-pass"):
            _require_data_audit(sampled, allow_sampled=False)
        assert _require_data_audit(sampled, allow_sampled=True)["status"] == "sampled_pass"


def test_full_audit_requires_all_session_and_runtime_gates() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        incomplete = _audit(root / "incomplete.json", "pass", complete=False)
        with pytest.raises(ValueError, match="session/runtime"):
            _require_data_audit(incomplete, allow_sampled=False)
        complete = _audit(root / "complete.json", "pass", complete=True)
        assert _require_data_audit(complete, allow_sampled=False)["status"] == "pass"
