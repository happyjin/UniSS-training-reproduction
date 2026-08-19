from __future__ import annotations

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training import (
    audit_frozen_stage_a,
)


def test_frozen_stage_a_audit_reports_exact_and_changed_candidates(monkeypatch) -> None:
    values = {
        "reference": {
            "checkpoint": "reference",
            "tensors": 2,
            "bytes": 8,
            "tree_sha256": "a" * 64,
            "tensor_sha256": {"stage_a_objective.a": "1", "stage_a_objective.b": "2"},
        },
        "exact": {
            "checkpoint": "exact",
            "tensors": 2,
            "bytes": 8,
            "tree_sha256": "a" * 64,
            "tensor_sha256": {"stage_a_objective.a": "1", "stage_a_objective.b": "2"},
        },
        "changed": {
            "checkpoint": "changed",
            "tensors": 2,
            "bytes": 8,
            "tree_sha256": "b" * 64,
            "tensor_sha256": {"stage_a_objective.a": "1", "stage_a_objective.b": "3"},
        },
    }
    monkeypatch.setattr(
        audit_frozen_stage_a, "tensor_hashes", lambda checkpoint: values[str(checkpoint)]
    )
    passed = audit_frozen_stage_a.audit_frozen_stage_a(
        "reference", [("candidate", "exact")]
    )
    assert passed["status"] == "passed"
    assert passed["exact_bitwise_match"] is True
    failed = audit_frozen_stage_a.audit_frozen_stage_a(
        "reference", [("candidate", "changed")]
    )
    assert failed["status"] == "failed"
    assert failed["candidates"][0]["changed"] == ["stage_a_objective.b"]
