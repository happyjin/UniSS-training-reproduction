from experiments.uniss_phasea_coverage_constrained_grpo_v3.data.audit_event_cache import (
    structural_violations,
)


def valid_row():
    return {
        "natural_action_target": "WRITE",
        "deadline_action_target": "WRITE",
        "deadline_forced_target": False,
        "previous_committed_length": 0,
        "stable_target_length": 1,
        "new_supported_count": 1,
        "translation_ids": [10, 11],
        "safe_commit_mask": [True, False],
        "target_text_delta_ids": [10],
        "chunk_end_ms": 320,
        "source_duration_ms": 1000,
        "soft_deadline_ms": 640,
        "hard_deadline_ms": 800,
    }


def test_structural_audit_accepts_consistent_write():
    assert structural_violations(valid_row()) == []


def test_structural_audit_rejects_future_delta_geometry():
    row = valid_row()
    row["target_text_delta_ids"] = [11]
    assert "delta_slice_mismatch" in structural_violations(row)

