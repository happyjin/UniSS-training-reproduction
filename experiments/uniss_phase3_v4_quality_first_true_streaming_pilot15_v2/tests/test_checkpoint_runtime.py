from pathlib import Path

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.checkpoint_runtime import (
    append_only_commit_audit,
    cache_growth_is_valid,
    resolve_iteration_checkpoint,
)


def test_append_only_commit_audit_tracks_irreversible_ledger() -> None:
    result = append_only_commit_audit(
        [
            {"predicted_tokens": [10, 11]},
            {"predicted_tokens": [12]},
            {"predicted_tokens": []},
        ]
    )
    assert result["events"] == 3
    assert result["committed_tokens"] == 3
    assert result["rollback_count"] == 0
    assert result["append_only"] is True
    assert [row["tokens_after"] for row in result["snapshots"]] == [2, 3, 3]


def test_cache_growth_accepts_normal_growth_and_explicit_reset() -> None:
    assert cache_growth_is_valid([8, 16, 24], [])
    assert cache_growth_is_valid([8, 16, 8, 16], [2])
    assert not cache_growth_is_valid([8, 24], [])
    assert not cache_growth_is_valid([8, 16, 16], [])


def test_resolve_iteration_checkpoint_reads_latest(tmp_path: Path) -> None:
    iteration = tmp_path / "iter_0000032"
    iteration.mkdir()
    (iteration / ".metadata").write_bytes(b"metadata")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("32\n")
    assert resolve_iteration_checkpoint(tmp_path) == iteration.resolve()
    assert resolve_iteration_checkpoint(iteration) == iteration.resolve()
