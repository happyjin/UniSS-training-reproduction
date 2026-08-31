"""CPU unit tests for the 0-A bridge parity diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.uniss_phase3_content_first_diagnostics_v1.diagnostics import (
    bridge_parity as bp,
)


def test_agreement_counts_prefix_matches() -> None:
    result = bp.agreement([1, 2, 3, 4], [1, 2, 9, 4])
    assert result["compared"] == 4
    assert result["matches"] == 3
    assert result["agreement"] == 0.75
    assert result["length_equal"] is True


def test_agreement_handles_unequal_lengths() -> None:
    result = bp.agreement([1, 2, 3], [1, 2])
    assert result["compared"] == 2
    assert result["matches"] == 2
    assert result["agreement"] == 1.0
    assert result["length_equal"] is False


def test_agreement_handles_empty() -> None:
    result = bp.agreement([], [1, 2])
    assert result["compared"] == 0
    assert result["agreement"] == 0.0


def test_nearest_codes_matches_reference_argmin() -> None:
    torch.manual_seed(0)
    codebook = torch.randn(64, 16)
    hidden = torch.randn(7, 16)
    expected = torch.cdist(hidden, codebook).argmin(dim=1)
    actual = bp.nearest_codes(codebook, hidden)
    assert torch.equal(actual, expected)


def test_nearest_codes_is_exact_for_codebook_rows() -> None:
    codebook = torch.eye(8)
    hidden = codebook[[3, 5, 0]]
    assert bp.nearest_codes(codebook, hidden).tolist() == [3, 5, 0]


def test_nearest_codes_handles_chunk_boundary() -> None:
    torch.manual_seed(1)
    codebook = torch.randn(32, 4)
    hidden = torch.randn(5000, 4)
    expected = torch.cdist(hidden, codebook).argmin(dim=1)
    assert torch.equal(bp.nearest_codes(codebook, hidden), expected)


def test_embedding_similarity_is_one_for_identical_codes() -> None:
    codebook = torch.randn(16, 8)
    metrics = bp.embedding_similarity(codebook, [1, 2, 3], [1, 2, 3])
    assert metrics["mean_cosine"] > 0.999
    assert metrics["p05_cosine"] > 0.999


def test_unique_components_is_deterministic_and_deduplicated(tmp_path: Path) -> None:
    def component(sample_id: str, index: int) -> dict[str, object]:
        return {
            "sample_id": sample_id,
            "source_audio": f"/tmp/{sample_id}.wav",
            "duration_ms": 1000 + index,
            "source_glm_length": 10 + index,
        }

    episodes = tmp_path / "episodes.jsonl"
    with episodes.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "episode_id": "e0",
                    "src_lang": "cmn",
                    "components": [component("a", 0), component("b", 1)],
                }
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {
                    "episode_id": "e1",
                    "src_lang": "eng",
                    "components": [component("b", 1), component("c", 2)],
                }
            )
            + "\n"
        )
    selected = bp.unique_components(episodes, 10)
    assert [row["sample_id"] for row in selected] == ["a", "b", "c"]
    assert selected[1]["episode_id"] == "e0"
    assert bp.unique_components(episodes, 2) == selected[:2]
