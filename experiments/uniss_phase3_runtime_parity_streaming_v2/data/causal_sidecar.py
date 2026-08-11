"""Indexed access to the existing Stage-A-v3 causal WhisperVQ clone sidecar."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import torch

from training.simul_uniss.jsonl_index import load_index


SIDECAR_SCHEMA = "simul_uniss_stage_a_v3_causal_sidecar_v1"


def runtime_commit_end_times(
    duration_ms: int,
    token_count: int,
    *,
    chunk_ms: int = 160,
    right_context_ms: int = 80,
    token_hop_ms: int = 80,
) -> list[int]:
    """Return when each causal token is stable under the deployment commit rule.

    The deployed frontend commits complete 160 ms token blocks only after the
    80 ms right context has arrived.  With an 80 ms token hop, tokens 0/1 are
    first visible at 320 ms, tokens 2/3 at 480 ms, and the final incomplete
    block is flushed at source EOS.
    """

    if duration_ms <= 0 or token_count <= 0:
        raise ValueError("duration_ms and token_count must be positive")
    if chunk_ms <= 0 or chunk_ms % token_hop_ms:
        raise ValueError("chunk_ms must be a positive token-hop multiple")
    if right_context_ms < 0 or right_context_ms % token_hop_ms:
        raise ValueError("right_context_ms must be a token-hop multiple")
    tokens_per_chunk = chunk_ms // token_hop_ms
    values = [
        min(
            int(duration_ms),
            math.ceil(
                (
                    (index // tokens_per_chunk + 1) * chunk_ms
                    + right_context_ms
                )
                / chunk_ms
            )
            * chunk_ms,
        )
        for index in range(int(token_count))
    ]
    # Very short utterances flush all remaining tokens at EOS.  For ordinary
    # utterances the min() above preserves monotonicity and the exact final
    # flush clock.
    return values


class CausalCloneSidecarReader:
    """Read one causal token row while caching only the current tensor shard."""

    def __init__(self, manifest: str | Path) -> None:
        self.manifest = Path(manifest).resolve()
        offsets = load_index(self.manifest)
        if offsets is None:
            raise ValueError(f"missing JSONL index for {self.manifest}")
        self.offsets = offsets
        self._handle = self.manifest.open("rb")
        self._shard_path: Path | None = None
        self._shard: Mapping[str, object] | None = None

    def __len__(self) -> int:
        return len(self.offsets)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "CausalCloneSidecarReader":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def row(self, index: int) -> dict[str, object]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        self._handle.seek(int(self.offsets[index]))
        value = json.loads(self._handle.readline())
        if value.get("schema_version") != SIDECAR_SCHEMA:
            raise ValueError(f"unexpected causal sidecar schema at row {index}")
        if int(value["source_manifest_index"]) != index:
            raise ValueError(
                f"causal sidecar index mismatch: row={index}, "
                f"source={value['source_manifest_index']}"
            )
        return value

    def tokens(self, index: int, *, expected_id: str | None = None) -> list[int]:
        row = self.row(index)
        if expected_id is not None and str(row.get("id")) != str(expected_id):
            raise ValueError(
                f"formal/causal IDs differ at {index}: "
                f"{expected_id!r} != {row.get('id')!r}"
            )
        shard_path = Path(str(row["shard_path"])).resolve()
        if shard_path != self._shard_path:
            if not shard_path.is_file():
                raise FileNotFoundError(shard_path)
            self._shard = torch.load(
                shard_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            self._shard_path = shard_path
        if self._shard is None:
            raise AssertionError("causal tensor shard was not loaded")
        start, end = int(row["target_start"]), int(row["target_end"])
        values = self._shard["target_tokens"]
        if not isinstance(values, torch.Tensor):
            raise TypeError("causal shard target_tokens is not a tensor")
        result = [int(value) for value in values[start:end].tolist()]
        if not result:
            raise ValueError(f"empty causal token row at index {index}")
        return result


__all__ = [
    "CausalCloneSidecarReader",
    "SIDECAR_SCHEMA",
    "runtime_commit_end_times",
]
