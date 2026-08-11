"""Schema and indexed reader for exact deployment frontend traces."""

from __future__ import annotations

import json
from pathlib import Path

from training.simul_uniss.jsonl_index import load_index


TRACE_SCHEMA = "uniss_runtime_exact_frontend_trace_v1"


class ExactRuntimeTraceReader:
    def __init__(self, manifest: str | Path) -> None:
        self.manifest = Path(manifest).resolve()
        offsets = load_index(self.manifest)
        if offsets is None:
            raise ValueError(f"missing trace index for {self.manifest}")
        self.offsets = offsets
        self._handle = self.manifest.open("rb")
        self._source_index_to_row: dict[int, int] | None = None

    def __len__(self) -> int:
        return len(self.offsets)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "ExactRuntimeTraceReader":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def record(self, index: int) -> dict[str, object]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        self._handle.seek(int(self.offsets[index]))
        value = json.loads(self._handle.readline())
        if value.get("schema_version") != TRACE_SCHEMA:
            raise ValueError(f"unexpected runtime trace schema at row {index}")
        return value

    def source_index_lookup(self) -> dict[int, int]:
        if self._source_index_to_row is None:
            lookup: dict[int, int] = {}
            for row_index in range(len(self)):
                source_index = int(self.record(row_index)["source_index"])
                if source_index in lookup:
                    raise ValueError(
                        f"duplicate formal source index in runtime trace: {source_index}"
                    )
                lookup[source_index] = row_index
            self._source_index_to_row = lookup
        return self._source_index_to_row

    def record_for_source_index(self, source_index: int) -> dict[str, object]:
        try:
            row_index = self.source_index_lookup()[int(source_index)]
        except KeyError as error:
            raise KeyError(
                f"formal source index {source_index} is absent from runtime trace"
            ) from error
        return self.record(row_index)

    def tokens_and_times(
        self, index: int, *, expected_id: str | None = None
    ) -> tuple[list[int], list[int]]:
        value = self.record(index)
        if expected_id is not None and str(value.get("id")) != str(expected_id):
            raise ValueError(
                f"formal/runtime trace IDs differ at {index}: "
                f"{expected_id!r} != {value.get('id')!r}"
            )
        tokens = [int(item) for item in value["runtime_source_glm"]]  # type: ignore[index]
        times = [int(item) for item in value["runtime_source_glm_commit_ms"]]  # type: ignore[index]
        if not tokens or len(tokens) != len(times):
            raise ValueError(f"malformed runtime trace at row {index}")
        if int(value.get("committed_revision_violations", -1)) != 0:
            raise ValueError(f"runtime trace contains committed revisions at row {index}")
        return tokens, times

    def tokens_and_times_for_source_index(
        self, source_index: int, *, expected_id: str | None = None
    ) -> tuple[list[int], list[int]]:
        value = self.record_for_source_index(source_index)
        if expected_id is not None and str(value.get("id")) != str(expected_id):
            raise ValueError(
                f"formal/runtime trace IDs differ at source {source_index}: "
                f"{expected_id!r} != {value.get('id')!r}"
            )
        tokens = [int(item) for item in value["runtime_source_glm"]]  # type: ignore[index]
        times = [int(item) for item in value["runtime_source_glm_commit_ms"]]  # type: ignore[index]
        if not tokens or len(tokens) != len(times):
            raise ValueError(f"malformed runtime trace at source {source_index}")
        if int(value.get("committed_revision_violations", -1)) != 0:
            raise ValueError(
                f"runtime trace contains committed revisions at source {source_index}"
            )
        return tokens, times


__all__ = ["ExactRuntimeTraceReader", "TRACE_SCHEMA"]
