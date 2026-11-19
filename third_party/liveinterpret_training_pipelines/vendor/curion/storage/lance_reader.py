"""Row-address-native reader for the Lance two-layer architecture (layer 1).

This is the read side of ``processed.lance`` — the single physical dataset that
replaces the ``index + shards + features`` split (see the Lance two-layer design
doc). Its whole reason for existing is ``take(row_ids)`` random access plus
predicate pushdown, with no cross-file byte-pointer jump.

Boundary with :mod:`curion.storage.shard_reader`
------------------------------------------------
The existing :class:`~curion.storage.shard_reader.ShardReader` protocol is
``read_payload(shard_uri, byte_offset, byte_length) -> bytes`` — a *byte-pointer*
contract, which is precisely the ``index -> tar`` indirection the two-layer
design removes. Lance addresses data by *row id*, not ``(uri, offset, length)``,
so it deliberately does **not** implement that protocol and is **not** registered
in :class:`ShardReaderRegistry`. The two readers coexist: WDS tar via the byte
pointer, Lance via row ids. The ``shard_format="lance"`` registry stub is left as
an unsupported stub, reserved for a future "oversized-audio pointer fallback".

Stable row ids vs row addresses
-------------------------------
When the dataset was created with ``enable_stable_row_ids=True`` (see
:mod:`curion.storage.lance_builder`), the ``_rowid`` column is a *stable* id that
survives compaction. :meth:`LanceRowReader.take_by_stable_rowid` uses it, so an
exact sample list (design "situation 2") stays correct across ``compact_files``.
Positional ``take`` (row address) is faster but drifts after compaction; it is
exposed as :meth:`take_by_address` only for the within-version fast path.

This module depends only on ``pylance`` and ``pyarrow`` (declared as the optional
``lance`` extra of ``curion-storage``); it does not import ``curion.core``.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Iterable, Iterator, Sequence

import lance
import pyarrow as pa

# A WDS-style byte reader: (shard_uri, byte_offset, byte_length) -> bytes.
# In production this is ObjectStoreWdsTarShardReader.read_payload; injected so
# this module stays free of curion.core / object-store imports.
ByteReader = Callable[[str, int, int], bytes]

# Chunk size for ``_rowid IN (...)`` retrieval. Keeps the filter expression and
# the coalesced read a sane size while still being one scan per chunk.
_ROWID_IN_CHUNK = 2048


class LanceRowReader:
    """Read rows from a ``processed.lance`` dataset by row id / predicate.

    Parameters
    ----------
    dataset_uri:
        Local path or ``s3://`` URI of the Lance dataset.
    version:
        Optional Lance version to pin (task reproducibility). ``None`` reads the
        latest version.
    storage_options:
        Optional object-store options forwarded to ``lance.dataset`` (e.g.
        ``{"region": "ap-southeast-3"}`` for S3).
    """

    def __init__(
        self,
        dataset_uri: str,
        *,
        version: int | None = None,
        storage_options: dict | None = None,
    ) -> None:
        self._uri = dataset_uri
        self._version = version
        self._ds = lance.dataset(
            dataset_uri, version=version, storage_options=storage_options
        )

    # -- introspection -----------------------------------------------------

    @property
    def dataset(self) -> lance.LanceDataset:
        return self._ds

    @property
    def version(self) -> int:
        return self._ds.version

    @property
    def schema(self) -> pa.Schema:
        """The Lance dataset's Arrow schema (for column-existence checks)."""
        return self._ds.schema

    def has_column(self, name: str) -> bool:
        """Check whether a column exists in the dataset schema."""
        return name in self._ds.schema.names

    def count_rows(self, filter: str | None = None) -> int:
        return self._ds.count_rows(filter=filter)

    @property
    def has_stable_row_ids(self) -> bool:
        return bool(getattr(self._ds, "has_stable_row_ids", False))

    # -- layer-1 reads -----------------------------------------------------

    def scan(
        self,
        *,
        filter: str | None = None,
        columns: Sequence[str] | None = None,
        with_row_id: bool = False,
    ) -> pa.Table:
        """Predicate-pushdown scan (design "situation 1": predicate reproducible).

        With a scalar index on the filtered columns this is index-accelerated;
        without one Lance falls back to a full scan.
        """
        return self._ds.to_table(
            columns=list(columns) if columns is not None else None,
            filter=filter,
            with_row_id=with_row_id,
        )

    def stable_row_ids(self, *, filter: str | None = None) -> list[int]:
        """Return the stable ``_rowid`` values (optionally matching ``filter``)."""
        tbl = self._ds.to_table(columns=[], filter=filter, with_row_id=True)
        return tbl.column("_rowid").to_pylist()

    def take_by_stable_rowid(
        self,
        rowids: Sequence[int],
        *,
        columns: Sequence[str] | None = None,
    ) -> pa.Table:
        """Fetch rows by *stable* ``_rowid`` (compaction-safe, design "situation 2").

        Rows whose id no longer exists (e.g. deleted) are simply absent from the
        result — no error, no drift. Order is not guaranteed to match ``rowids``;
        the ``_rowid`` column is included so callers can re-align if needed.
        """
        col_list = list(columns) if columns is not None else None
        # Sort so ``_rowid IN (...)`` hits the same fragment contiguously.
        # Callers already re-align by ``_rowid``; order is not a contract.
        ordered = sorted({int(r) for r in rowids})
        parts: list[pa.Table] = []
        for chunk in _chunked(ordered, _ROWID_IN_CHUNK):
            if not chunk:
                continue
            in_list = ",".join(str(int(r)) for r in chunk)
            parts.append(
                self._ds.to_table(
                    columns=col_list,
                    filter=f"_rowid IN ({in_list})",
                    with_row_id=True,
                )
            )
        if not parts:
            # Empty result with the right schema.
            return self._ds.to_table(
                columns=col_list, filter="_rowid IN (-1)", with_row_id=True
            )
        return pa.concat_tables(parts)

    def take_by_address(
        self,
        indices: Sequence[int],
        *,
        columns: Sequence[str] | None = None,
    ) -> pa.Table:
        """Positional ``take`` by row address (fast within-version path).

        WARNING: row addresses drift after ``compact_files``. Use
        :meth:`take_by_stable_rowid` for anything pinned across versions.
        """
        return self._ds.take(
            list(indices), columns=list(columns) if columns is not None else None
        )

    # -- shuffle planning (dataloader) -------------------------------------

    def plan_shuffle(
        self,
        *,
        seed: int,
        filter: str | None = None,
        rowids: Sequence[int] | None = None,
        n_bins_hint: int | None = None,
    ) -> list[int]:
        """Plan a length-bucketed shuffle, returning stable ``_rowid`` order.

        Reads only the ``bin_id`` column (cheap — avoids scanning the full table
        every epoch), groups rows by ``bin_id``, shuffles within each bin and the
        bin order, then concatenates. Deterministic in ``seed``.

        The returned ids are stable ``_rowid`` values, so feeding them to
        :meth:`iter_batches` is compaction-safe.

        Parameters
        ----------
        filter:
            Predicate to select rows (situation 1). Use this when shuffling the
            full task. Mutually exclusive with ``rowids``.
        rowids:
            A pre-resolved list of stable ``_rowid`` values (e.g. the train
            split from :func:`resolve_split`). Use this to shuffle only a subset.
            Mutually exclusive with ``filter``.
        """
        if filter is not None and rowids is not None:
            raise ValueError("pass either `filter` or `rowids`, not both")
        # If both are None, scan all rows (no filter) — backward compat.

        cols = ["bin_id"]
        if rowids is not None:
            # Shuffle a pre-resolved subset (e.g. train split after resolve_split).
            tbl = self.take_by_stable_rowid(rowids, columns=cols)
        else:
            tbl = self._ds.to_table(columns=cols, filter=filter, with_row_id=True)
        rowids_out = tbl.column("_rowid").to_pylist()
        bins = tbl.column("bin_id").to_pylist()

        by_bin: dict[int, list[int]] = defaultdict(list)
        for rid, b in zip(rowids_out, bins):
            by_bin[-1 if b is None else int(b)].append(rid)

        rng = random.Random(seed)
        order = sorted(by_bin.keys())
        rng.shuffle(order)
        out: list[int] = []
        for b in order:
            members = by_bin[b]
            rng.shuffle(members)
            out.extend(members)
        _ = n_bins_hint  # accepted for API symmetry; binning happens at build time
        return out

    def iter_batches(
        self,
        rowids: Sequence[int],
        *,
        batch_size: int,
        columns: Sequence[str] | None = None,
    ) -> Iterator[pa.Table]:
        """Yield row batches for a shuffled ``_rowid`` order via coalesced take.

        Each batch is a single ``take_by_stable_rowid`` — Lance coalesces the
        underlying reads. No cross-file ``open`` per sample (contrast: WDS shuffle
        re-opens a different tar almost every step).
        """
        for chunk in _chunked(rowids, batch_size):
            if chunk:
                yield self.take_by_stable_rowid(chunk, columns=columns)


def resolve_audio(
    row: dict,
    side: str,
    byte_reader: ByteReader | None = None,
) -> bytes | None:
    """Resolve one row's audio bytes: inline first, else pointer fallback.

    The two-layer design inlines audio into ``<side>_audio_bytes``. Oversized or
    not-yet-inlined rows keep the WDS pointer (``<side>_audio_shard_uri`` +
    ``byte_offset`` + ``byte_length``); this reads those back through the *existing*
    ``wds_tar`` reader — no Lance-specific byte path.

    ``byte_reader`` is any callable ``(shard_uri, byte_offset, byte_length) -> bytes``
    — inject the production ``ObjectStoreWdsTarShardReader.read_payload`` (or its
    local counterpart). Dependency-injected so this stays free of ``curion.core`` /
    object-store imports and is unit-testable with a fake reader.

    Parameters
    ----------
    row:
        A mapping for one row (e.g. ``table.to_pylist()[i]``) carrying the
        ``<side>_audio_bytes`` blob and/or the pointer columns.
    side:
        ``"source"`` or ``"target"``.

    Returns
    -------
    The audio bytes, or ``None`` if neither an inline blob nor a resolvable
    pointer is present.
    """
    if side not in ("source", "target"):
        raise ValueError(f"side must be 'source' or 'target', got {side!r}")

    inline = row.get(f"{side}_audio_bytes")
    if inline is not None:
        return bytes(inline)

    uri = row.get(f"{side}_audio_shard_uri")
    offset = row.get(f"{side}_audio_byte_offset")
    length = row.get(f"{side}_audio_byte_length")
    if uri and offset is not None and length:
        if byte_reader is None:
            raise ValueError(
                f"{side} row has no inline bytes and a pointer "
                f"({uri!r}) but no byte_reader was provided for fallback"
            )
        return byte_reader(uri, int(offset), int(length))
    return None


def _chunked(items: Iterable[int], size: int) -> Iterator[list[int]]:
    batch: list[int] = []
    for it in items:
        batch.append(it)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
