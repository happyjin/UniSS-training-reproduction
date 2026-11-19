"""Layer-2 task/manifest for the Lance two-layer design: predicate + version.

In the two-layer architecture the ``curated -> manifests -> train_ready`` physical
snapshots collapse into a *query*: a task is no longer a physical partition or a
copied table, it is a filter predicate over one ``processed.lance`` plus a pinned
version. The ``zh_en`` rows are stored once; task1/task2 both ``WHERE`` them, zero
duplicate copies (design §6).

Two reuse semantics (design §6.1.1), which differ in what must be frozen:

* **Situation 1 — predicate reproducible.** The sample set is a pure condition
  filter (``direction='zh_en' AND utmos>=3.0``). Freezing
  ``(lance_version, filter, seed)`` fully reproduces it. No stable row id needed.
* **Situation 2 — exact sample set.** The set was chosen by random subsampling or
  hand-picking and cannot be re-derived by SQL. It must pin the exact
  ``(lance_version, rowid_list)`` of *stable* ``_rowid`` values, which survive
  compaction (see :meth:`LanceRowReader.take_by_stable_rowid`).

A frozen manifest is a small JSON-serializable dict — deterministic, auditable,
split-aware, framework-neutral (matching the manifest properties in the existing
``12-training-view-design.md``). It holds no tensors and no bytes.

Depends only on stdlib + the :class:`LanceRowReader`; no ``curion.core``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import lance
import pyarrow as pa

MANIFEST_SCHEMA_VERSION = "lance-task-manifest/v1"
_SPLIT_NAMES = ("train", "val", "test")

# Schema-metadata keys for materialized task traceability.
META_SOURCE_URI = "curion.source_uri"
META_SOURCE_VERSION = "curion.source_version"
META_SOURCE_FILTER = "curion.source_filter"
META_TASK_NAME = "curion.task_name"


def _lance_row_reader_cls():
    """Import :class:`LanceRowReader` lazily.

    Kept out of module scope so ``lance_task`` imports without pulling
    ``curion.storage.__init__`` (which imports ``curion.core``, requiring
    Python >= 3.10). Prefers the package path, falls back to a flat import for
    standalone dev/test on the reference Python.
    """
    try:
        from curion.storage.lance_reader import LanceRowReader
    except Exception:  # pragma: no cover - env-dependent fallback
        from lance_reader import LanceRowReader  # type: ignore
    return LanceRowReader


@dataclass(frozen=True)
class SplitPolicy:
    """How train/val/test are derived (group-aware to avoid speaker leakage).

    The default policy ``group_stratified_random_hash`` hashes each group's id
    (e.g. ``source_speaker_id``) together with ``seed`` to deterministically
    assign the *entire group* to one split. This prevents the same speaker from
    appearing in both train and val (leakage).
    """

    policy: str = "group_stratified_random_hash"
    group_by: str = "source_speaker_id"
    seed: int = 0
    # (train, val, test) — must sum to 1.0.
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)

    def __post_init__(self) -> None:
        if abs(sum(self.ratios) - 1.0) > 1e-6:
            raise ValueError(
                f"split ratios must sum to 1.0, got {self.ratios} "
                f"(sum={sum(self.ratios):.6f})"
            )
        if len(self.ratios) != 3:
            raise ValueError(
                f"expected 3 ratios (train/val/test), got {len(self.ratios)}"
            )


def _group_hash_to_split(
    group_value: str, seed: int, ratios: tuple[float, float, float]
) -> str:
    """Deterministic hash of ``(seed, group_value)`` → split name.

    Uses ``hashlib`` (not Python ``hash()``) for cross-run determinism
    (``hash()`` is affected by ``PYTHONHASHSEED``).
    """
    h = hashlib.sha1(f"{seed}:{group_value}".encode("utf-8")).digest()
    h_val = int.from_bytes(h[:4], "big") / 0x100000000  # uniform [0, 1)
    cum = 0.0
    for name, ratio in zip(_SPLIT_NAMES, ratios):
        cum += ratio
        if h_val < cum:
            return name
    return _SPLIT_NAMES[-1]  # floating-point rounding safety


@dataclass(frozen=True)
class Sampling:
    """Runtime sampling config (weights/oversample) — lives in config, not Lance.

    NOTE: this is a *config holder only*. Execution (applying weights to the
    dataloader's sampler) is intentionally deferred to the training/dataloader
    side per the design (§6.1: "Sampling exec → dataloader side"). The layer-2
    API just carries the spec so the frozen manifest is self-describing.
    """

    mode: str = "uniform"  # e.g. "weights"
    weight_expr: str | None = None  # optional column/expr used to derive weights


@dataclass(frozen=True)
class TaskSpec:
    """A task = predicate + pinned version (+ runtime policy).

    Exactly one selection mode is active:

    * ``filter`` set, ``rowid_list`` None  -> situation 1 (predicate reproducible)
    * ``rowid_list`` set, ``filter`` None   -> situation 2 (exact sample set)
    """

    name: str
    lance_version: int
    filter: str | None = None
    rowid_list: Sequence[int] | None = None
    seed: int = 0
    split_policy: SplitPolicy = field(default_factory=SplitPolicy)
    sampling: Sampling = field(default_factory=Sampling)
    # Reference to a prompt recipe (e.g. "s2s_v1") — part of task identity for
    # the training side. Layer-2 carries it so the frozen manifest is
    # self-describing; it does not affect row resolution.
    prompt_recipe_id: str | None = None

    def __post_init__(self) -> None:
        has_filter = self.filter is not None
        has_rowids = self.rowid_list is not None
        if has_filter == has_rowids:
            raise ValueError(
                "TaskSpec requires exactly one of `filter` (situation 1) or "
                "`rowid_list` (situation 2)"
            )

    @property
    def situation(self) -> int:
        return 1 if self.filter is not None else 2


@dataclass(frozen=True)
class ResolvedTask:
    """The concrete row set a task resolves to on a pinned Lance version."""

    name: str
    lance_version: int
    situation: int
    rowids: list[int]

    @property
    def num_rows(self) -> int:
        return len(self.rowids)


def _resolve_reader(reader_or_uri, spec: TaskSpec, **reader_kwargs):
    """Get a version-pinned :class:`LanceRowReader` from a reader or URI."""
    LanceRowReader = _lance_row_reader_cls()
    if isinstance(reader_or_uri, LanceRowReader):
        reader = reader_or_uri
        if reader.version != spec.lance_version:
            raise ValueError(
                f"reader pinned to version {reader.version} but task "
                f"{spec.name!r} wants {spec.lance_version}"
            )
    else:
        reader = LanceRowReader(
            reader_or_uri, version=spec.lance_version, **reader_kwargs
        )
    return reader


def resolve_task(reader_or_uri, spec: TaskSpec, **reader_kwargs) -> ResolvedTask:
    """Resolve a :class:`TaskSpec` to a concrete list of stable ``_rowid`` values.

    The reader is always pinned to ``spec.lance_version`` for reproducibility. A
    :class:`LanceRowReader` or a dataset URI may be passed.
    """
    reader = _resolve_reader(reader_or_uri, spec, **reader_kwargs)

    if spec.situation == 1:
        rowids = reader.stable_row_ids(filter=spec.filter)
    else:
        # Situation 2: the exact pinned set; keep only ids still present.
        requested = [int(r) for r in (spec.rowid_list or [])]
        present = set(reader.take_by_stable_rowid(requested, columns=[]).column("_rowid").to_pylist())
        rowids = [r for r in requested if r in present]

    return ResolvedTask(
        name=spec.name,
        lance_version=spec.lance_version,
        situation=spec.situation,
        rowids=rowids,
    )


def resolve_all_splits(
    reader_or_uri,
    spec: TaskSpec,
    **reader_kwargs,
) -> dict[str, ResolvedTask]:
    """Resolve a :class:`TaskSpec` into train/val/test in a **single pass**.

    This is the efficient entry point for training: one predicate scan + one
    group-column read, then partition into all three splits in Python. Calling
    :func:`resolve_split` three times would do 3× predicate scans + 3× group
    reads — on 2.55M rows that's a 6× waste.

    Returns ``{"train": ResolvedTask, "val": ResolvedTask, "test": ResolvedTask}``.
    The union of all three equals :func:`resolve_task`; splits are disjoint; no
    group (e.g. speaker) appears in more than one split.
    """
    reader = _resolve_reader(reader_or_uri, spec, **reader_kwargs)

    # 1. Validate group_by column exists (clear error before expensive scan).
    group_col = spec.split_policy.group_by
    if not reader.has_column(group_col):
        raise ValueError(
            f"split_policy.group_by={group_col!r} not found in dataset schema "
            f"(columns: {reader.schema.names[:10]}...). "
            f"Set split_policy.group_by to an existing column."
        )

    # 2. Resolve task ONCE (predicate → rowids).
    resolved = resolve_task(reader, spec)
    if not resolved.rowids:
        empty = ResolvedTask(
            name=f"{resolved.name}:_empty",
            lance_version=resolved.lance_version,
            situation=resolved.situation,
            rowids=[],
        )
        return {s: empty for s in _SPLIT_NAMES}

    # 3. Read group_by column ONCE for all matched rows.
    tbl = reader.take_by_stable_rowid(resolved.rowids, columns=[group_col])
    rowids = tbl.column("_rowid").to_pylist()
    groups = tbl.column(group_col).to_pylist()

    # 4. Partition into train/val/test in one pass.
    sp = spec.split_policy
    buckets: dict[str, list[int]] = {s: [] for s in _SPLIT_NAMES}
    for rid, g in zip(rowids, groups):
        split_name = _group_hash_to_split(str(g), sp.seed, sp.ratios)
        buckets[split_name].append(rid)

    return {
        s: ResolvedTask(
            name=f"{resolved.name}:{s}",
            lance_version=resolved.lance_version,
            situation=resolved.situation,
            rowids=buckets[s],
        )
        for s in _SPLIT_NAMES
    }


def resolve_split(
    reader_or_uri,
    spec: TaskSpec,
    *,
    split: str,
    **reader_kwargs,
) -> ResolvedTask:
    """Resolve a :class:`TaskSpec` to one split's rowids (group-stratified).

    Convenience wrapper around :func:`resolve_all_splits` that returns only one
    split. If you need train+val+test, call :func:`resolve_all_splits` directly
    to avoid 3× repeated scans.

    Parameters
    ----------
    split:
        One of ``"train"``, ``"val"``, ``"test"``.
    """
    if split not in _SPLIT_NAMES:
        raise ValueError(f"split must be one of {_SPLIT_NAMES}, got {split!r}")
    all_splits = resolve_all_splits(reader_or_uri, spec, **reader_kwargs)
    return all_splits[split]


def freeze_manifest(spec: TaskSpec, *, dataset_uri: str | None = None) -> dict:
    """Serialize the reproducibility contract for a task (situation 1 or 2).

    Situation 1 freezes ``(lance_version, filter, seed)``; situation 2 additionally
    pins the exact stable ``rowid_list``. Returns a plain JSON-serializable dict.
    """
    manifest = {
        "schema": MANIFEST_SCHEMA_VERSION,
        "name": spec.name,
        "dataset_uri": dataset_uri,
        "lance_version": spec.lance_version,
        "situation": spec.situation,
        "seed": spec.seed,
        "split_policy": asdict(spec.split_policy),
        "sampling": asdict(spec.sampling),
        "prompt_recipe_id": spec.prompt_recipe_id,
    }
    if spec.situation == 1:
        manifest["filter"] = spec.filter
    else:
        manifest["rowid_list"] = [int(r) for r in (spec.rowid_list or [])]
    return manifest


def manifest_to_json(manifest: dict) -> str:
    """Canonical, deterministic JSON for a frozen manifest (auditable/diffable)."""
    return json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2)


def task_from_config(name: str, cfg: dict) -> TaskSpec:
    """Build a :class:`TaskSpec` from a plain config dict (e.g. parsed tasks.yaml)."""
    sp = cfg.get("split_policy", {})
    sa = cfg.get("sampling", {})
    return TaskSpec(
        name=name,
        lance_version=int(cfg["lance_version"]),
        filter=cfg.get("filter"),
        rowid_list=cfg.get("rowid_list"),
        seed=int(cfg.get("seed", 0)),
        split_policy=SplitPolicy(
            policy=sp.get("policy", "group_stratified_random_hash"),
            group_by=sp.get("group_by", "source_speaker_id"),
            seed=int(sp.get("seed", 0)),
        ),
        sampling=Sampling(
            mode=sa.get("mode", "uniform"),
            weight_expr=sa.get("weight_expr"),
        ),
        prompt_recipe_id=cfg.get("prompt_recipe_id"),
    )


def load_tasks_config(path: str | Path) -> dict[str, TaskSpec]:
    """Load multiple task specs from a YAML or JSON file.

    The file should have a top-level ``tasks`` mapping (matching the design
    doc's ``tasks.yaml`` format)::

        tasks:
          task1_zh2en:
            lance_version: 1
            filter: "direction = 'zh_en' AND target_audio_utmos_score >= 3.0"
            prompt_recipe_id: s2s_v1
            split_policy: {group_by: source_speaker_id, seed: 42}
          task2_bidir:
            lance_version: 2
            filter: "direction IN ('zh_en','en_zh')"
            prompt_recipe_id: s2s_v2

    YAML requires the optional ``pyyaml`` dependency (follows the repo
    convention used by curion-compiler / curion-deploy). JSON needs no extra
    dependency.

    Returns ``{task_name: TaskSpec, ...}``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"tasks config not found: {p}")

    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                f"loading {suffix} requires pyyaml; install curion-storage[yaml] "
                f"or use a .json file instead"
            ) from e
        with open(p) as f:
            data = yaml.safe_load(f)
    elif suffix == ".json":
        with open(p) as f:
            data = json.load(f)
    else:
        raise ValueError(f"unsupported config format {suffix!r}; use .yaml/.yml/.json")

    tasks = (data or {}).get("tasks", {})
    if not tasks:
        raise ValueError(f"no 'tasks' key found in {p}")
    return {name: task_from_config(name, cfg) for name, cfg in tasks.items()}


# ---------------------------------------------------------------------------
# Materialization: logical view → physical dataset
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaterializedTask:
    """Handle to a materialized task (physical Lance dataset for training).

    A materialized task is a **cache** of the logical view: the rows matching
    a task's predicate + split, written to a separate Lance dataset so training
    can do sequential scan (no predicate, no random take — maximum throughput).

    The source of truth is always the :class:`TaskSpec` (logical view); the
    materialized dataset can be deleted and rebuilt at any time from the spec.
    """

    task_name: str
    uri: str                          # e.g. "s3://.../materialized/task1.lance"
    version: int                      # version of this materialized dataset
    source_uri: str                   # original processed.lance URI
    source_version: int               # which processed.lance version was materialized
    source_filter: str | None         # the predicate that was materialized
    num_rows: int
    split_counts: dict[str, int]      # {"train": N, "val": N, "test": N}


def materialize_task(
    reader_or_uri,
    spec: TaskSpec,
    out_dir: str,
    *,
    columns: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    batch_size: int = 10000,
    storage_options: dict | None = None,
    tag: str | None = None,
) -> MaterializedTask:
    """Materialize a task's splits into a physical Lance dataset.

    Resolves the task (predicate + version + split), reads all matching rows
    in batches, adds a ``split`` column, and writes to
    ``{out_dir}/{spec.name}.lance``.

    * First materialization: ``mode="create"`` (new dataset).
    * Re-materialization: ``mode="overwrite"`` (new version, old versions
      preserved for rollback via ``open_materialized(uri, version=...)``).

    A BITMAP index on the ``split`` column makes ``filter="split='train'"``
    nearly free at training time. Schema metadata records the full source
    :class:`TaskSpec` (as JSON) for traceability.

    Parameters
    ----------
    columns:
        Columns to carry from the source dataset. ``None`` = all columns.
        **For large datasets, always pass columns** — reading all columns
        (including audio bytes) for 2.55M rows without batching will OOM.
    splits:
        Which splits to materialize (default: all three). E.g.
        ``("train",)`` to only materialize the train split.
    batch_size:
        Rows per batch (default 10000). Each batch is one
        ``take_by_stable_rowid`` + one Lance write. Larger = fewer writes
        but more memory.
    tag:
        Optional Lance tag name to freeze this materialization (e.g.
        ``"task1_v1"``). Enables ``open_materialized(uri, version="task1_v1")``
        for rollback. If the tag already exists, it is updated to the new
        version.
    """
    # Capture source URI for traceability.
    LanceRowReader = _lance_row_reader_cls()
    if isinstance(reader_or_uri, LanceRowReader):
        source_uri = reader_or_uri.dataset.uri
    else:
        source_uri = str(reader_or_uri)

    reader = _resolve_reader(reader_or_uri, spec, storage_options=storage_options or {})

    # 1. Resolve all splits (one predicate scan + one group-column read).
    all_splits = resolve_all_splits(reader, spec)

    # 2. Determine which splits to materialize.
    target_splits = tuple(splits) if splits is not None else _SPLIT_NAMES
    for s in target_splits:
        if s not in _SPLIT_NAMES:
            raise ValueError(f"invalid split {s!r}; must be one of {_SPLIT_NAMES}")

    # 3. Write in batches to avoid OOM on large datasets.
    out_uri = f"{out_dir.rstrip('/')}/{spec.name}.lance"
    try:
        lance.dataset(out_uri, storage_options=storage_options)
        first_mode = "overwrite"
    except Exception:
        first_mode = "create"

    split_counts: dict[str, int] = {s: 0 for s in _SPLIT_NAMES}
    is_first = True
    total_written = 0

    for split_name in target_splits:
        rowids = all_splits[split_name].rowids
        split_counts[split_name] = len(rowids)
        if not rowids:
            continue

        for i in range(0, len(rowids), batch_size):
            batch_rowids = rowids[i : i + batch_size]
            tbl = reader.take_by_stable_rowid(batch_rowids, columns=columns)
            # Drop _rowid (Lance internal) before writing.
            if "_rowid" in tbl.column_names:
                tbl = tbl.drop(["_rowid"])
            split_arr = pa.array([split_name] * len(batch_rowids), type=pa.string())
            tbl = tbl.append_column("split", split_arr)

            mode = first_mode if is_first else "append"
            lance.write_dataset(
                tbl,
                out_uri,
                mode=mode,
                enable_stable_row_ids=True,
                storage_options=storage_options,
            )
            is_first = False
            total_written += len(batch_rowids)

    # Handle empty task: create a minimal dataset with just the split column.
    if is_first:
        empty = pa.table({"split": pa.array([], type=pa.string())})
        lance.write_dataset(
            empty, out_uri, mode=first_mode,
            enable_stable_row_ids=True, storage_options=storage_options,
        )

    ds = lance.dataset(out_uri, storage_options=storage_options)

    # 4. BITMAP index on `split` (low-cardinality: 3 values).
    if ds.count_rows() > 0 and "split" in ds.schema.names:
        try:
            ds.create_scalar_index("split", "BITMAP")
            ds = lance.dataset(out_uri, storage_options=storage_options)
        except Exception as e:
            import logging
            logging.getLogger("lance_task").warning(
                "materialize_task: BITMAP index on 'split' failed: %s", e
            )

    # 5. Schema metadata: full TaskSpec as JSON for complete traceability.
    spec_json = manifest_to_json(freeze_manifest(spec, dataset_uri=source_uri))
    ds.update_schema_metadata({
        META_SOURCE_URI: source_uri,
        META_SOURCE_VERSION: str(spec.lance_version),
        META_SOURCE_FILTER: spec.filter or "",
        META_TASK_NAME: spec.name,
        "curion.source_spec_json": spec_json,
    })
    ds = lance.dataset(out_uri, storage_options=storage_options)

    # 6. Optional tag for freezing / rollback (update if already exists).
    if tag is not None:
        try:
            ds.tags.create(tag, ds.version)
        except Exception:
            ds.tags.update(tag, ds.version)

    return MaterializedTask(
        task_name=spec.name,
        uri=out_uri,
        version=ds.version,
        source_uri=source_uri,
        source_version=spec.lance_version,
        source_filter=spec.filter,
        num_rows=ds.count_rows(),
        split_counts=split_counts,
    )


def open_materialized(
    uri: str,
    *,
    version: int | str | None = None,
    storage_options: dict | None = None,
):
    """Open a materialized task dataset for training.

    Parameters
    ----------
    version:
        * ``int``: pin a specific Lance version (rollback to a previous
          materialization).
        * ``str``: pin a Lance tag name (e.g. ``"task1_v1"``).
        * ``None``: read the latest version.

    Returns a :class:`LanceRowReader` pinned to the requested version/tag.
    Training typically reads with ``filter="split = 'train'"`` (BITMAP-accelerated).
    """
    LanceRowReader = _lance_row_reader_cls()
    return LanceRowReader(uri, version=version, storage_options=storage_options)


def list_materialized_tasks(
    out_dir: str,
    *,
    storage_options: dict | None = None,
) -> list[MaterializedTask]:
    """List all materialized task datasets in ``out_dir``.

    Scans the directory for ``*.lance`` datasets, reads schema metadata for
    traceability, and returns a :class:`MaterializedTask` for each. Useful for
    discovering what has already been materialized without tracking URIs manually.
    """
    import os

    results: list[MaterializedTask] = []
    dir_path = out_dir.rstrip("/")

    # For local paths, use os.listdir; for S3, use pyarrow.fs.
    if dir_path.startswith("s3://"):
        import pyarrow.fs as fs
        bucket_parts = dir_path[5:].split("/", 1)
        bucket = bucket_parts[0]
        prefix = bucket_parts[1] if len(bucket_parts) > 1 else ""
        region = (storage_options or {}).get("region")
        s3 = fs.S3FileSystem(region=region) if region else fs.S3FileSystem()
        try:
            entries = s3.get_file_info(fs.FileSelector(prefix, allow_not_found=True, recursive=False))
        except Exception:
            return []
        subdirs = [e.path for e in entries if e.is_file and e.path.endswith(".lance")]
        # S3 Lance datasets are directories ending in .lance — check for dirs
        subdirs = []
        for e in entries:
            if e.path.endswith(".lance"):
                subdirs.append(f"s3://{bucket}/{e.path}")
    else:
        if not os.path.isdir(dir_path):
            return []
        subdirs = [
            os.path.join(dir_path, name)
            for name in os.listdir(dir_path)
            if name.endswith(".lance") and os.path.isdir(os.path.join(dir_path, name))
        ]

    for uri in sorted(subdirs):
        try:
            ds = lance.dataset(uri, storage_options=storage_options)
        except Exception:
            continue
        md = ds.schema.metadata or {}
        task_name = md.get(META_TASK_NAME.encode(), b"").decode() or Path(uri).name.replace(".lance", "")
        source_uri = md.get(META_SOURCE_URI.encode(), b"").decode()
        source_version = int(md.get(META_SOURCE_VERSION.encode(), b"0"))
        source_filter = md.get(META_SOURCE_FILTER.encode(), b"").decode() or None

        # Count rows per split if split column exists.
        split_counts = {s: 0 for s in _SPLIT_NAMES}
        if "split" in ds.schema.names:
            for s in _SPLIT_NAMES:
                try:
                    split_counts[s] = ds.count_rows(filter=f"split = '{s}'")
                except Exception:
                    pass

        results.append(MaterializedTask(
            task_name=task_name,
            uri=uri,
            version=ds.version,
            source_uri=source_uri,
            source_version=source_version,
            source_filter=source_filter,
            num_rows=ds.count_rows(),
            split_counts=split_counts,
        ))
    return results


__all__ = [
    "TaskSpec",
    "SplitPolicy",
    "Sampling",
    "ResolvedTask",
    "MaterializedTask",
    "resolve_task",
    "resolve_split",
    "resolve_all_splits",
    "materialize_task",
    "open_materialized",
    "list_materialized_tasks",
    "freeze_manifest",
    "manifest_to_json",
    "task_from_config",
    "load_tasks_config",
    "MANIFEST_SCHEMA_VERSION",
]
