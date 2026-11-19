#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FSx cache prewarm / release / audio-path extraction / audit CLI.

Stateless command-line toolkit used by ``worker.sh`` (the shell orchestrator).
Each subcommand does ONE thing and exits; no in-memory state is shared across
invocations, so the orchestration (batching, sequencing, maker calls) lives in
bash where it belongs.

Subcommands:
  extract-paths  Map a batch of parquet files to the on-FSx audio paths they
                 reference (mirrors dataset_maker/* path logic).
  produce        Background producer: ``lfs hsm_restore`` audio dirs in bulk
                 (directory-batched), polling until ready, recording each
                 restored directory to a state file for the consumer.
  ready          Exit 0 iff every audio path's parent directory is recorded
                 ready in the producer's state file (pure file lookup, no
                 ``lfs hsm_action`` — avoids ambiguous FSx cache state).
  release        ``lfs hsm_release`` a list of audio paths (evict from cache,
                 keep metadata) so the working set stays bounded.
  audit          Sum the byte size of every audio file referenced by jsonl(s).

The warm uses the Lustre HSM user-space tool ``lfs``. When ``lfs`` is not
installed (or the container is not privileged) warm/release degrade to a no-op
with a warning rather than failing the whole pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Optional, Sequence

DEFAULT_FSX_MOUNT = "/mnt/fsx"


def _log(msg: str) -> None:
    print(f"[prewarm] {time.strftime('%H:%M:%S')} {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# URI / path helpers (mirror dataset_maker/* logic so prewarm stays consistent)
# ---------------------------------------------------------------------------
def basename_from_uri(value: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(value)
    path = parsed.path if parsed.scheme else value
    path = path.split("#", 1)[0].split("?", 1)[0]
    return os.path.basename(path)


def _s3_to_fsx(uri: str, fsx_mount: str) -> Optional[str]:
    """Map ``s3://<bucket>/<key>`` -> ``<fsx_mount>/<key>`` (bucket-insensitive)."""
    from urllib.parse import urlparse

    if not isinstance(uri, str) or not uri.startswith("s3://"):
        return None
    parsed = urlparse(uri)
    if not parsed.netloc or not parsed.path:
        return None
    key = parsed.path.lstrip("/")
    if not key:
        return None
    return os.path.join(fsx_mount, key)


def _tts_audio_path(audio_uri: str, fsx_mount: str) -> Optional[str]:
    uri = (audio_uri or "").strip()
    if not uri:
        return None
    if os.path.isabs(uri) and os.path.isfile(uri):
        return os.path.abspath(uri)
    fsx = _s3_to_fsx(uri, fsx_mount)
    if fsx:
        return fsx
    # last resort: a bare basename that may already live on fsx
    base = basename_from_uri(uri)
    if base:
        return os.path.join(fsx_mount, base)
    return None


def _s2s_audio_path(uri: str, input_dir: str, subdir: str) -> Optional[str]:
    uri = (uri or "").strip()
    if not uri:
        return None
    if os.path.isabs(uri) and os.path.isfile(uri):
        return os.path.abspath(uri)
    base = basename_from_uri(uri)
    if not base:
        return None
    return os.path.join(input_dir, subdir, base)


# ---------------------------------------------------------------------------
# Audio-path extraction from a parquet batch
# ---------------------------------------------------------------------------
_TTS_RE = re.compile(r"examples\.parquet$")
_S2S_RE = re.compile(r"^(?:part\d+|zh_en_part_\d+)\.parquet$")


def extract_audio_paths(
    parquet_path: str,
    kind: str,
    fsx_mount: str = DEFAULT_FSX_MOUNT,
    input_dir: Optional[str] = None,
) -> list[str]:
    """Return the on-FSx audio file paths referenced by one parquet file.

    ``kind`` is ``"tts"`` or ``"s2s"``.  For S2S, ``input_dir`` is the
    curated dataset directory (``.../source_audio`` and ``.../target_audio``
    live underneath it).
    """
    import pyarrow.parquet as pq

    kind = kind.lower()
    if kind == "tts":
        cols = ["audio_uri"]
    else:
        cols = ["source_audio_uri", "target_audio_uri"]

    try:
        # Read the SINGLE file directly (ParquetFile), NOT pq.read_table(): the
        # latter triggers hive-partition inference on the ``dataset=<name>/``
        # path component, adding a dictionary-typed ``dataset`` partition column
        # that conflicts with the file's own string ``dataset`` column
        # ("Unable to merge: Field dataset has incompatible types"). ParquetFile
        # reads only the file's own columns with no partition discovery — the
        # same way the makers read (pd.read_parquet / ParquetFile.read_row_group).
        pf = pq.ParquetFile(parquet_path)
        available = set(pf.schema_arrow.names)
        read_cols = [c for c in cols if c in available]
        if not read_cols:
            _log(f"  skip extract (no audio columns {cols} in {parquet_path})")
            return []
        table = pf.read(columns=read_cols)
    except Exception as exc:  # noqa: BLE001
        _log(f"  skip extract (unreadable parquet {parquet_path}): {exc}")
        return []

    present = set(table.column_names)
    out: list[str] = []
    if kind == "tts":
        for u in table.column("audio_uri").to_pylist():
            p = _tts_audio_path(u, fsx_mount)
            if p:
                out.append(p)
    else:
        n = table.num_rows
        src = table.column("source_audio_uri").to_pylist() if "source_audio_uri" in present else [None] * n
        tgt = table.column("target_audio_uri").to_pylist() if "target_audio_uri" in present else [None] * n
        for s, t in zip(src, tgt):
            for uri, sub in ((s, "source_audio"), (t, "target_audio")):
                p = _s2s_audio_path(uri, input_dir or "", sub)
                if p:
                    out.append(p)
    # de-dup while preserving order
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


# ---------------------------------------------------------------------------
# FSx HSM warm (lfs hsm_restore + poll)
# ---------------------------------------------------------------------------
def _lfs_available() -> bool:
    return shutil.which("lfs") is not None


def _restore_one(path: str) -> None:
    try:
        subprocess.run(
            ["lfs", "hsm_restore", path],
            capture_output=True, text=True, timeout=300,
        )
    except Exception:  # noqa: BLE001
        pass  # restore is best-effort; poll loop reports what stuck


def _hsm_state(path: str) -> str:
    try:
        proc = subprocess.run(
            ["lfs", "hsm_action", path],
            capture_output=True, text=True, timeout=300,
        )
        return "NOOP" if "NOOP" in proc.stdout else "PENDING"
    except Exception:  # noqa: BLE001
        return "PENDING"


def warm_paths(
    paths: Sequence[str],
    parallel: int = 16,
    poll_interval: int = 30,
    timeout: int = 7200,
) -> int:
    """Warm ``paths`` into the FSx SSD cache. Returns # files successfully cached.

    No-op (returns 0) when ``lfs`` is unavailable.
    """
    uniq = list(dict.fromkeys(p for p in paths if p))
    if not uniq:
        return 0
    if not _lfs_available():
        _log("lfs binary not available (container not privileged / lustre-client "
             "missing) — skipping HSM restore. FSx reads will trigger on-demand recall.")
        return 0

    _log(f"warming {len(uniq)} audio files into FSx SSD cache ({parallel} concurrent) ...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        list(ex.map(_restore_one, uniq))
    _log(f"  restore queued in {time.time() - t0:.0f}s, polling ...")

    pending = set(uniq)
    t0 = time.time()
    while pending and (time.time() - t0) < timeout:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            states = list(ex.map(_hsm_state, list(pending)))
        done = {p for p, st in zip(list(pending), states) if st == "NOOP"}
        if done:
            pending -= done
            _log(f"  {len(uniq) - len(pending)}/{len(uniq)} cached "
                 f"(elapsed {time.time() - t0:.0f}s)")
        if not pending:
            break
        time.sleep(poll_interval)

    if pending:
        _log(f"  WARNING: {len(pending)} files still restoring after {timeout}s timeout")
    else:
        _log(f"  all {len(uniq)} files cached in {time.time() - t0:.0f}s")
    return len(uniq) - len(pending)


def _release_one(path: str) -> None:
    """Evict a single file from the FSx SSD cache (best-effort).

    ``lfs hsm_release`` marks the file as released so its data is dropped from
    the cache (freeing space) while the metadata/size stays on the MDT. Reading
    a released file later triggers an on-demand recall, which is what the
    training jobs do. Releasing a not-yet-cached or already-released file is a
    no-op/error we swallow — it is purely advisory cache management.
    """
    try:
        subprocess.run(
            ["lfs", "hsm_release", path],
            capture_output=True, text=True, timeout=300,
        )
    except Exception:  # noqa: BLE001
        pass


def release_paths(paths: Sequence[str]) -> int:
    """Release ``paths`` from the FSx SSD cache. Returns # files released.

    No-op (returns 0) when ``lfs`` is unavailable or the list is empty. This is
    what keeps the rolling working set bounded: once a batch's audio has been
    consumed by the data_maker we drop it, so the cache never accumulates every
    batch's audio at once. Release is best-effort and serialized (no client
    concurrency knob) — HSM recall throughput is server-side bounded anyway.
    """
    uniq = list(dict.fromkeys(p for p in paths if p))
    if not uniq:
        return 0
    if not _lfs_available():
        return 0
    _log(f"releasing {len(uniq)} audio files from FSx cache (batch consumed) ...")
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(_release_one, uniq))
    return len(uniq)


# ---------------------------------------------------------------------------
# Full preload of whole directory trees (task-start prewarm)
# ---------------------------------------------------------------------------
def preload_roots(
    roots: Sequence[str],
    exts: Optional[Sequence[str]] = None,
    dir_batch: int = 2000,
    poll_interval: int = 15,
    timeout: int = 7200,
) -> int:
    """Proactively ``lfs hsm_restore`` EVERY audio file under ``roots`` into the
    FSx SSD cache at task start (per the AWS "preload file contents via HSM DRA"
    recipe: https://docs.aws.amazon.com/fsx/latest/LustreGuide/preload-file-contents-hsm-dra.html).

    Walks each root, keeps files matching ``exts`` (``None`` = every file),
    groups them by parent directory and restores each directory in bulk
    ``dir_batch``-sized chunks (one ``lfs hsm_restore`` call per chunk keeps the
    HSM agent at max throughput — the same lever the rolling producer uses).
    Then polls until every directory reports NOOP or ``timeout`` elapses
    (best-effort: a cold read still works afterwards, so the pipeline never
    deadlocks even if the cache is smaller than the full audio set).

    Returns the number of directories queued. No-op (returns 0) when ``lfs`` is
    unavailable (container not privileged / lustre-client missing); the makers
    then trigger on-demand HSM recall when they read the audio.
    """
    if not _lfs_available():
        _log("lfs unavailable — skipping full preload "
             "(makers will trigger on-demand recall).")
        return 0

    import fnmatch

    files: list[str] = []
    for root in roots:
        if not root:
            continue
        if not os.path.isdir(root):
            _log(f"  skip non-directory root: {root}")
            continue
        _log(f"  scanning {root} ...")
        for dirpath, _dirs, fnames in os.walk(root):
            for fn in fnames:
                if exts and not any(
                    fnmatch.fnmatchcase(fn.lower(), e.lower()) for e in exts
                ):
                    continue
                files.append(os.path.join(dirpath, fn))
    _log(f"  collected {len(files):,} candidate files under {len(roots)} root(s)")
    if not files:
        return 0

    # group by parent directory so a single lfs hsm_restore call can batch the
    # whole directory's files (server-side recall, not per-file RPC).
    dirs: dict[str, list[str]] = {}
    for f in files:
        dirs.setdefault(os.path.dirname(f), []).append(f)

    t0 = time.time()
    for d, fl in dirs.items():
        for i in range(0, len(fl), dir_batch):
            chunk = fl[i:i + dir_batch]
            try:
                subprocess.run(["lfs", "hsm_restore", *chunk],
                               capture_output=True, text=True, timeout=300)
            except Exception:  # noqa: BLE001
                pass
    _log(f"  restore queued in {time.time() - t0:.0f}s; "
         f"{'no-wait mode (timeout<=0): HSM recall proceeds server-side in the '
          'background, not blocking the pipeline.' if timeout <= 0 else f'polling {len(dirs):,} dirs (timeout={timeout}s) ...'}")

    if timeout <= 0:
        # --no-wait: just queue the restore requests and return. The actual
        # archive->cache copy is driven by the FSx/HSM agent asynchronously and
        # overlaps with data processing, so there is nothing to wait for here.
        return len(dirs)

    t0 = time.time()
    remaining = set(dirs.keys())
    while remaining and (time.time() - t0) < timeout:
        done = set()
        for d in list(remaining):
            fl = dirs[d]
            non_noop = 0
            for i in range(0, len(fl), dir_batch):
                chunk = fl[i:i + dir_batch]
                try:
                    proc = subprocess.run(["lfs", "hsm_action", *chunk],
                                          capture_output=True, text=True, timeout=300)
                    non_noop += sum(1 for ln in proc.stdout.splitlines()
                                    if "NOOP" not in ln)
                except Exception:  # noqa: BLE001
                    non_noop += len(chunk)
            if non_noop == 0:
                done.add(d)
        remaining -= done
        if not remaining:
            break
        _log(f"  {len(dirs) - len(remaining)}/{len(dirs)} dirs ready "
             f"(elapsed {time.time() - t0:.0f}s)")
        time.sleep(poll_interval)

    if remaining:
        _log(f"  WARNING: {len(remaining)} dirs still restoring after {timeout}s "
             f"(proceeding; cold reads possible).")
    else:
        _log(f"  all {len(dirs):,} dirs recalled in {time.time() - t0:.0f}s")
    return len(dirs)


def _cmd_preload(args: argparse.Namespace) -> int:
    exts = None if args.all else list(args.exts)
    # --no-wait => queue restore requests and return immediately (timeout 0),
    # so the FSx/HSM agent recalls server-side while processing runs in parallel.
    wait_timeout = 0 if args.no_wait else args.wait_timeout
    preload_roots(
        args.roots,
        exts,
        args.dir_batch,
        args.poll_interval,
        wait_timeout,
    )
    return 0


# ---------------------------------------------------------------------------
# Audio-size audit of a generated jsonl (single file or shard directory)
# ---------------------------------------------------------------------------
_AUDIO_KEYS = ("audios", "reference_audios", "source_audio")


def _resolve_jsonl_paths(paths: Iterable[str]) -> list[str]:
    out: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.endswith(".jsonl"):
                    fp = os.path.join(p, f)
                    if os.path.isfile(fp) and os.path.getsize(fp) > 0:
                        out.append(fp)
        elif os.path.isfile(p) and os.path.getsize(p) > 0:
            out.append(p)
    return out


def _audio_paths_from_record(rec: dict[str, Any]) -> list[str]:
    """Return the audio file paths referenced by one jsonl record."""
    paths: list[str] = []
    ta = rec.get("target_audio")
    if isinstance(ta, str) and ta:
        paths.append(ta)
    for key in _AUDIO_KEYS:
        v = rec.get(key)
        if isinstance(v, list):
            paths.extend(x for x in v if isinstance(x, str) and x)
        elif isinstance(v, str) and v:
            paths.append(v)
    return paths


def extract_audio_paths_from_jsonl(jsonl_paths: Iterable[str]) -> list[str]:
    """Collect every audio path referenced by the given jsonl(s).

    Accepts a single file or a shard directory (same resolution as
    ``audit_audio_sizes``). Returns deduped, order-preserving paths. Used by the
    post-pipeline task-jsonl prewarm so a single node can warm all referenced
    audio into the FSx cache ahead of training.
    """
    files = _resolve_jsonl_paths(jsonl_paths)
    seen: set[str] = set()
    out: list[str] = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if not isinstance(rec, dict):
                    continue
                for p in _audio_paths_from_record(rec):
                    if p and p not in seen:
                        seen.add(p)
                        out.append(p)
    return out


def audit_audio_sizes(jsonl_paths: Iterable[str], report_file: Optional[str] = None):
    """Sum byte sizes of audio files referenced by ``jsonl_paths``.

    Returns ``(total_referenced_bytes, unique_bytes, n_unique_files, n_refs)``.
    ``total_referenced_bytes`` counts every reference (duplicates included);
    ``unique_bytes`` counts each distinct file once.
    """
    files = _resolve_jsonl_paths(jsonl_paths)
    total_referenced = 0
    unique_total = 0
    seen: set[str] = set()
    n_refs = 0
    missing = 0

    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                paths = _audio_paths_from_record(rec)
                for p in paths:
                    n_refs += 1
                    if p in seen:
                        # distinct file already accounted for in unique_total
                        if os.path.isfile(p):
                            try:
                                total_referenced += os.path.getsize(p)
                            except OSError:
                                pass
                        continue
                    seen.add(p)
                    if os.path.isfile(p):
                        try:
                            sz = os.path.getsize(p)
                        except OSError:
                            sz = 0
                            missing += 1
                        unique_total += sz
                        total_referenced += sz
                    else:
                        missing += 1

    if report_file:
        try:
            os.makedirs(os.path.dirname(report_file) or ".", exist_ok=True)
            with open(report_file, "w", encoding="utf-8") as rf:
                rf.write(f"audio_files_unique={len(seen)}\n")
                rf.write(f"audio_bytes_unique={unique_total}\n")
                rf.write(f"audio_bytes_total_referenced={total_referenced}\n")
                rf.write(f"audio_references={n_refs}\n")
                rf.write(f"audio_missing={missing}\n")
        except OSError as exc:
            _log(f"  WARNING: could not write report {report_file}: {exc}")

    return total_referenced, unique_total, len(seen), n_refs


def _human(n: int) -> str:
    if n >= 1 << 40:
        return f"{n / (1 << 40):.2f} TiB"
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GiB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MiB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.2f} KiB"
    return f"{n} B"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _read_paths(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def _cmd_extract_paths(args: argparse.Namespace) -> int:
    parquets = list(args.parquet)
    if args.parquet_file:
        parquets += _read_paths(args.parquet_file)
    parquets = [p for p in parquets if p]
    seen: set[str] = set()
    for pf in parquets:
        for p in extract_audio_paths(pf, args.kind, args.fsx_mount, args.input_dir):
            if p not in seen:
                seen.add(p)
                print(p)
    return 0


def _cmd_produce(args: argparse.Namespace) -> int:
    """Background producer: restore audio dirs in bulk, record ready dirs.

    Groups the referenced audio paths by parent directory and ``lfs hsm_restore``
    each directory's files in one batched call (splitting huge directories into
    ``--dir-batch`` chunks). This is the efficient lever — one call lets the
    Lustre HSM agent recall the whole directory server-side, instead of per-file
    RPCs. We never wait for one directory's NOOP before queueing the next (so the
    agent's queue stays full and runs at max throughput), but we throttle on FSx
    free space so we never overflow the cache. As each directory reaches NOOP we
    append it to ``--dir-state``; the consumer reads that file (no ``lfs
    hsm_action``) to decide readiness, so it never confuses our files with other
    tenants' cached data.
    """
    paths = _read_paths(args.audio_list)
    dirs: dict[str, set[str]] = {}
    for p in paths:
        dirs.setdefault(os.path.dirname(p), set()).add(p)
    _log(f"produce: {len(paths)} files across {len(dirs)} directories")

    if not _lfs_available():
        _log("lfs unavailable — marking all dirs ready (maker will cold-read).")
        with open(args.dir_state, "a", encoding="utf-8") as sf:
            for d in dirs:
                sf.write(d + "\n")
        return 0

    dir_batch = max(1, args.dir_batch)

    order = list(dirs.items())
    # phase 1: queue restores (directory-batched). No free-space throttle:
    # when the cache is full, Lustre HSM evicts other cached files by LRU on
    # recall ("direct overwrite"), so we never block on remaining capacity.
    for _d, _files in order:
        _fl = list(_files)
        for i in range(0, len(_fl), dir_batch):
            _ch = _fl[i:i + dir_batch]
            try:
                subprocess.run(["lfs", "hsm_restore", *_ch],
                               capture_output=True, text=True, timeout=300)
            except Exception:  # noqa: BLE001
                pass

    # phase 2: poll each directory (batched lfs hsm_action) until all its files
    # are NOOP, recording ready dirs. Use a PER-DIRECTORY deadline (reset per
    # dir) so every directory gets the full wait_timeout to recall, instead of a
    # single global elapsed cap that would starve later directories.
    with open(args.dir_state, "a", encoding="utf-8") as sf:
        for _d, _files in order:
            _fl = list(_files)
            _ready = False
            _deadline = time.time() + args.wait_timeout
            while time.time() < _deadline:
                _non_noop = 0
                for i in range(0, len(_fl), dir_batch):
                    _ch = _fl[i:i + dir_batch]
                    try:
                        _proc = subprocess.run(["lfs", "hsm_action", *_ch],
                                               capture_output=True, text=True, timeout=300)
                        _non_noop += sum(1 for _ln in _proc.stdout.splitlines()
                                         if "NOOP" not in _ln)
                    except Exception:  # noqa: BLE001
                        _non_noop += len(_ch)
                if _non_noop == 0:
                    _ready = True
                    break
                time.sleep(args.poll_interval)
            if _ready:
                _log(f"  dir ready: {_d}")
            else:
                _log(f"  WARNING: dir timed out (proceeding cold): {_d}")
            sf.write(_d + "\n")
            sf.flush()
    _log("produce: all directories processed.")
    return 0


def _cmd_ready(args: argparse.Namespace) -> int:
    """Exit 0 iff every audio path's parent directory is recorded ready.

    Pure file lookup against the producer's ``--dir-state`` file — no
    ``lfs hsm_action`` call, so it never confuses this run's warmed files with
    other tenants' cached data.
    """
    paths = _read_paths(args.audio_list)
    needed = {os.path.dirname(p) for p in paths}
    if not needed:
        return 0
    try:
        with open(args.dir_state, "r", encoding="utf-8") as f:
            ready = set(ln.strip() for ln in f if ln.strip())
    except FileNotFoundError:
        return 1
    return 0 if needed <= ready else 1


def _cmd_release(args: argparse.Namespace) -> int:
    paths = _read_paths(args.list)
    n = release_paths(paths)
    _log(f"released {n}/{len(paths)} files")
    return 0


def _cmd_purge(args: argparse.Namespace) -> int:  # pragma: no cover
    """Deprecated: full release is no longer used at startup (restore overwrites)."""
    _log("purge subcommand is deprecated; restore overwrites the cache directly.")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    total, unique, n_files, n_refs = audit_audio_sizes(args.jsonl, args.report or None)
    _log("AUDIO SIZE AUDIT")
    _log(f"  audio references       = {n_refs:,}")
    _log(f"  unique audio files     = {n_files:,}")
    _log(f"  total referenced bytes = {total:,} ({_human(total)})")
    _log(f"  unique bytes           = {unique:,} ({_human(unique)})")
    if args.report:
        _log(f"  report written to      = {args.report}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="FSx prewarm / release / audit CLI.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ep = sub.add_parser("extract-paths", help="map parquet(s) -> on-FSx audio paths")
    ep.add_argument("--kind", required=True, choices=["tts", "s2s"])
    ep.add_argument("--input-dir", default="")
    ep.add_argument("--fsx-mount", default=DEFAULT_FSX_MOUNT)
    ep.add_argument("--parquet", action="append", default=[],
                    help="parquet file (repeatable); OR use --parquet-file")
    ep.add_argument("--parquet-file", default="",
                    help="file with one parquet path per line")
    ep.set_defaults(func=_cmd_extract_paths)


    pr = sub.add_parser("produce", help="background producer: restore dirs, record ready")
    pr.add_argument("--audio-list", required=True, help="file with one FSx audio path per line")
    pr.add_argument("--dir-state", required=True,
                    help="file recording restored directories (appended as they become ready)")
    pr.add_argument("--fsx-mount", default=DEFAULT_FSX_MOUNT)
    pr.add_argument("--dir-batch", type=int, default=2000,
                    help="max files per lfs hsm_restore call (split large dirs)")
    pr.add_argument("--poll-interval", type=int, default=15)
    pr.add_argument("--wait-timeout", type=int, default=7200)
    pr.set_defaults(func=_cmd_produce)

    rd = sub.add_parser("ready", help="exit 0 iff all audio dirs are recorded ready")
    rd.add_argument("--audio-list", required=True, help="file with one FSx audio path per line")
    rd.add_argument("--dir-state", required=True)
    rd.set_defaults(func=_cmd_ready)

    rp = sub.add_parser("release", help="release a list of audio paths from FSx cache")
    rp.add_argument("--list", required=True, help="file with one FSx audio path per line")
    rp.set_defaults(func=_cmd_release)

    pg = sub.add_parser("purge", help="FULL release everything under a root (ignore free space)")
    pg.add_argument("--root", default=DEFAULT_FSX_MOUNT,
                    help="directory tree to fully evict from FSx cache")
    pg.add_argument("--parallel", type=int, default=32)
    pg.set_defaults(func=_cmd_purge)
    au = sub.add_parser("audit", help="sum audio file sizes referenced by jsonl(s)")
    au.add_argument("--jsonl", nargs="+", required=True)
    au.add_argument("--report", default="")
    au.set_defaults(func=_cmd_audit)

    pl = sub.add_parser(
        "preload",
        help="fully preload (lfs hsm_restore) all audio under given roots",
    )
    pl.add_argument(
        "--roots", nargs="+", required=True,
        help="directories whose audio files to preload, e.g. "
             "TTS_DATA_PATH S2S_DATA_PATH TTS_RAW_PATH_PREFIX",
    )
    pl.add_argument(
        "--exts", nargs="*",
        default=["*.wav", "*.mp3", "*.flac", "*.opus", "*.ogg",
                 "*.aac", "*.m4a", "*.wv", "*.webm"],
        help="file globs to include (default: common audio extensions)",
    )
    pl.add_argument(
        "--all", action="store_true",
        help="preload EVERY file under the roots regardless of extension",
    )
    pl.add_argument("--dir-batch", type=int, default=2000,
                    help="max files per lfs hsm_restore call")
    pl.add_argument("--poll-interval", type=int, default=15)
    pl.add_argument("--wait-timeout", type=int, default=7200,
                    help="max seconds to wait for the full preload to recall")
    pl.add_argument(
        "--no-wait", action="store_true",
        help="queue lfs hsm_restore for all dirs then return immediately; "
             "do NOT poll until recalled. The HSM agent recalls server-side in "
             "the background, overlapping with data processing (use for large "
             "datasets that exceed the FSx cache).",
    )
    pl.set_defaults(func=_cmd_preload)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
