# Copyright (c) ModelScope Contributors. All rights reserved.
"""Resolve indexed ``wds://`` audio references to encoded audio bytes.

The manifests used by the Common Voice S2S pipeline store audio in local tar
shards and refer to each member by byte range, for example::

    wds://curated_audio/audio/processed/shards/.../process_version=clean_v1/shard.tar\
        ?offset=4096&length=12345&format=wav

This is a project-specific indexed WebDataset URI, not a filesystem protocol.

**Path resolution**: The ``wds://`` scheme is replaced by ``/`` and the
``netloc`` (e.g. ``curated_audio``) becomes the first path component.  Thus
``wds://curated_audio/audio/...`` maps directly to
``/curated_audio/audio/...`` on the filesystem.  The container's volumeMount
(pytorchjob.yaml: ``wds-audio`` hostPath → ``/curated_audio``) makes this
1:1 mapping work without any configuration variable.

**Byte-range read**: The resolved path is a **tar archive** (a shard of many
audio members).  The ``offset`` and ``length`` query parameters specify the
exact byte range of the desired member within the tar.  The resolver opens
the tar with ``os.open`` and uses ``os.pread`` to read exactly ``length``
bytes starting at ``offset``, returning the raw encoded WAV/MP3 bytes.  The
existing ms-swift audio loader then performs decoding and resampling.

Configuration (read at call time so tests and launch scripts can override it):

* ``WDS_AUDIO_MODE=auto`` (default): resolve WDS references, pass other inputs
  through unchanged.
* ``WDS_AUDIO_MODE=off``: reject WDS references explicitly.
* ``WDS_AUDIO_MODE=required``: require string audio references to use WDS.
* ``WDS_AUDIO_STRICT=true`` (default): require a ``.tar`` path and a declared
  audio format in addition to the mandatory offset/length integrity checks.
* ``WDS_AUDIO_MAX_BYTES``: maximum accepted member size (default 256 MiB).

Lance source (Step 2 — lance对接训练架构设计.md)
------------------------------------------------
When training reads from the Lance table instead of wds:// tar shards, audio
references in the manifest use the ``lance://`` scheme::

    lance://<_rowid>/source    # source audio (inline bytes preferred, pointer fallback)
    lance://<_rowid>/codes     # target codec codes (2-D [T, C])

Configuration (read at call time):

* ``LANCE_TRAIN_SOURCE_URI``: the ``processed.lance`` URI (S3 or local). Required
  when any ``lance://`` reference is resolved.
* ``LANCE_TRAIN_SOURCE_VERSION``: optional table version (default: latest).
* ``LANCE_WDS_DATA_ROOT``: optional absolute local root for pointer-fallback
  ``*_audio_shard_uri`` (relative keys resolved under this root). Absolute
  shard paths and ``s3://`` URIs are also accepted when a reader is available.
"""

from __future__ import annotations

import functools
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np


_VALID_MODES = {'auto', 'off', 'required'}
_SAFE_COMPONENT_RE = re.compile(r'^[A-Za-z0-9._=-]+$')
_DEFAULT_MAX_BYTES = 256 * 1024 * 1024


class WDSAudioError(ValueError):
    """Raised when an indexed WDS audio reference is invalid or unreadable."""


@dataclass(frozen=True)
class WDSAudioReference:
    """Parsed byte-range reference to a member inside a local tar shard."""

    uri: str
    shard_path: Path            # absolute on-disk path to the .tar file
    offset: int                 # byte offset of the member within the tar
    length: int                 # byte length of the member
    audio_format: Optional[str]  # 'wav', 'mp3', etc.


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def get_wds_audio_mode() -> str:
    """Return and validate the configured resolver mode."""
    mode = os.environ.get('WDS_AUDIO_MODE', 'auto').strip().lower()
    if mode not in _VALID_MODES:
        raise WDSAudioError(
            f'Invalid WDS_AUDIO_MODE={mode!r}; expected one of {sorted(_VALID_MODES)}.')
    return mode


def is_wds_uri(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower().startswith('wds://')


def _single_query_int(query, name: str, uri: str) -> int:
    values = query.get(name)
    if not values or len(values) != 1:
        raise WDSAudioError(f'WDS URI requires exactly one {name!r} query value: {uri}')
    try:
        value = int(values[0])
    except (TypeError, ValueError) as exc:
        raise WDSAudioError(f'WDS URI has non-integer {name}={values[0]!r}: {uri}') from exc
    return value


def _resolve_shard_path(netloc: str, path_parts: Tuple[str, ...]) -> Path:
    """Map wds://<netloc>/<path> to an absolute filesystem path.

    The netloc becomes the first path component, matching the container's
    volumeMount layout (hostPath → /curated_audio).
    """
    return Path('/').joinpath(netloc, *path_parts)


def parse_wds_uri(uri: str, *, strict: Optional[bool] = None) -> WDSAudioReference:
    """Parse a project indexed-WebDataset audio URI into a byte-range reference.

    The URI format is::

        wds://<netloc>/<path...>/<shard.tar>?offset=<int>&length=<int>&format=<ext>

    The ``<netloc>`` + ``<path...>`` part maps 1:1 to a filesystem path via
    the container's volumeMount.  The ``offset`` and ``length`` specify the
    byte range of the audio member within the tar shard.
    """
    if not is_wds_uri(uri):
        raise WDSAudioError(f'Not a wds:// audio URI: {uri!r}')
    if strict is None:
        strict = _env_bool('WDS_AUDIO_STRICT', True)

    parsed = urlparse(uri.strip())
    if parsed.scheme.lower() != 'wds' or not parsed.netloc:
        raise WDSAudioError(f'Invalid wds:// URI: {uri!r}')

    netloc = unquote(parsed.netloc)
    path_parts = tuple(
        unquote(part) for part in Path(unquote(parsed.path)).parts
        if part not in {'/', ''})

    if not path_parts:
        raise WDSAudioError(f'WDS URI has no path after netloc: {uri}')

    # Validate path components (no traversal, no shell metacharacters)
    for part in path_parts:
        if part in {'.', '..'} or not _SAFE_COMPONENT_RE.fullmatch(part):
            raise WDSAudioError(f'Unsafe WDS path component {part!r}: {uri}')

    # The last component must be a .tar file (the shard)
    if strict and not path_parts[-1].lower().endswith('.tar'):
        raise WDSAudioError(f'WDS shard must end in .tar in strict mode: {uri}')

    shard_path = _resolve_shard_path(netloc, path_parts)

    query = parse_qs(parsed.query, keep_blank_values=True)
    offset = _single_query_int(query, 'offset', uri)
    length = _single_query_int(query, 'length', uri)
    if offset < 0:
        raise WDSAudioError(f'WDS offset must be >= 0, got {offset}: {uri}')
    if length <= 0:
        raise WDSAudioError(f'WDS length must be > 0, got {length}: {uri}')

    max_bytes_raw = os.environ.get('WDS_AUDIO_MAX_BYTES', str(_DEFAULT_MAX_BYTES))
    try:
        max_bytes = int(max_bytes_raw)
    except ValueError as exc:
        raise WDSAudioError(f'Invalid WDS_AUDIO_MAX_BYTES={max_bytes_raw!r}.') from exc
    if max_bytes <= 0:
        raise WDSAudioError(f'WDS_AUDIO_MAX_BYTES must be > 0, got {max_bytes}.')
    if length > max_bytes:
        raise WDSAudioError(
            f'WDS member length {length} exceeds WDS_AUDIO_MAX_BYTES={max_bytes}: {uri}')

    formats = query.get('format')
    if formats and len(formats) != 1:
        raise WDSAudioError(f'WDS URI requires at most one format query value: {uri}')
    audio_format = formats[0].strip().lower() if formats else None
    if audio_format == '':
        audio_format = None
    if audio_format is not None and not _SAFE_COMPONENT_RE.fullmatch(audio_format):
        raise WDSAudioError(f'Unsafe WDS audio format {audio_format!r}: {uri}')
    if strict and not audio_format:
        raise WDSAudioError(f'WDS URI requires format=... in strict mode: {uri}')

    return WDSAudioReference(
        uri=uri,
        shard_path=shard_path,
        offset=offset,
        length=length,
        audio_format=audio_format,
    )


def _pread_exact(fd: int, length: int, offset: int) -> bytes:
    """Read exactly *length* bytes from file descriptor *fd* at *offset*."""
    chunks = []
    remaining = length
    cursor = offset
    while remaining:
        chunk = os.pread(fd, remaining, cursor)
        if not chunk:
            break
        chunks.append(chunk)
        cursor += len(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


def read_wds_audio_bytes(uri: str) -> bytes:
    """Read and return exactly one encoded audio member from its local tar shard.

    Opens the tar shard, validates the byte range, and uses ``os.pread`` to
    read exactly ``length`` bytes starting at ``offset``.
    """
    reference = parse_wds_uri(uri)
    shard_path = reference.shard_path

    if not shard_path.is_file():
        raise FileNotFoundError(
            f'WDS shard does not exist: {shard_path} (from {reference.uri})')

    fd = os.open(shard_path, os.O_RDONLY)
    try:
        shard_size = os.fstat(fd).st_size
        end = reference.offset + reference.length
        if end > shard_size:
            raise WDSAudioError(
                f'WDS range [{reference.offset}, {end}) exceeds shard size '
                f'{shard_size}: {shard_path}')
        payload = _pread_exact(fd, reference.length, reference.offset)
    finally:
        os.close(fd)

    if len(payload) != reference.length:
        raise WDSAudioError(
            f'Short WDS read from {shard_path}: expected {reference.length} bytes, '
            f'got {len(payload)} bytes at offset {reference.offset}.')
    return payload


def is_lance_uri(value: Any) -> bool:
    """True if *value* is a ``lance://`` audio/codec reference string."""
    return isinstance(value, str) and value.strip().lower().startswith('lance://')


# ---------------------------------------------------------------------------
# Lance source support (Step 2 — lance对接训练架构设计.md)
#
# Manifest 音频 / 特征引用形如：
#     lance://<_rowid>/source    # source 音频（内联优先，回退 pointer）
#     lance://<_rowid>/codes     # target codec codes（2-D [T, C]）
#     lance://<_rowid>/features  # 预计算 Whisper mel → source_input_features*
# Mel 与 codes 对称：清单自带 ``source_input_features`` 指针（见 lance_manifest.py），
# 训练侧 resolve_* + template._build_precomputed_source_mel_fields 消费。
# curion.storage 仅在调用时 lazy import，避免在未安装 curion 的训练镜像里
# 让 dataset.py 的 import 链整体失败。
# ---------------------------------------------------------------------------

# Column names written by build_lance_input_features_ray.py (mel-first flat).
_LANCE_FEATURE_COLS = (
    'source_input_features',
    'source_feature_attention_mask',
    'source_input_features_frames',
)
_LANCE_FEATURE_SIZE = 128
_LANCE_SPEAKER_COL = 'source_speaker_embedding'
_LANCE_SPEAKER_DIM = 2048

_LANCE_READER = None
_LANCE_READER_PID = None
_LANCE_BYTE_READER = None
# Inherited Lance datasets after fork. Never close() / GC them: the native
# destructor on a forked handle is a common SIGSEGV source.
_FORK_ORPHANS = []
_AFTER_FORK_REGISTERED = False


def discard_forked_lance_reader() -> None:
    """Drop a Lance reader inherited across ``fork()`` without running ``__del__``.

    pylance is not fork-safe. DataLoader workers (PyTorch default: fork) would
    otherwise keep using the parent's mmap / Rust runtime and SIGSEGV under
    concurrent ``take``. Spawn/forkserver cannot be used with DeepSpeed
    (unpickleable engine hooks), so the child must open a fresh dataset.
    """
    global _LANCE_READER, _LANCE_READER_PID
    reader = _LANCE_READER
    _LANCE_READER = None
    _LANCE_READER_PID = None
    if reader is not None:
        _FORK_ORPHANS.append(reader)


def reset_native_io_after_fork() -> None:
    """Child-side hook: discard inherited Lance and shrink Arrow thread pools."""
    discard_forked_lance_reader()
    discard_forked_child_resolver()
    try:
        import pyarrow as pa
        if hasattr(pa, 'set_cpu_count'):
            pa.set_cpu_count(1)
        if hasattr(pa, 'set_io_thread_count'):
            pa.set_io_thread_count(1)
    except Exception:
        pass


def _register_lance_fork_hooks() -> None:
    global _AFTER_FORK_REGISTERED
    if _AFTER_FORK_REGISTERED:
        return
    register = getattr(os, 'register_at_fork', None)
    if register is None:  # pragma: no cover - Windows
        return
    register(after_in_child=reset_native_io_after_fork)
    _AFTER_FORK_REGISTERED = True


_register_lance_fork_hooks()


# --- main-process Lance isolation ------------------------------------------
#
# ``import lance`` installs process-global Rust state (tokio runtime, object
# store cache). A DataLoader worker forked from a process that already loaded
# it inherits a runtime whose threads do not exist in the child; pylance says so
# itself ("lance is not fork-safe ... use spawn or forkserver"). Reopening the
# dataset in the worker is not enough — the global cannot be reset from Python,
# and concurrent ``take`` eventually SIGSEGVs mid-training. DeepSpeed rules out
# spawn/forkserver for the DataLoader (engine hooks are unpickleable), so the
# fix is the other direction: keep the *trainer* free of pylance, and let each
# forked worker import it fresh.
#
# ms-swift resolves one sample in the trainer itself (the ``[INPUT_IDS]`` example
# it logs before training), which is what used to load pylance there. Those
# main-process reads are routed to a long-lived helper child started with
# ``spawn``, so the trainer never imports pylance. DataLoader workers keep
# reading Lance in-process — this adds nothing to the hot path.
_CHILD_ENV_MARKER = 'LANCE_RESOLVER_CHILD'
_CHILD_ENV_MODULE = 'LANCE_RESOLVER_MODULE'
_CHILD_RESOLVER = None
_CHILD_RESOLVER_PID = None
_CHILD_FALLBACK_WARNED = False

# Public resolvers that touch pylance. Wrapped by _install_child_isolation.
_ISOLATED_RESOLVER_NAMES = (
    'resolve_lance_audio',
    'resolve_lance_audio_batch',
    'resolve_lance_codec_codes',
    'resolve_lance_codec_codes_batch',
    'resolve_lance_source_features',
    'resolve_lance_source_features_batch',
    'resolve_lance_speaker_embedding',
    'resolve_lance_speaker_embedding_batch',
    'resolve_lance_training_rows_batch',
)
# name -> undecorated implementation; the helper child dispatches through this.
_ISOLATED_RESOLVERS = {}

_CHILD_BOOTSTRAP = """
import importlib.util, os, pickle, struct, sys
os.environ[{marker!r}] = '1'
_spec = importlib.util.spec_from_file_location(
    '_lance_resolver_child', os.environ[{module!r}])
_mod = importlib.util.module_from_spec(_spec)
# dataclasses resolves annotations through sys.modules[cls.__module__].
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
_in, _out = sys.stdin.buffer, sys.stdout.buffer
while True:
    _hdr = _in.read(4)
    if len(_hdr) < 4:
        break
    _name, _args, _kwargs = pickle.loads(_in.read(struct.unpack('<I', _hdr)[0]))
    try:
        _reply = (True, _mod._ISOLATED_RESOLVERS[_name](*_args, **_kwargs))
    except BaseException as _exc:
        try:
            pickle.dumps(_exc)
        except Exception:
            _exc = RuntimeError(repr(_exc))
        _reply = (False, _exc)
    try:
        _blob = pickle.dumps(_reply, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as _exc:
        _blob = pickle.dumps((False, RuntimeError(repr(_exc))))
    _out.write(struct.pack('<I', len(_blob)))
    _out.write(_blob)
    _out.flush()
""".format(marker=_CHILD_ENV_MARKER, module=_CHILD_ENV_MODULE)


class _ChildResolverError(RuntimeError):
    """Helper-child transport failure (spawn / pipe / protocol), not a data error."""


class _ChildResolver:
    """Runs ``wds_audio`` Lance resolvers in a separate ``spawn``-ed interpreter."""

    def __init__(self) -> None:
        import subprocess
        import threading
        env = dict(os.environ)
        env[_CHILD_ENV_MARKER] = '1'
        env[_CHILD_ENV_MODULE] = os.path.abspath(__file__)
        self._lock = threading.Lock()
        self._proc = subprocess.Popen(
            [sys.executable, '-c', _CHILD_BOOTSTRAP],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
        )

    def call(self, name: str, args: tuple, kwargs: dict):
        import pickle
        import struct
        blob = pickle.dumps((name, args, kwargs), protocol=pickle.HIGHEST_PROTOCOL)
        with self._lock:
            proc = self._proc
            if proc.poll() is not None:
                raise _ChildResolverError(
                    f'lance resolver child exited with {proc.returncode}')
            try:
                proc.stdin.write(struct.pack('<I', len(blob)))
                proc.stdin.write(blob)
                proc.stdin.flush()
                header = proc.stdout.read(4)
                if len(header) < 4:
                    raise _ChildResolverError('lance resolver child closed its pipe')
                reply = proc.stdout.read(struct.unpack('<I', header)[0])
            except (OSError, ValueError) as exc:
                raise _ChildResolverError(f'lance resolver child pipe failed: {exc}') from exc
            ok, payload = pickle.loads(reply)
        if not ok:
            raise payload
        return payload


def discard_forked_child_resolver() -> None:
    """Drop a helper-child handle inherited across ``fork()`` without closing it.

    The pipe and the child belong to the parent; a forked worker must neither
    use nor reap them. Workers read Lance in-process anyway.
    """
    global _CHILD_RESOLVER, _CHILD_RESOLVER_PID
    resolver = _CHILD_RESOLVER
    _CHILD_RESOLVER = None
    _CHILD_RESOLVER_PID = None
    if resolver is not None:
        _FORK_ORPHANS.append(resolver)


def _child_isolation_enabled() -> bool:
    raw = os.environ.get('LANCE_MAIN_PROCESS_ISOLATION', '1').strip().lower()
    return raw not in ('0', 'false', 'no', 'off')


def _in_dataloader_worker() -> bool:
    """True inside a PyTorch DataLoader worker (never imports torch itself)."""
    tud = sys.modules.get('torch.utils.data')
    if tud is None:
        return False
    try:
        return tud.get_worker_info() is not None
    except Exception:
        return False


def _should_isolate_lance() -> bool:
    if os.environ.get(_CHILD_ENV_MARKER) == '1':
        return False  # already inside the helper child
    if not _child_isolation_enabled():
        return False
    return not _in_dataloader_worker()


def _get_child_resolver() -> '_ChildResolver':
    """Long-lived helper child, keyed by PID so a fork never reuses the pipe."""
    global _CHILD_RESOLVER, _CHILD_RESOLVER_PID
    pid = os.getpid()
    if _CHILD_RESOLVER is not None and _CHILD_RESOLVER_PID != pid:
        discard_forked_child_resolver()
    if _CHILD_RESOLVER is None:
        _CHILD_RESOLVER = _ChildResolver()
        _CHILD_RESOLVER_PID = pid
    return _CHILD_RESOLVER


def _make_isolated(fn):
    name = fn.__name__
    # resolve_lance_training_rows_batch resolves in place; the child works on a
    # pickled copy, so its result has to be merged back into the caller's rows.
    merges_rows = name == 'resolve_lance_training_rows_batch'

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # An injected byte_reader is a live callable: it cannot cross to the child.
        if kwargs.get('byte_reader') is not None or not _should_isolate_lance():
            return fn(*args, **kwargs)
        try:
            result = _get_child_resolver().call(name, args, kwargs)
        except _ChildResolverError as exc:
            global _CHILD_FALLBACK_WARNED
            discard_forked_child_resolver()
            if not _CHILD_FALLBACK_WARNED:
                print(
                    f'[wds-audio] lance resolver child unavailable ({exc}); '
                    'resolving in-process. DataLoader workers may then inherit '
                    'pylance state — keep DATALOADER_NUM_WORKERS low.',
                    flush=True)
                _CHILD_FALLBACK_WARNED = True
            return fn(*args, **kwargs)
        if merges_rows and args:
            for original, updated in zip(args[0], result):
                if isinstance(original, dict) and isinstance(updated, dict):
                    original.update(updated)
            return args[0]
        return result

    return wrapper


def _install_child_isolation() -> None:
    for name in _ISOLATED_RESOLVER_NAMES:
        fn = globals()[name]
        _ISOLATED_RESOLVERS[name] = fn
        globals()[name] = _make_isolated(fn)


def _import_lance():
    """Lazy import curion.storage.lance_reader; returns (LanceRowReader, resolve_audio)."""
    try:
        from curion.storage.lance_reader import LanceRowReader, resolve_audio
    except Exception as exc:
        raise ValueError(
            'resolve_lance_audio: 无法 import curion.storage.lance_reader '
            '(训练镜像需 pip install "curion-storage[lance]")：'
            f'{exc}') from exc
    return LanceRowReader, resolve_audio


def _lance_storage_options():
    """Optional object-store options; region from env so S3 opens reliably.

    Does not change which rows/bytes are returned — only how the store is reached.
    """
    region = (
        os.environ.get('LANCE_STORAGE_REGION')
        or os.environ.get('AWS_REGION')
        or os.environ.get('AWS_DEFAULT_REGION')
    )
    if region:
        return {'region': region}
    return None


def _get_lance_reader():
    """Per-process LanceRowReader from env (LANCE_TRAIN_SOURCE_URI / _VERSION).

    The cache is keyed by PID so a DataLoader worker that inherited the
    parent's singleton via ``fork()`` opens its own dataset instead of
    touching the parent's native handle.
    """
    global _LANCE_READER, _LANCE_READER_PID
    pid = os.getpid()
    if _LANCE_READER is not None and _LANCE_READER_PID != pid:
        discard_forked_lance_reader()
    if _LANCE_READER is not None:
        return _LANCE_READER
    uri = os.environ.get('LANCE_TRAIN_SOURCE_URI')
    if not uri:
        raise ValueError(
            'resolve_lance_audio: LANCE_TRAIN_SOURCE_URI 未设置。'
            '训练镜像需 export LANCE_TRAIN_SOURCE_URI=s3://.../processed.lance '
            '（推荐先 cache_lance_local.py sync 到 NVMe 再指向本地路径）')
    version_raw = os.environ.get('LANCE_TRAIN_SOURCE_VERSION')
    version = int(version_raw) if version_raw else None
    LanceRowReader, _ = _import_lance()
    # Prefer kwargs when the installed curion supports storage_options.
    try:
        _LANCE_READER = LanceRowReader(
            uri, version=version, storage_options=_lance_storage_options())
    except TypeError:
        _LANCE_READER = LanceRowReader(uri, version=version)
    _LANCE_READER_PID = pid
    return _LANCE_READER


def _pread_file_range(path: Path, offset: int, length: int) -> bytes:
    """Read exactly ``length`` bytes at ``offset`` from a local file."""
    if not path.is_file():
        raise FileNotFoundError(f'lance pointer shard not found: {path}')
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        end = offset + length
        if end > size:
            raise WDSAudioError(
                f'lance pointer range [{offset}, {end}) exceeds shard size '
                f'{size}: {path}')
        payload = _pread_exact(fd, length, offset)
    finally:
        os.close(fd)
    if len(payload) != length:
        raise WDSAudioError(
            f'Short lance pointer read from {path}: expected {length} bytes, '
            f'got {len(payload)} at offset {offset}.')
    return payload


def _default_lance_byte_reader(shard_uri: str, offset: int, length: int) -> bytes:
    """Pointer-fallback reader for lance rows without inline audio bytes.

    Resolution order:
      1. Absolute local path that exists.
      2. ``LANCE_WDS_DATA_ROOT`` as ``s3://`` / ``file://`` object-store root
         + relative ``shard_uri``.
      3. ``LANCE_WDS_DATA_ROOT`` as local directory + relative ``shard_uri``
         (prefers curion ``WdsTarShardReader``).
    """
    if offset < 0 or length <= 0:
        raise ValueError(
            f'invalid lance pointer range offset={offset} length={length} '
            f'uri={shard_uri!r}')

    uri = (shard_uri or '').strip()
    if not uri:
        raise ValueError('lance pointer shard_uri is empty')

    direct = Path(uri)
    if direct.is_file():
        return _pread_file_range(direct, int(offset), int(length))

    root = os.environ.get('LANCE_WDS_DATA_ROOT', '').strip()
    if not root:
        raise ValueError(
            f'lance pointer fallback needs inline bytes, an absolute shard path, '
            f'or LANCE_WDS_DATA_ROOT; got shard_uri={uri!r}.')

    if root.startswith(('s3://', 'file://')):
        try:
            from curion.storage.object_store import resolve_object_store
            from curion.storage.shard_reader import ObjectStoreWdsTarShardReader
            store = resolve_object_store(root)
            return ObjectStoreWdsTarShardReader(store).read_payload(
                uri, int(offset), int(length))
        except Exception as exc:
            raise ValueError(
                f'lance pointer object-store fallback failed '
                f'(root={root!r}, shard={uri!r}): {exc}'
            ) from exc

    root_path = Path(root).expanduser().resolve()
    try:
        from curion.storage.shard_reader import WdsTarShardReader
        return WdsTarShardReader(root_path).read_payload(
            uri, int(offset), int(length))
    except ImportError:
        candidate = (root_path / uri).resolve()
        try:
            candidate.relative_to(root_path)
        except ValueError as exc:
            raise ValueError(
                f'lance pointer shard_uri escapes LANCE_WDS_DATA_ROOT: '
                f'{uri!r}') from exc
        return _pread_file_range(candidate, int(offset), int(length))


def _get_lance_byte_reader():
    """Singleton default byte_reader for pointer-only lance audio rows."""
    global _LANCE_BYTE_READER
    if _LANCE_BYTE_READER is None:
        _LANCE_BYTE_READER = _default_lance_byte_reader
    return _LANCE_BYTE_READER


def _parse_lance_uri(uri: str):
    """Parse ``lance://<_rowid>[/side]`` -> (rowid: int, side: str)."""
    body = uri.strip()[len('lance://'):]
    parts = body.split('/')
    if not parts or not parts[0]:
        raise ValueError(f'Invalid lance:// URI (missing rowid): {uri}')
    try:
        rowid = int(parts[0])
    except ValueError as exc:
        raise ValueError(f'Invalid lance:// URI (rowid not int): {uri}') from exc
    side = parts[1] if len(parts) > 1 and parts[1] else 'source'
    return rowid, side


def _unique_sorted_rowids(rowids):
    """Dedup + sort so Lance ``IN`` / take can scan one fragment at a time.

    Result tensors are re-aligned by ``_rowid``, so this does not change
    which rows come back or their values.
    """
    return sorted({int(r) for r in rowids})


def _cell_as_py(cell: Any):
    if cell is None:
        return None
    if hasattr(cell, 'as_py'):
        try:
            if getattr(cell, 'is_valid', True) is False:
                return None
        except Exception:
            pass
        return cell.as_py()
    return cell


def _cell_to_1d(cell: Any, *, dtype):
    """Arrow list/scalar or Python sequence → 1-D numpy (no ``to_pylist``)."""
    if cell is None:
        return None
    values = getattr(cell, 'values', None)
    if values is not None and hasattr(values, 'to_numpy'):
        try:
            if getattr(cell, 'is_valid', True) is False:
                return None
            return np.asarray(values.to_numpy(zero_copy_only=False), dtype=dtype)
        except Exception:
            pass
    py = _cell_as_py(cell)
    if py is None:
        return None
    return np.asarray(py, dtype=dtype).reshape(-1)


def _cell_to_codes(cell: Any):
    """Arrow nested list or Python codes → 2-D ``[T, C]`` int16 array."""
    if cell is None:
        return None
    values = getattr(cell, 'values', None)
    if values is not None and hasattr(values, '__len__'):
        try:
            if getattr(cell, 'is_valid', True) is False:
                return None
            rows = []
            for i in range(len(values)):
                inner = values[i]
                if inner is None or getattr(inner, 'is_valid', True) is False:
                    continue
                inner_vals = getattr(inner, 'values', None)
                if inner_vals is not None and hasattr(inner_vals, 'to_numpy'):
                    rows.append(np.asarray(
                        inner_vals.to_numpy(zero_copy_only=False)))
                else:
                    rows.append(np.asarray(_cell_as_py(inner)))
            if not rows:
                return None
            return np.stack(rows, axis=0)
        except Exception:
            pass
    py = _cell_as_py(cell)
    if py is None:
        return None
    arr = np.asarray(py)
    if arr.dtype == object:
        arr = np.asarray([list(x) for x in py])
    return arr


def _rows_from_take_table(table) -> dict:
    """Index a take result by stable ``_rowid``.

    Prefers per-column Arrow access so mel/speaker stay numpy. Fake tables
    used in tests only implement ``to_pylist`` and fall back.
    """
    if table is None or getattr(table, 'num_rows', 0) == 0:
        return {}
    schema = getattr(table, 'schema', None)
    names = list(getattr(schema, 'names', []) or [])
    if names and hasattr(table, 'column'):
        try:
            cols = {name: table.column(name) for name in names}
            out = {}
            for i in range(table.num_rows):
                rid_cell = cols['_rowid'][i]
                rid = int(_cell_as_py(rid_cell))
                row = {'_rowid': rid}
                for name in names:
                    if name == '_rowid':
                        continue
                    cell = cols[name][i]
                    if name == 'source_input_features':
                        row[name] = _cell_to_1d(cell, dtype=np.float32)
                    elif name == 'source_feature_attention_mask':
                        row[name] = _cell_to_1d(cell, dtype=np.int8)
                    elif name == 'source_speaker_embedding':
                        row[name] = _cell_to_1d(cell, dtype=np.float32)
                    elif name == 'codes':
                        row[name] = _cell_to_codes(cell)
                    else:
                        row[name] = _cell_as_py(cell)
                out[rid] = row
            return out
        except Exception:
            pass
    return {
        int(row['_rowid']): row
        for row in table.to_pylist()
    }


def _pack_source_features(row: dict) -> dict:
    """Validate one Lance row → ``{flat, mask, frames, feature_size}``.

    ``flat`` / ``mask`` stay numpy when possible so encode can
    ``torch.as_tensor`` without a Python-float roundtrip.
    """
    flat = row.get('source_input_features')
    mask = row.get('source_feature_attention_mask')
    frames = row.get('source_input_features_frames')
    if flat is None or frames is None:
        raise ValueError(
            'lance features row missing source_input_features / '
            f'source_input_features_frames: keys={sorted(row)}')
    t = int(frames)
    if t <= 0:
        raise ValueError(f'lance features frames must be >0, got {frames!r}')
    flat = np.asarray(flat, dtype=np.float32).reshape(-1)
    if flat.size != _LANCE_FEATURE_SIZE * t:
        raise ValueError(
            f'lance features flat length {int(flat.size)} != '
            f'{_LANCE_FEATURE_SIZE}*{t} (mel-first layout)')
    if mask is None:
        mask = np.ones(t, dtype=np.int8)
    else:
        mask = np.asarray(mask).reshape(-1)
        if mask.size != t:
            padded = np.ones(t, dtype=mask.dtype)
            n = min(int(mask.size), t)
            padded[:n] = mask[:n]
            mask = padded
    return {
        'flat': flat,
        'mask': mask,
        'frames': t,
        'feature_size': _LANCE_FEATURE_SIZE,
    }


def _pack_speaker_embedding(emb, *, uri: str = '', rowid=None):
    if emb is None:
        where = uri or (f'rowid={rowid}' if rowid is not None else '')
        raise ValueError(f'lance:// speaker has null {_LANCE_SPEAKER_COL}: {where}')
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    if emb.size != _LANCE_SPEAKER_DIM:
        raise ValueError(
            f'lance speaker dim {int(emb.size)} != {_LANCE_SPEAKER_DIM}'
            + (f' for {uri}' if uri else f' rowid={rowid}'))
    return emb


def resolve_lance_source_features(uri: str) -> dict:
    """Resolve ``lance://<_rowid>/features`` → mel dict (flat/mask/frames)."""
    rowid, side = _parse_lance_uri(uri)
    if side not in ('features', 'mel', 'input_features'):
        raise ValueError(
            f'resolve_lance_source_features 仅接受 lance://<_rowid>/features，得到: {uri}')
    reader = _get_lance_reader()
    tbl = reader.take_by_stable_rowid([rowid], columns=list(_LANCE_FEATURE_COLS))
    if tbl.num_rows == 0:
        raise ValueError(f'lance:// features row {rowid} not found: {uri}')
    packed = _rows_from_take_table(tbl).get(int(rowid))
    if packed is None:
        raise ValueError(f'lance:// features row {rowid} not found: {uri}')
    return _pack_source_features(packed)


def resolve_lance_speaker_embedding(uri: str):
    """Resolve ``lance://<_rowid>/speaker`` → 2048-d float list (ECAPA x-vector)."""
    rowid, side = _parse_lance_uri(uri)
    if side not in ('speaker', 'speaker_embedding', 'xvector', 'xvec'):
        raise ValueError(
            f'resolve_lance_speaker_embedding 仅接受 lance://<_rowid>/speaker，得到: {uri}')
    reader = _get_lance_reader()
    tbl = reader.take_by_stable_rowid([rowid], columns=[_LANCE_SPEAKER_COL])
    if tbl.num_rows == 0:
        raise ValueError(f'lance:// speaker row {rowid} not found: {uri}')
    packed = _rows_from_take_table(tbl).get(int(rowid))
    if packed is None:
        raise ValueError(f'lance:// speaker row {rowid} not found: {uri}')
    return _pack_speaker_embedding(
        packed.get(_LANCE_SPEAKER_COL), uri=uri, rowid=rowid)


def resolve_lance_speaker_embedding_batch(uris):
    """Batch-resolve ``lance://<_rowid>/speaker`` (one coalesced take)."""
    parsed = []
    for i, uri in enumerate(uris):
        if uri is None or uri == '':
            parsed.append((i, None, uri))
            continue
        if isinstance(uri, (list, tuple)) and len(uri) == _LANCE_SPEAKER_DIM:
            parsed.append((i, None, list(uri)))
            continue
        if not is_lance_uri(uri):
            parsed.append((i, None, uri))
            continue
        rowid, side = _parse_lance_uri(uri)
        if side not in ('speaker', 'speaker_embedding', 'xvector', 'xvec'):
            raise ValueError(
                f'resolve_lance_speaker_embedding_batch 仅接受 speaker URI，得到: {uri}')
        parsed.append((i, rowid, uri))

    rowids = [rowid for _, rowid, _ in parsed if rowid is not None]
    by_id = {}
    if rowids:
        reader = _get_lance_reader()
        unique = _unique_sorted_rowids(rowids)
        tbl = reader.take_by_stable_rowid(unique, columns=[_LANCE_SPEAKER_COL])
        for rid, row in _rows_from_take_table(tbl).items():
            emb = row.get(_LANCE_SPEAKER_COL)
            if emb is None:
                continue
            by_id[int(rid)] = _pack_speaker_embedding(emb, rowid=rid)

    out = [None] * len(uris)
    for i, rowid, uri in parsed:
        if rowid is None:
            out[i] = uri
        elif rowid not in by_id:
            raise ValueError(f'lance:// speaker row {rowid} not found: {uri}')
        else:
            out[i] = by_id[rowid]
    return out


def resolve_lance_source_features_batch(uris):
    """Batch-resolve ``lance://<_rowid>/features`` (one coalesced take).

    Non-lance / empty / already-resolved dict entries pass through unchanged.
    """
    parsed = []
    for i, uri in enumerate(uris):
        if uri is None or uri == '':
            parsed.append((i, None, uri))
            continue
        if isinstance(uri, dict) and 'flat' in uri and 'frames' in uri:
            parsed.append((i, None, uri))
            continue
        if not is_lance_uri(uri):
            parsed.append((i, None, uri))
            continue
        rowid, side = _parse_lance_uri(uri)
        if side not in ('features', 'mel', 'input_features'):
            raise ValueError(
                f'resolve_lance_source_features_batch 仅接受 features URI，得到: {uri}')
        parsed.append((i, rowid, uri))

    rowids = [rowid for _, rowid, _ in parsed if rowid is not None]
    by_id = {}
    if rowids:
        reader = _get_lance_reader()
        unique = _unique_sorted_rowids(rowids)
        tbl = reader.take_by_stable_rowid(unique, columns=list(_LANCE_FEATURE_COLS))
        for rid, row in _rows_from_take_table(tbl).items():
            by_id[int(rid)] = _pack_source_features(row)

    out = [None] * len(uris)
    for i, rowid, uri in parsed:
        if rowid is None:
            out[i] = uri
            continue
        packed = by_id.get(rowid)
        if packed is None:
            raise ValueError(f'lance:// features row {rowid} not found: {uri}')
        out[i] = packed
    return out


def resolve_lance_training_rows_batch(rows):
    """Resolve hot-path Lance fields for a DataLoader batch in one table take.

    ``rows`` is the list of lightweight HF rows selected by one DataLoader
    batch.  Any unresolved ``source_input_features``,
    ``source_speaker_embedding`` and ``target_codec_codes`` pointers are
    collected first, then fetched with one ``take_by_stable_rowid`` over the
    union of row ids and columns.  Results are written back in place, aligned
    by stable ``_rowid`` because Lance does not promise take-result ordering.

    Already materialized and non-Lance values pass through unchanged.  This
    keeps the helper safe with old Arrow caches where codes/speaker were
    resolved during ``datasets.map``.
    """
    if not rows:
        return rows

    requests = []
    rowids = []
    need_features = False
    need_speaker = False
    need_codes = False
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        for key, allowed_sides in (
            ('source_input_features', ('features', 'mel', 'input_features')),
            ('source_speaker_embedding',
             ('speaker', 'speaker_embedding', 'xvector', 'xvec')),
            ('target_codec_codes', ('codes',)),
        ):
            value = row.get(key)
            if not is_lance_uri(value):
                continue
            rowid, side = _parse_lance_uri(value)
            if side not in allowed_sides:
                raise ValueError(
                    f'{key} 需要 lance://.../{allowed_sides[0]}，得到: {value}')
            requests.append((row_index, key, rowid, value))
            rowids.append(rowid)
            need_features = need_features or key == 'source_input_features'
            need_speaker = need_speaker or key == 'source_speaker_embedding'
            need_codes = need_codes or key == 'target_codec_codes'

    if not requests:
        return rows

    columns = []
    if need_features:
        columns.extend(_LANCE_FEATURE_COLS)
    if need_speaker:
        columns.append(_LANCE_SPEAKER_COL)
    if need_codes:
        columns.append('codes')

    reader = _get_lance_reader()
    unique_rowids = _unique_sorted_rowids(rowids)
    table = reader.take_by_stable_rowid(unique_rowids, columns=columns)
    by_rowid = _rows_from_take_table(table)

    for row_index, key, rowid, uri in requests:
        lance_row = by_rowid.get(rowid)
        if lance_row is None:
            raise ValueError(f'lance:// row {rowid} not found: {uri}')
        if key == 'source_input_features':
            value = _pack_source_features(lance_row)
        elif key == 'source_speaker_embedding':
            value = _pack_speaker_embedding(
                lance_row.get(_LANCE_SPEAKER_COL), uri=uri, rowid=rowid)
        else:
            value = lance_row.get('codes')
            if value is None:
                raise ValueError(f'lance:// row {rowid} 的 codes 为空: {uri}')
            value = np.asarray(value)
            if value.size == 0:
                raise ValueError(f'lance:// row {rowid} 的 codes 为空: {uri}')
        rows[row_index][key] = value
    return rows


def resolve_lance_audio(uri: str, *, byte_reader=None) -> bytes:
    """Resolve ``lance://<_rowid>[/source|target]`` to encoded audio bytes.

    Inline ``*_audio_bytes`` wins; otherwise pointer columns are read via
    ``byte_reader`` (defaults to :func:`_default_lance_byte_reader`).
    """
    rowid, side = _parse_lance_uri(uri)
    if side == 'codes':
        raise ValueError(
            f'resolve_lance_audio 不接受 codes URI（用 resolve_lance_codec_codes）: {uri}')
    if side in ('features', 'mel', 'input_features'):
        raise ValueError(
            f'resolve_lance_audio 不接受 features URI'
            f'（用 resolve_lance_source_features）: {uri}')
    reader = _get_lance_reader()
    cols = [
        f'{side}_audio_bytes',
        f'{side}_audio_shard_uri',
        f'{side}_audio_byte_offset',
        f'{side}_audio_byte_length',
    ]
    tbl = reader.take_by_stable_rowid([rowid], columns=cols)
    if tbl.num_rows == 0:
        raise ValueError(f'lance:// row {rowid} not found: {uri}')
    row = tbl.to_pylist()[0]
    _, resolve_audio = _import_lance()
    payload = resolve_audio(
        row, side, byte_reader=byte_reader or _get_lance_byte_reader())
    if payload is None:
        raise ValueError(
            f'lance:// row {rowid} side={side} 既无内联字节也无可用指针: {uri}')
    return payload


def resolve_lance_codec_codes(uri: str):
    """Resolve ``lance://<_rowid>/codes`` to a 2-D [T, C] codes value (list of lists)."""
    rowid, side = _parse_lance_uri(uri)
    if side != 'codes':
        raise ValueError(
            f'resolve_lance_codec_codes 仅接受 lance://<_rowid>/codes，得到: {uri}')
    reader = _get_lance_reader()
    tbl = reader.take_by_stable_rowid([rowid], columns=['codes'])
    if tbl.num_rows == 0:
        raise ValueError(f'lance:// codes row {rowid} not found: {uri}')
    packed = _rows_from_take_table(tbl).get(int(rowid))
    codes = None if packed is None else packed.get('codes')
    if codes is None:
        raise ValueError(f'lance:// row {rowid} 的 codes 为空: {uri}')
    return np.asarray(codes)


def resolve_lance_codec_codes_batch(uris):
    """Batch-resolve ``lance://<_rowid>/codes`` URIs (one coalesced take).

    Returns a list aligned with ``uris``. Non-lance / empty entries are returned
    unchanged. Use this from batched preprocess to avoid per-row point queries.
    """
    parsed = []
    for i, uri in enumerate(uris):
        if not is_lance_uri(uri):
            parsed.append((i, None, uri))
            continue
        rowid, side = _parse_lance_uri(uri)
        if side != 'codes':
            raise ValueError(
                f'resolve_lance_codec_codes_batch 仅接受 codes URI，得到: {uri}')
        parsed.append((i, rowid, uri))

    rowids = [rowid for _, rowid, _ in parsed if rowid is not None]
    by_rowid = {}
    if rowids:
        reader = _get_lance_reader()
        unique = _unique_sorted_rowids(rowids)
        tbl = reader.take_by_stable_rowid(unique, columns=['codes'])
        for rid, row in _rows_from_take_table(tbl).items():
            by_rowid[int(rid)] = row.get('codes')

    out = [None] * len(uris)
    for i, rowid, uri in parsed:
        if rowid is None:
            out[i] = uri
            continue
        codes = by_rowid.get(rowid)
        if codes is None:
            raise ValueError(f'lance:// codes row {rowid} not found: {uri}')
        out[i] = np.asarray(codes)
    return out


def resolve_lance_audio_batch(uris, *, byte_reader=None):
    """Batch-resolve ``lance://<_rowid>/source|target`` URIs (coalesced take).

    Groups URIs by side so each side issues **one** ``take_by_stable_rowid``
    over the unique rowids (mirrors :func:`resolve_lance_codec_codes_batch`).
    Non-lance / empty entries pass through unchanged. ``codes`` URIs raise —
    use :func:`resolve_lance_codec_codes_batch` for those.
    """
    parsed = []  # (index, rowid|None, side|None, uri)
    for i, uri in enumerate(uris):
        if not is_lance_uri(uri):
            parsed.append((i, None, None, uri))
            continue
        rowid, side = _parse_lance_uri(uri)
        if side == 'codes':
            raise ValueError(
                f'resolve_lance_audio_batch 不接受 codes URI'
                f'（用 resolve_lance_codec_codes_batch）: {uri}')
        if side in ('features', 'mel', 'input_features'):
            raise ValueError(
                f'resolve_lance_audio_batch 不接受 features URI'
                f'（用 resolve_lance_source_features_batch）: {uri}')
        if side not in ('source', 'target'):
            raise ValueError(
                f'resolve_lance_audio_batch 仅接受 source/target side，得到: {uri}')
        parsed.append((i, rowid, side, uri))

    by_key = {}  # (side, rowid) -> bytes
    sides_needed = sorted({side for _, rowid, side, _ in parsed if rowid is not None})
    if sides_needed:
        reader = _get_lance_reader()
        _, resolve_audio = _import_lance()
        br = byte_reader or _get_lance_byte_reader()
        for side in sides_needed:
            rowids = [rowid for _, rowid, s, _ in parsed if rowid is not None and s == side]
            unique = _unique_sorted_rowids(rowids)
            cols = [
                f'{side}_audio_bytes',
                f'{side}_audio_shard_uri',
                f'{side}_audio_byte_offset',
                f'{side}_audio_byte_length',
            ]
            tbl = reader.take_by_stable_rowid(unique, columns=cols)
            for row in tbl.to_pylist():
                rid = int(row['_rowid'])
                payload = resolve_audio(row, side, byte_reader=br)
                if payload is None:
                    raise ValueError(
                        f'lance:// row {rid} side={side} 既无内联字节也无可用指针')
                by_key[(side, rid)] = payload

    out = [None] * len(uris)
    for i, rowid, side, uri in parsed:
        if rowid is None:
            out[i] = uri
            continue
        payload = by_key.get((side, rowid))
        if payload is None:
            raise ValueError(f'lance:// audio row {rowid} side={side} not found: {uri}')
        out[i] = payload
    return out


def resolve_audio_reference(audio: Any) -> Any:
    """Resolve WDS/lance references to bytes and preserve legacy audio inputs.

    Non-string inputs (already decoded arrays or encoded bytes) always pass
    through.  ``lance://`` is an independent scheme resolved directly from the
    Lance table and is unaffected by ``WDS_AUDIO_MODE``.  In ``required`` mode,
    ordinary string paths are rejected so a new WDS-only run can detect
    accidental legacy rows early.
    """
    if is_lance_uri(audio):
        return resolve_lance_audio(audio)
    mode = get_wds_audio_mode()
    if is_wds_uri(audio):
        if mode == 'off':
            raise WDSAudioError(
                f'Encountered wds:// audio while WDS_AUDIO_MODE=off: {audio}')
        return read_wds_audio_bytes(audio)
    if mode == 'required' and isinstance(audio, str):
        raise WDSAudioError(
            f'WDS_AUDIO_MODE=required but audio is not a wds:// reference: {audio}')
    return audio


def resolve_audio_references(audios):
    """Resolve a list of audio refs; coalesces all ``lance://`` takes in one go.

    Non-lance entries still go through :func:`resolve_audio_reference` one by
    one (WDS / path / already-bytes). Empty / ``None`` input returns ``[]``.
    """
    if not audios:
        return []
    items = list(audios)
    lance_idx = [i for i, a in enumerate(items) if is_lance_uri(a)]
    if lance_idx:
        resolved = resolve_lance_audio_batch([items[i] for i in lance_idx])
        for i, payload in zip(lance_idx, resolved):
            items[i] = payload
    # Remaining non-bytes still need WDS / path / mode handling.
    return [
        a if isinstance(a, (bytes, bytearray)) else resolve_audio_reference(a)
        for a in items
    ]


_install_child_isolation()


__all__ = [
    'WDSAudioError',
    'WDSAudioReference',
    'get_wds_audio_mode',
    'is_wds_uri',
    'is_lance_uri',
    'parse_wds_uri',
    'read_wds_audio_bytes',
    'resolve_audio_reference',
    'resolve_audio_references',
    'resolve_lance_audio',
    'resolve_lance_audio_batch',
    'resolve_lance_codec_codes',
    'resolve_lance_codec_codes_batch',
    'resolve_lance_source_features',
    'resolve_lance_source_features_batch',
    'resolve_lance_speaker_embedding',
    'resolve_lance_speaker_embedding_batch',
    'resolve_lance_training_rows_batch',
    'discard_forked_lance_reader',
    'discard_forked_child_resolver',
    'reset_native_io_after_fork',
]
