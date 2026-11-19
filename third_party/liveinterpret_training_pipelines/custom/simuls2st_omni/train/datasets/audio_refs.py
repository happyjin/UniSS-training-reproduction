"""Resolve Stage 2 manifest `audio` references to 16 kHz mono waveforms.

Reference forms, chosen so no corpus has to be re-exported as millions of
individual wav files:

    parquet:<relpath>:<row_group>:<row>   audio bytes in a parquet cell
    tar:<relpath>:<member>                audio bytes in a tar member
    pack:<relpath>:<offset>:<length>      a byte range of a concatenated pack
    file:<relpath>                        a plain file, optionally windowed
    gigaspeech:<segment_id>               unresolved segment id

Handles are cached per instance. DataLoader workers each hold their own copy
after fork, so a parquet file or tar index is opened at most once per worker
rather than once per sample; without the cache, throughput collapses because
every `__getitem__` would reopen a multi-GB file.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

SAMPLE_RATE_16K = 16_000


class AudioRefError(ValueError):
    """A manifest audio reference that cannot be decoded."""


class AudioRefResolver:
    """Decode manifest audio references to 16kHz mono float32."""

    def __init__(self, roots: dict[str, Path]):
        """`roots` maps a corpus key to the directory its relpaths resolve against."""
        self.roots = {k: Path(v) for k, v in roots.items()}
        self._tars: dict[str, tarfile.TarFile] = {}
        self._indexes: dict[str, dict[str, tuple[int, int]]] = {}
        self._raw: dict[str, object] = {}
        self._fds: dict[str, int] = {}

    def _root(self, corpus: str) -> Path:
        root = self.roots.get(corpus)
        assert root is not None, f"no root configured for corpus {corpus!r}"
        return root

    @staticmethod
    def _to_mono_16k(wav: torch.Tensor, sr: int) -> np.ndarray:
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE_16K:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE_16K)
        return wav.squeeze(0).numpy().astype(np.float32)

    def _decode_bytes(self, raw: bytes) -> np.ndarray:
        wav, sr = torchaudio.load(io.BytesIO(raw))
        return self._to_mono_16k(wav, sr)

    def _parquet_file(self, path: Path):
        """Open a fresh ParquetFile per read, deliberately not cached.

        pyarrow 25.0.1's ParquetFile retains row-group buffers on the handle:
        a cached handle measured 38 GB of resident (still-referenced) Arrow
        memory after ~1300 single-row reads, while fresh handles held 1.8 GB
        for the same workload. With 32 dataloader workers the cached form
        exhausted the node (measured 1.3 TB anon memory, workers at 56 GB
        each). The footer read a fresh open costs (~1 MB) is noise next to the
        row-group payload it fetches anyway.
        """
        import pyarrow.parquet as pq

        return pq.ParquetFile(path)

    def _tar_file(self, path: Path) -> tarfile.TarFile:
        key = str(path)
        handle = self._tars.get(key)
        if handle is None:
            handle = tarfile.open(path)
            self._tars[key] = handle
        return handle

    def _seek_index(self, path: Path) -> dict[str, tuple[int, int]]:
        """Member offset index written alongside a repacked tar."""
        key = str(path)
        index = self._indexes.get(key)
        if index is None:
            index_path = path.with_suffix(".index.json")
            assert index_path.exists(), f"missing offset index: {index_path}"
            with index_path.open(encoding="utf-8") as f:
                index = {k: (int(v[0]), int(v[1])) for k, v in json.load(f).items()}
            self._indexes[key] = index
        return index

    def _raw_file(self, path: Path):
        key = str(path)
        handle = self._raw.get(key)
        if handle is None:
            handle = path.open("rb")
            self._raw[key] = handle
        return handle

    def resolve(
        self,
        ref: str,
        corpus: str,
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> np.ndarray:
        """Return the waveform `ref` points at, as 16kHz mono float32.

        `offset`/`duration` window a `file:` reference, which WenetSpeech needs
        because its cuts index into long recordings.
        """
        assert ref, "empty audio reference"
        kind, _, rest = ref.partition(":")

        if kind == "parquet":
            relpath, row_group, row = rest.rsplit(":", 2)
            handle = self._parquet_file(self._root(corpus) / relpath)
            table = handle.read_row_group(int(row_group), columns=["audio"])
            cell = table.column("audio")[int(row)].as_py()
            if not (cell and cell.get("bytes")):
                raise AudioRefError(f"{ref}: empty audio cell")
            return self._decode_bytes(cell["bytes"])

        if kind == "tar":
            relpath, member = rest.split(":", 1)
            handle = self._tar_file(self._root(corpus) / relpath)
            extracted = handle.extractfile(member)
            assert extracted is not None, f"{ref}: member not found"
            return self._decode_bytes(extracted.read())

        if kind == "seektar":
            # Uncompressed tar plus a JSON offset index, so a member is one
            # seek+read rather than a walk of the archive.
            relpath, member = rest.split(":", 1)
            path = self._root(corpus) / relpath
            index = self._seek_index(path)
            entry = index.get(member)
            assert entry is not None, f"{ref}: member not in index {path}.index.json"
            offset, size = entry
            handle = self._raw_file(path)
            handle.seek(offset)
            return self._decode_bytes(handle.read(size))

        if kind == "pack":
            relpath, start, size = rest.rsplit(":", 2)
            path = str(self._root(corpus) / relpath)
            fd = self._fds.get(path)
            if fd is None:
                fd = os.open(path, os.O_RDONLY)
                self._fds[path] = fd
            # pread keeps no file position, so a descriptor shared across a
            # DataLoader fork cannot interleave reads between workers.
            raw = os.pread(fd, int(size), int(start))
            if len(raw) != int(size):
                raise AudioRefError(f"{ref}: short read")
            return self._decode_bytes(raw)

        if kind == "file":
            path = self._root(corpus) / rest
            assert path.exists(), f"{ref}: missing file {path}"
            if duration is not None and duration > 0:
                info = torchaudio.info(str(path))
                frame_offset = int(offset * info.sample_rate)
                num_frames = int(duration * info.sample_rate)
                wav, sr = torchaudio.load(
                    str(path), frame_offset=frame_offset, num_frames=num_frames
                )
            else:
                wav, sr = torchaudio.load(str(path))
            return self._to_mono_16k(wav, sr)

        if kind == "gigaspeech":
            raise ValueError(
                f"{ref}: unresolved GigaSpeech segment id; the manifest must "
                "store a parquet/tar/file reference instead."
            )

        raise ValueError(f"unknown audio reference kind {kind!r} in {ref!r}")

    def close(self) -> None:
        for handle in self._tars.values():
            handle.close()
        for handle in self._raw.values():
            handle.close()
        for fd in self._fds.values():
            os.close(fd)
        self._tars.clear()
        self._raw.clear()
        self._fds.clear()
        self._indexes.clear()
