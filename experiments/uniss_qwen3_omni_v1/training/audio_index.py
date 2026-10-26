"""Read side of the audio index: the same rows, without the JSON.

``read_audio_rows`` returns a list of ``AudioRow``. This returns something
that answers to the same two things ``Stage2Corpus`` asks of it -- a
length and ``[i]`` -- while the text stays memory-mapped and only the
drawn row is decoded. Eight ranks then start in seconds rather than each
parsing 19,186,799 lines.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.uniss_omni_s2st_v1.training.stage2_data import AudioRow


def _blob(path: Path, name: str):
    """A utf-8 blob and the offsets of its NUL-separated fields."""
    raw = np.load(path / f"{name}.npy", mmap_mode="r")
    data = bytes(raw) if raw.size < (1 << 26) else raw
    return data


class AudioIndex:
    """Indexable over the rows build_audio_index kept."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        self.rows = int(self.meta["rows"])
        self._fields = {}
        for name in ("uri", "src_text", "tgt_text", "langs", "parquet"):
            raw = np.load(self.root / f"{name}.npy", mmap_mode="r")
            text = bytes(raw).decode("utf-8")
            parts = text.split("\x00")
            if len(parts) != self.rows:
                raise ValueError(
                    f"{name} holds {len(parts)} fields, meta says {self.rows} rows")
            self._fields[name] = parts
        self.row_index = np.load(self.root / "row_index.npy")
        if self.row_index.size != self.rows:
            raise ValueError("row_index length disagrees with meta")
        # Optional, written by data_clean/build_clean_index.py: the source's
        # duration (so a run without the Thinker need not decode audio to
        # know it) and the source GLM tokens re-tokenised from the local
        # audio, which is what inference sees.
        self.source_ms = np.load(self.root / "source_ms.npy") if (self.root / "source_ms.npy").exists() else None
        glm = self.root / "teacher_glm_values.npy"
        self._teacher_glm = (np.load(glm, mmap_mode="r"), np.load(self.root / "teacher_glm_offsets.npy")) \
            if glm.exists() else None

    def teacher_glm(self, i: int) -> list[int]:
        values, offsets = self._teacher_glm
        return values[offsets[i] : offsets[i + 1]].tolist()

    def parquet_names(self) -> set[str]:
        """The parquet files these rows point at, without building rows.

        ParquetCodeIndex needs the set; constructing 19 million AudioRow
        objects to collect it would undo the point of the index.
        """
        return set(self._fields["parquet"])

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, i: int) -> AudioRow:
        i = int(i)
        src, tgt = self._fields["langs"][i].split("|", 1)
        return AudioRow(
            audio_uri=self._fields["uri"][i],
            source_transcript=self._fields["src_text"][i],
            target_text=self._fields["tgt_text"][i],
            source_lang=src,
            target_lang=tgt,
            parquet=self._fields["parquet"][i],
            row_index=int(self.row_index[i]),
        )
