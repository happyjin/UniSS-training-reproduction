"""Stage-2 corpus: audio-backed rows for the speech tasks, cache for the rest.

Two pools, because they are not the same size. ASR, S2TT and S2ST need the
source waveform, and only the shards whose tars were fetched have one --
969,266 utterances, about 2,000 hours. MT and TTS need text and codes
only, and the prebuilt cache holds 16,184,517 of those. Sampling the five
tasks at the paper's ratio out of one pool would throw away 94% of what is
available to the text tasks.

Audio rows are read once into a compact index: the wds:// byte range, the
transcript, the translation and the provenance needed to recover codes and
speaker tokens. The waveform itself is decoded per batch in a dataloader
worker -- 2,000 hours of FLAC is 114 GB and does not belong in RAM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from experiments.uniss_omni_s2st_v1.data.wds_resolver import (
    available_shards,
    parse_wds_uri,
    read_audio,
)

PARQUET_COLUMNS = ["id", "target_bicodec", "bicodec_global"]


@dataclass
class AudioRow:
    audio_uri: str
    source_transcript: str
    target_text: str
    source_lang: str
    target_lang: str
    parquet: str
    row_index: int


def read_audio_rows(
    manifests: list[str | Path],
    *,
    audio_root: str | Path,
    limit: int = 0,
    task: str = "speech_to_speech_translation",
) -> list[AudioRow]:
    """Manifest rows whose tar is present locally."""
    present = available_shards(audio_root)
    rows: list[AudioRow] = []
    for manifest in manifests:
        with Path(manifest).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if task and record.get("task") != task:
                    continue
                audios = record.get("audios") or []
                if not audios:
                    continue
                if parse_wds_uri(audios[0]).shard not in present:
                    continue
                provenance = record.get("provenance") or {}
                rows.append(
                    AudioRow(
                        audio_uri=audios[0],
                        source_transcript=record.get("source_transcript") or "",
                        target_text=record.get("target_text") or "",
                        source_lang=record["source_lang"],
                        target_lang=record["target_lang"],
                        parquet=provenance["source_parquet"],
                        row_index=int(provenance["source_row_index"]),
                    )
                )
                if limit and len(rows) >= limit:
                    return rows
    return rows


class ParquetCodeIndex:
    """Codes and speaker tokens for the audio rows, keyed by provenance.

    Loaded per parquet and held as flat arrays: the ten shards together are
    2.5M rows of codes, which is 1.3 GB as int16 and far more as Python.
    """

    def __init__(self, parquet_root: str | Path, names: set[str],
                 *, with_source: bool = False, with_glm: bool = False):
        """``with_source`` also loads each row's own audio codes.

        The a2s task supervises the source utterance rather than the
        translation, so it needs source_bicodec. Left False nothing is
        read that was not read before, and the extra array costs roughly
        what target_bicodec costs -- about 10 GiB per rank across all 198
        shards, against 1,992 GiB of memory.

        ``with_glm`` also loads each row's source_glm, the GLM-4 content
        tokens UniSS reads the source speech as (12.5 a second, a quarter
        of what source_bicodec costs). Left False nothing changes.
        """
        self.codes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.globals: dict[str, np.ndarray] = {}
        self.source_codes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.glm: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.with_source = bool(with_source)
        self.with_glm = bool(with_glm)
        columns = (list(PARQUET_COLUMNS) + (["source_bicodec"] if with_source else [])
                   + (["source_glm"] if with_glm else []))
        for name in sorted(names):
            table = pq.read_table(Path(parquet_root) / name, columns=columns)
            codes = table["target_bicodec"].combine_chunks()
            self.codes[name] = (
                codes.values.to_numpy(zero_copy_only=False).astype(np.int16),
                codes.offsets.to_numpy().astype(np.int64),
            )
            globals_ = table["bicodec_global"].combine_chunks()
            self.globals[name] = (
                globals_.values.to_numpy(zero_copy_only=False)
                .astype(np.int16)
                .reshape(-1, 32)
            )
            if with_source:
                src = table["source_bicodec"].combine_chunks()
                self.source_codes[name] = (
                    src.values.to_numpy(zero_copy_only=False).astype(np.int16),
                    src.offsets.to_numpy().astype(np.int64),
                )
            if with_glm:
                glm = table["source_glm"].combine_chunks()
                self.glm[name] = (
                    # GLM-4 ids run to 16,383, inside int16's range.
                    glm.values.to_numpy(zero_copy_only=False).astype(np.int16),
                    glm.offsets.to_numpy().astype(np.int64),
                )

    def get(self, parquet: str, index: int) -> tuple[np.ndarray, np.ndarray]:
        values, offsets = self.codes[parquet]
        return values[offsets[index] : offsets[index + 1]], self.globals[parquet][index]

    def get_source(self, parquet: str, index: int) -> np.ndarray:
        values, offsets = self.source_codes[parquet]
        return values[offsets[index] : offsets[index + 1]]

    def get_glm(self, parquet: str, index: int) -> np.ndarray:
        values, offsets = self.glm[parquet]
        return values[offsets[index] : offsets[index + 1]]


class Stage2Corpus:
    """Both pools behind one indexable surface."""

    def __init__(
        self,
        audio_rows: list[AudioRow],
        code_index: ParquetCodeIndex,
        text_corpus,
        *,
        audio_root: str | Path,
    ):
        self.audio_rows = audio_rows
        self.code_index = code_index
        self.text_corpus = text_corpus
        self.audio_root = audio_root

    def audio_sample(self, index: int) -> dict:
        row = self.audio_rows[index % len(self.audio_rows)]
        codes, globals_ = self.code_index.get(row.parquet, row.row_index)
        sample = {
            "audio": row.audio_uri,
            "source_transcript": row.source_transcript,
            "target_text": row.target_text,
            "source_lang": row.source_lang,
            "target_lang": row.target_lang,
            "codes": codes,
            "globals": globals_,
        }
        if self.code_index.with_source:
            sample["source_codes"] = self.code_index.get_source(
                row.parquet, row.row_index)
        if getattr(self.code_index, "with_glm", False):
            sample["source_glm"] = self.code_index.get_glm(row.parquet, row.row_index)
        return sample

    def text_sample(self, index: int) -> dict:
        row = self.text_corpus[index % len(self.text_corpus)]
        return {
            "prompt_ids": row["prompt_ids"],
            "reply_ids": row["reply_ids"],
            "codes": row["codes"],
            "globals": row["globals"],
        }

    def waveform(self, uri: str):
        return read_audio(parse_wds_uri(uri), self.audio_root)
