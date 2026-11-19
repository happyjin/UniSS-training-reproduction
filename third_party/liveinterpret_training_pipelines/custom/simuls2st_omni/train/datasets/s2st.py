"""Offline (whole-utterance) S2ST dataset for Stage 2.

Same DualCodec manifests as the Stage 3 streaming dataset, read differently:
Stage 2 trains the offline task, so each line is one utterance -- full source
audio in, full target text plus the whole code sequence out. The chunk
trajectory is ignored here; only Stage 3 consumes it.

The production pool is `data/s2st/dualcodec_rechunk_d` (si_s4_rechunk_D
sentence clips, 12.86kh). Source wavs stay under `si_s4/audio`; codes are
the manifest's slice into the shard npy.
"""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from simuls2st_omni.train.datasets.subsample import subsample_uniform
from simuls2st_omni.train.prompt_formats import TASK_S2S_TRANSLATE, build_audio_span, build_prompt_bundle

logger = logging.getLogger(__name__)

SAMPLE_RATE_16K = 16_000
AUDIO_SPAN_MARKER = "<|AUDIO_SPAN|>"
LANG_NAMES = {"en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean"}


class OfflineS2STDataset(Dataset):
    """Whole-utterance S2ST: source audio -> target text + semantic codes."""

    def __init__(
        self,
        root: str | Path,
        si_root: str | Path,
        directions: Optional[Iterable[str]] = None,
        max_samples: int = 0,
        max_hours: float = 0.0,
        sample_seed: int = 0,
        max_codes: int = 1500,
        min_codes: int = 4,
        max_source_sec: float = 30.0,
    ):
        self.root = Path(root)
        self.si_root = Path(si_root)
        self.max_source_sec = float(max_source_sec)
        self.records: list[dict] = []
        self._code_arrays: dict[str, np.ndarray] = {}
        self._pack_fds: dict[str, int] = {}

        manifest_root = self.root / "manifest"
        assert manifest_root.is_dir(), f"missing manifest dir: {manifest_root}"
        dir_names = (
            list(directions)
            if directions
            else sorted(p.name for p in manifest_root.iterdir() if p.is_dir())
        )

        dropped = 0
        for direction in dir_names:
            dir_path = manifest_root / direction
            if not dir_path.is_dir():
                logger.warning("skip missing manifest dir %s", dir_path)
                continue
            for shard in sorted(dir_path.glob("*.jsonl")):
                with shard.open(encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        n_codes = self._code_len(record["codes"])
                        if n_codes < min_codes or n_codes > max_codes:
                            dropped += 1
                            continue
                        if not (record.get("target_text") or "").strip():
                            dropped += 1
                            continue
                        # Keep only fields Stage 2 reads. The 12kh rechunk_D
                        # manifests also store a streaming trajectory that
                        # OfflineS2STDataset never uses.
                        self.records.append(
                            {
                                "id": record["id"],
                                "direction": record["direction"],
                                "source_audio": record["source_audio"],
                                "target_text": record["target_text"],
                                "codes": record["codes"],
                            }
                        )

        n_pool = len(self.records)
        self.records = subsample_uniform(
            self.records,
            max_samples=max_samples,
            max_hours=max_hours,
            seed=sample_seed,
            duration_sec_fn=lambda rec: self._code_len(rec["codes"]) * 0.08,
        )
        assert self.records, f"no usable records under {manifest_root}"
        logger.info(
            "OfflineS2STDataset: %d samples from pool %d (dropped %d, max_hours=%s)",
            len(self.records),
            n_pool,
            dropped,
            max_hours or "off",
        )

    @staticmethod
    def _code_len(ref: str) -> int:
        _, start, end = ref.rsplit(":", 2)
        return int(end) - int(start)

    def _codes(self, ref: str) -> np.ndarray:
        rel, start, end = ref.rsplit(":", 2)
        array = self._code_arrays.get(rel)
        if array is None:
            path = self.root / "codes" / rel
            assert path.exists(), f"missing code file: {path}"
            array = np.load(path, mmap_mode="r")
            self._code_arrays[rel] = array
        return np.asarray(array[int(start) : int(end)], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def pack_cost(n_codes: int) -> int:
        """Sampler cost for one S2ST row, in talker tokens.

        The collator builds `talker_len = prefix_len + 3 + n_codes`, and
        `prefix_len` is dominated by source-audio placeholders (~25/s). DualCodec
        is 12.5 codes/s, so two audio tokens per code plus a small prompt
        overhead. Packing on `n_codes` alone lets 256 short clips share a batch
        whose real tensors are `prefix + codes` and OOMs the text head.
        """
        return int(n_codes) + 2 * int(n_codes) + 80

    def code_length(self, index: int) -> int:
        """Talker-side cost for the batch sampler."""
        return self.pack_cost(self._code_len(self.records[index]["codes"]))

    token_length = code_length

    def _read_packed(self, rel: str, offset: int, length: int) -> bytes:
        fd = self._pack_fds.get(rel)
        if fd is None:
            path = self.si_root / rel
            assert path.exists(), f"missing source audio pack: {path}"
            fd = os.open(path, os.O_RDONLY)
            self._pack_fds[rel] = fd
        # pread keeps no file position, so a descriptor inherited across a
        # DataLoader fork cannot interleave reads between workers.
        data = os.pread(fd, length, offset)
        assert len(data) == length, f"short read from {rel} at {offset}"
        return data

    def _load_source(self, rel: str) -> np.ndarray:
        # Packed clips are "<pack>.bin:<offset>:<length>"; older manifests name one file per clip.
        pack, _, span = rel.partition(".bin:")
        if span:
            offset, length = span.split(":")
            raw = self._read_packed(pack + ".bin", int(offset), int(length))
            wav, sr = torchaudio.load(io.BytesIO(raw))
        else:
            path = self.si_root / rel
            assert path.exists(), f"missing source audio: {path}"
            wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE_16K:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE_16K)
        audio = wav.squeeze(0).numpy().astype(np.float32)
        limit = int(self.max_source_sec * SAMPLE_RATE_16K)
        # Trim rather than drop: the target text still covers the whole
        # utterance, and si_s4 clips rarely exceed the limit.
        return audio[:limit] if audio.shape[0] > limit else audio

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        src, _, tgt = record["direction"].partition("2")
        src_lang = LANG_NAMES.get(src, "English")
        tgt_lang = LANG_NAMES.get(tgt, "Chinese")

        bundle = build_prompt_bundle(
            task_type=TASK_S2S_TRANSLATE,
            target_text=record["target_text"].strip(),
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            # One patch token yields the full audio structure; the collator
            # expands the marker once it knows the clip's frame count.
            n_audio_tokens=1,
        )
        one_token_span = build_audio_span(1)
        assert one_token_span in bundle.user_payload, (
            f"{record['id']}: no audio span to replace"
        )
        return {
            "id": record["id"],
            "task_type": TASK_S2S_TRANSLATE,
            "system_prompt": bundle.system_prompt,
            "user_payload": bundle.user_payload.replace(
                one_token_span, AUDIO_SPAN_MARKER
            ),
            "assistant_payload": bundle.assistant_payload,
            "audio": self._load_source(record["source_audio"]),
            "codes": self._codes(record["codes"]),
        }
