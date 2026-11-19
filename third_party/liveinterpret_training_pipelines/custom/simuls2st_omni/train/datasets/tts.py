"""TTS dataset over DualCodec-tokenized shards.

Reads this on-disk layout:

    <root>/manifest/<LANG>/<shard>.jsonl   one record per utterance
    <root>/codes/<LANG>/<shard>.npy        uint16 codes, referenced as
                                           "<LANG>/<shard>.npy:<start>:<end>"

Unpaired TTS needs no audio at train time (`TASK_TTS_UNPAIRED`'s user payload is
plain text), so this dataset touches only manifests and code arrays. Code files
are opened lazily with `mmap_mode="r"` and cached per worker, which keeps memory
flat regardless of corpus size.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from simuls2st_omni.train.datasets.subsample import subsample_uniform
from simuls2st_omni.train.prompt_formats import (
    TASK_TTS_UNPAIRED,
    build_prompt_bundle,
)

logger = logging.getLogger(__name__)

LANG_NAME = {"en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean"}


def _tts_duration_sec(record: dict) -> float:
    duration = record.get("duration")
    if duration is not None:
        return float(duration)
    _, start, end = str(record["codes"]).rsplit(":", 2)
    return (int(end) - int(start)) * 0.08


class TTSCodesDataset(Dataset):
    """Unpaired TTS samples: (target text, DualCodec semantic codes)."""

    def __init__(
        self,
        root: str | Path,
        langs: Optional[list[str]] = None,
        max_shards: int = 0,
        max_samples: int = 0,
        max_hours: float = 0.0,
        sample_seed: int = 0,
        min_codes: int = 1,
        max_codes: int = 1_000,
        min_dnsmos: float = 0.0,
    ):
        self.root = Path(root)
        self.records: list[dict] = []
        self._code_arrays: dict[str, np.ndarray] = {}

        manifest_root = self.root / "manifest"
        assert manifest_root.is_dir(), f"missing manifest dir: {manifest_root}"

        lang_dirs = (
            [manifest_root / lang for lang in langs]
            if langs
            else sorted(p for p in manifest_root.iterdir() if p.is_dir())
        )

        shard_count = 0
        for lang_dir in lang_dirs:
            if not lang_dir.is_dir():
                logger.warning("skip missing manifest dir %s", lang_dir)
                continue
            for shard in sorted(lang_dir.glob("*.jsonl")):
                if max_shards and shard_count >= max_shards:
                    break
                shard_count += 1
                with shard.open(encoding="utf-8") as f:
                    for line in f:
                        record = json.loads(line)
                        n_codes = self._code_len(record["codes"])
                        if n_codes < min_codes or n_codes > max_codes:
                            continue
                        dnsmos = record.get("dnsmos")
                        if dnsmos is not None and float(dnsmos) < min_dnsmos:
                            continue
                        if not (record.get("text") or "").strip():
                            continue
                        self.records.append(record)

        n_pool = len(self.records)
        self.records = subsample_uniform(
            self.records,
            max_samples=max_samples,
            max_hours=max_hours,
            seed=sample_seed,
            duration_sec_fn=_tts_duration_sec,
        )
        assert self.records, (
            f"no usable records under {manifest_root} "
            f"(langs={langs}, codes in [{min_codes}, {max_codes}], dnsmos >= {min_dnsmos})"
        )
        logger.info(
            "TTSCodesDataset: %d samples from pool %d / %d shard(s) (max_hours=%s)",
            len(self.records),
            n_pool,
            shard_count,
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
            # mmap so worker processes share page cache instead of copying.
            array = np.load(path, mmap_mode="r")
            self._code_arrays[rel] = array
        return np.asarray(array[int(start) : int(end)], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.records)

    def code_length(self, index: int) -> int:
        """Cheap length lookup for the batch sampler's bucketing."""
        return self._code_len(self.records[index]["codes"])

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        lang = LANG_NAME.get(str(record.get("language", "")).lower(), "English")
        text = record["text"].strip()

        # Unpaired TTS: the assistant repeats the user's text verbatim while the
        # talker emits the matching codes.
        bundle = build_prompt_bundle(
            task_type=TASK_TTS_UNPAIRED,
            text=text,
            target_text=text,
            tgt_lang=lang,
        )
        return {
            "id": record["id"],
            "task_type": TASK_TTS_UNPAIRED,
            "system_prompt": bundle.system_prompt,
            "user_payload": bundle.user_payload,
            "assistant_payload": bundle.assistant_payload,
            "codes": self._codes(record["codes"]),
        }
