"""Load the prefix-to-prefix pools, cutting each row's audio at its own point.

Why this exists rather than reusing the base dataset directly
-------------------------------------------------------------
``packed_task_to_runtime_item`` accepts these packed rows on everything except
one line, ``family not in TASK_FAMILIES``, which a test in this experiment
pins down.  So this module borrows it for the whole token-side decode and adds
the one thing it cannot do: load a *prefix* of the audio.

The base reader always loads the whole file, and
``StageAObjective._inject_causal_glm`` then raises unless the frontend's token
count for that waveform equals ``glm_lengths``.  A prefix-to-prefix ASR sample
deliberately binds fewer GLM positions than the utterance has, so the audio
has to be cut at the sample's ``source_pcm_end`` for the two to agree.  That
cut is safe because the frontend is block-causal: measured on 201 event
boundaries, a prefix cut at ``source_pcm_end`` reproduces the full run's
tokens bit for bit, so nothing the model sees comes from audio the session has
not heard.

Every cut is re-checked here against the closed form before it reaches a GPU,
so a bad pool fails in the dataloader with the sample id in hand rather than
inside the objective's tensor bookkeeping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.runtime_dataset import (  # noqa: E501
    _default_audio_loader,
    collate_e2e_family,
    packed_task_to_runtime_item,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.training.task_samples_p2st import (
    FAMILY_P2ST_ASR,
    FAMILY_P2ST_MT,
    FAMILY_P2ST_TTS,
    P2ST_FAMILIES,
    causal_glm_token_count,
)
from training.simul_uniss.jsonl_index import load_index

AudioLoader = Callable[[Path], tuple[torch.Tensor, int]]

# The base reader's whitelist is keyed on shape, not meaning: it only needs to
# know whether a row is acoustic or text.  These proxies say which of the two
# each p2st family is, and the real family is restored before the item is
# returned so nothing downstream sees the substitution.
PROXY_FAMILY = {
    FAMILY_P2ST_ASR: "streaming_asr_event",
    FAMILY_P2ST_MT: "incremental_mt_event",
    FAMILY_P2ST_TTS: "incremental_mt_event",
}


def _refuse_audio(path: Path) -> tuple[torch.Tensor, int]:
    raise AssertionError(
        "the base reader must not load audio for a p2st row; this module cuts "
        "it instead"
    )


def p2st_packed_task_to_runtime_item(
    value: Mapping[str, object],
    *,
    seq_length: int,
    load_audio: bool = True,
    audio_loader: AudioLoader | None = None,
) -> dict[str, object]:
    family = str(value.get("family"))
    if family not in P2ST_FAMILIES:
        raise ValueError(f"unknown p2st task family {family!r}")
    proxy = dict(value)
    proxy["family"] = PROXY_FAMILY[family]
    item = packed_task_to_runtime_item(
        proxy,
        seq_length=int(seq_length),
        load_audio=False,
        audio_loader=_refuse_audio,
    )
    item["family"] = family
    if item["teacher_bindings"]:
        raise ValueError("a p2st row carries a teacher binding; the pool is pure CE")
    item["teacher_posteriors"] = []

    acoustic_rows = item["acoustic_rows"]
    if not isinstance(acoustic_rows, list):
        raise TypeError("p2st acoustic rows are malformed")
    if not load_audio or not acoustic_rows:
        return item

    loader = audio_loader or _default_audio_loader
    cache: dict[str, torch.Tensor] = {}
    for row in acoustic_rows:
        cut = int(row.get("source_pcm_end", 0))
        glm_length = int(row["source_glm_length"])
        if cut <= 0:
            raise ValueError(
                f"p2st acoustic row {row.get('sample_id')} carries no audio cut"
            )
        path = str(row["source_audio"])
        waveform = cache.get(path)
        if waveform is None:
            loaded, sample_rate = loader(Path(path))
            loaded = torch.as_tensor(loaded, dtype=torch.float32)
            if loaded.ndim == 2 and loaded.shape[0] == 1:
                loaded = loaded[0]
            if loaded.ndim != 1 or int(sample_rate) != 16_000:
                raise ValueError("p2st source audio must be mono 16 kHz")
            if not torch.isfinite(loaded).all() or loaded.numel() <= 0:
                raise ValueError("p2st source audio is empty or contains NaN/Inf")
            waveform = loaded
            cache[path] = waveform
        if cut > int(waveform.numel()):
            raise ValueError(
                f"p2st audio cut {cut} exceeds {waveform.numel()} samples in "
                f"{path}"
            )
        prefix = waveform[:cut].contiguous()
        # The check the trainer would otherwise make far downstream, made here
        # where the sample id is still in hand.
        produced = causal_glm_token_count(int(prefix.numel()))
        if produced != glm_length:
            raise ValueError(
                "p2st audio cut does not yield the promised GLM length for "
                f"{row.get('sample_id')}: cut {cut} samples yields {produced} "
                f"tokens, row promises {glm_length}"
            )
        row["waveform"] = prefix
        row["waveform_length"] = int(prefix.numel())
    return item


def collate_p2st_family(batch: list[dict[str, object]]) -> dict[str, object]:
    """Batch one p2st family with the established collator.

    ``collate_e2e_family`` carries the same whitelist as the reader and does
    all the waveform, GLM and boundary padding, so the family is proxied for
    the call and restored on the result.
    """
    if not batch:
        raise ValueError("cannot collate an empty p2st batch")
    families = {str(value.get("family")) for value in batch}
    if len(families) != 1:
        raise ValueError("one optimizer microbatch cannot mix p2st task families")
    family = next(iter(families))
    if family not in P2ST_FAMILIES:
        raise ValueError(f"unknown p2st task family {family!r}")
    proxied = []
    for value in batch:
        item = dict(value)
        item["family"] = PROXY_FAMILY[family]
        proxied.append(item)
    merged = collate_e2e_family(proxied)
    merged["family"] = family
    return merged


class P2STPackedFamilyDataset(Dataset[dict[str, object]]):
    """Fork-safe random access to one p2st family's packed file.

    Deliberately smaller than ``E2EPackedFamilyDataset``: there are no teacher
    readers to resolve and no build-report contract to honour, because this
    pool has no teacher bindings at all.  The file handle is opened per access
    for the same reason the base dataset does it -- a handle held across
    ``__getitem__`` does not survive a dataloader fork.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        family: str,
        seq_length: int = 18_000,
        load_audio: bool = True,
        audio_loader: AudioLoader | None = None,
    ) -> None:
        if family not in P2ST_FAMILIES:
            raise ValueError(f"unknown p2st task family {family!r}")
        self.path = Path(path).resolve()
        self.family = str(family)
        self.seq_length = int(seq_length)
        self.load_audio = bool(load_audio)
        self.audio_loader = audio_loader
        offsets = load_index(self.path)
        if offsets is None:
            raise ValueError(f"p2st packed file has no offset index: {self.path}")
        self.offsets: Sequence[int] = offsets

    @classmethod
    def from_pool_manifest(
        cls,
        manifest: str | Path,
        *,
        family: str,
        load_audio: bool = True,
        audio_loader: AudioLoader | None = None,
    ) -> "P2STPackedFamilyDataset":
        payload = json.loads(Path(manifest).read_text())
        families = payload.get("families")
        if not isinstance(families, dict) or family not in families:
            raise ValueError(f"pool manifest is missing family {family!r}")
        entry = families[family]
        if not isinstance(entry, dict):
            raise ValueError("pool manifest family entry is malformed")
        return cls(
            entry["path"],
            family=family,
            seq_length=int(payload["seq_length"]),
            load_audio=load_audio,
            audio_loader=audio_loader,
        )

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        with self.path.open("rb") as handle:
            handle.seek(int(self.offsets[index]))
            value = json.loads(handle.readline())
        if value.get("family") != self.family:
            raise ValueError("p2st packed record escaped its family dataset")
        return p2st_packed_task_to_runtime_item(
            value,
            seq_length=self.seq_length,
            load_audio=self.load_audio,
            audio_loader=self.audio_loader,
        )


__all__ = [
    "AudioLoader",
    "PROXY_FAMILY",
    "P2STPackedFamilyDataset",
    "collate_p2st_family",
    "p2st_packed_task_to_runtime_item",
]
