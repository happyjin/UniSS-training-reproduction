"""Stage08 data view retaining the full offline Phase3 replay record."""

from __future__ import annotations

import json
from collections.abc import Mapping

from bridge_data import B2BridgeAudioDataset


REPLAY_FIELDS = (
    "id",
    "src_lang",
    "tgt_lang",
    "transcription",
    "translation",
    "source_glm",
    "bicodec_global",
    "target_bicodec",
)


def full_phase3_record(source: Mapping[str, object]) -> dict[str, object]:
    missing = [name for name in REPLAY_FIELDS if name not in source]
    if missing:
        raise KeyError(f"source record is missing offline replay fields: {missing}")
    record = {name: source[name] for name in REPLAY_FIELDS}
    for name in ("source_glm", "bicodec_global", "target_bicodec"):
        if not isinstance(record[name], list) or not record[name]:
            raise ValueError(f"offline replay field {name} must be a non-empty list")
    return record


class ReplayB2BridgeAudioDataset(B2BridgeAudioDataset):
    """B2 audio dataset with immutable source GLM tokens for replay."""

    def __getitem__(self, index: int) -> dict[str, object]:
        value = super().__getitem__(index)
        target = self._target_row(index)
        source_index = int(target["source_manifest_index"])
        with self.source_manifest.open("rb") as handle:
            handle.seek(int(self.source_offsets[source_index]))
            source = json.loads(handle.readline())
        if str(source["id"]) != str(value["id"]):
            raise ValueError(
                f"offline replay/audio mismatch: {source['id']} != {value['id']}"
            )
        value["phase3_record"] = full_phase3_record(source)
        return value
