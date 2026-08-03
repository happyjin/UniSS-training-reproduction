"""Phase3 endpoint metadata joined to the Stage03 audio dataset."""

from __future__ import annotations

import json

import torch

from audio_data import EndpointCTCAudioDataset, collate_audio


class B2BridgeAudioDataset(EndpointCTCAudioDataset):
    def __getitem__(self, index: int) -> dict[str, object]:
        value = super().__getitem__(index)
        target = self._target_row(index)
        source_index = int(target["source_manifest_index"])
        with self.source_manifest.open("rb") as handle:
            handle.seek(int(self.source_offsets[source_index]))
            source = json.loads(handle.readline())
        value["phase3_record"] = {
            "id": source["id"],
            "tgt_lang": source["tgt_lang"],
            "translation": source["translation"],
            "bicodec_global": source["bicodec_global"],
            "target_bicodec": source["target_bicodec"],
        }
        return value


def collate_bridge(batch: list[dict[str, object]]) -> dict[str, object]:
    value = collate_audio(batch)
    value["phase3_records"] = [row["phase3_record"] for row in batch]
    return value

