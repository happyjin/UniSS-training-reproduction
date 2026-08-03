"""Direction-aware view of the isolated Stage-B-v3 mixed sidecar."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch

from training.simul_uniss.subsecond_v2.stage_b_v2_data import (
    StageBV2SidecarDataset,
    collate_stage_b_v2,
)


DIRECTION_TO_ID = {"eng->cmn": 0, "cmn->eng": 1}
SUPERVISION_TO_ID = {"exact_prefix80_hidden": 0, "streaming_clone_hidden": 1}


class StageBV3MixedDataset(StageBV2SidecarDataset):
    def __init__(self, sidecar_manifest: str | Path, *args, **kwargs) -> None:
        super().__init__(sidecar_manifest, *args, **kwargs)
        directions: list[int] = []
        supervision: list[int] = []
        with self.sidecar_path.open("rb") as handle:
            for offset in self.offsets:
                handle.seek(offset)
                row = json.loads(handle.readline())
                directions.append(DIRECTION_TO_ID[str(row["direction"])])
                supervision.append(SUPERVISION_TO_ID[str(row["supervision_mode"])])
        self.direction_ids = directions
        self.supervision_ids = supervision

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        value = super().__getitem__(index)
        value["direction_id"] = torch.tensor(self.direction_ids[index], dtype=torch.long)
        value["supervision_id"] = torch.tensor(
            self.supervision_ids[index], dtype=torch.long
        )
        return value


def collate_stage_b_v3(batch: list[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    value = collate_stage_b_v2(batch)
    value["direction_id"] = torch.stack([row["direction_id"] for row in batch])
    value["supervision_id"] = torch.stack([row["supervision_id"] for row in batch])
    return value
