from __future__ import annotations

from torch.utils.data import Dataset

from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize12_microblock.pretrain_generalize12 import (
    SynchronizedValidationDataset,
)


class _ToyDataset(Dataset):
    def __init__(self, count: int) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        if not 0 <= index < self.count:
            raise IndexError(index)
        return {"sample_kind": "trajectory", "index": index}


def test_two_pack_canary_is_balanced_across_eight_ranks() -> None:
    dataset = SynchronizedValidationDataset(
        [_ToyDataset(2)], data_parallel_size=8
    )
    assert dataset.unpadded_length == 2
    assert len(dataset) == 8
    assert [dataset[index]["index"] for index in range(len(dataset))] == [
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
    ]


def test_formal_validation_only_pads_to_next_complete_dp_batch() -> None:
    dataset = SynchronizedValidationDataset(
        [_ToyDataset(607)], data_parallel_size=8
    )
    assert dataset.unpadded_length == 607
    assert len(dataset) == 608
    assert dataset[606]["index"] == 606
    assert dataset[607]["index"] == 0


def test_aligned_validation_needs_no_padding() -> None:
    dataset = SynchronizedValidationDataset(
        [_ToyDataset(16)], data_parallel_size=8
    )
    assert len(dataset) == 16
    assert dataset[-1]["index"] == 15
