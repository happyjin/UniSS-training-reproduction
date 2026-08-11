from __future__ import annotations

import sys
import unittest
from pathlib import Path

from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
MEGATRON_ROOT = REPO_ROOT / "third_party" / "Megatron-LM"
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))

from megatron.core.datasets.utils import Split  # noqa: E402
from experiments.uniss_phase3_dense_aligned_streaming_pilot15_v1.training.pretrain_dense_aligned_megatron import (  # noqa: E402
    JointValidationDataset,
)


class _Toy(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        return {"sample_kind": "replay", "index": index}


class MegatronValidationSplitTest(unittest.TestCase):
    def test_joint_validation_uses_megatron_split_enum(self) -> None:
        dataset = JointValidationDataset([_Toy()])
        self.assertIs(dataset.split, Split.valid)


if __name__ == "__main__":
    unittest.main()
