from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from training import constants_uniss as c
from training.pretrain_simul_uniss_megatron import build_simul_datasets, validate_simul_args
from training.simul_uniss import PACKED_SCHEMA_VERSION


def packed_item(seq_length: int) -> dict[str, object]:
    return {
        "schema_version": PACKED_SCHEMA_VERSION,
        "tokens": [1] * seq_length,
        "labels": [2] * seq_length,
        "loss_mask": [0.0, 4.0, *([1.0] * (seq_length - 2))],
        "position_ids": list(range(seq_length)),
        "sample_boundaries": [[0, seq_length]],
        "tasks": ["simul_s2st"],
        "source_ids": ["x"],
    }


class PretrainSimulUniSSTests(unittest.TestCase):
    def test_validate(self) -> None:
        args = SimpleNamespace(
            sft=True,
            create_attention_mask_in_dataloader=False,
            context_parallel_size=1,
            eval_iters=0,
            full_validation=False,
            simul_packed_valid=None,
            simul_schema_version=PACKED_SCHEMA_VERSION,
            vocab_size=c.VOCAB_SIZE,
            add_bias_linear=False,
            add_qkv_bias=True,
        )
        validate_simul_args(args)

    def test_dataset_builder_preserves_float_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packed.jsonl"
            path.write_text(json.dumps(packed_item(8)) + "\n", encoding="utf-8")
            args = SimpleNamespace(
                seq_length=8,
                simul_packed_train=str(path),
                simul_packed_valid=None,
                simul_packed_test=None,
            )
            train, valid, test = build_simul_datasets(args)
            self.assertIsNone(valid)
            self.assertIsNone(test)
            self.assertEqual(train[0]["loss_mask"].tolist()[:2], [0.0, 4.0])


if __name__ == "__main__":
    unittest.main()
