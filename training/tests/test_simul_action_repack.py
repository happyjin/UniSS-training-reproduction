from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from training import constants_uniss as c
from training.simul_uniss import SAMPLE_SCHEMA_VERSION
from training.simul_uniss.jsonl_index import load_index
from training.simul_uniss.repack_action_only import (
    assemble_action,
    generate_schedule,
    pack_action,
    verify_assembly,
    verify_part,
)


def sample(name: str, length: int) -> dict[str, object]:
    middle = [c.TOKEN_ENG] * max(0, length - 3)
    ids = [c.TOKEN_START_CONTENT, *middle, c.TOKEN_WAIT_READ, c.TOKEN_EOS]
    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "id": name,
        "task": "simul_s2st",
        "input_ids": ids,
        "token_weights": [1.0] * len(ids),
    }


class ActionRepackTests(unittest.TestCase):
    def test_pack_assemble_and_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            parts = root / "parts"
            for index in range(2):
                part = parts / f"train-{index:05d}"
                source = root / f"samples-{index}.jsonl"
                source.write_text(
                    "".join(json.dumps(sample(f"{index}-{item}", 5 + item)) + "\n" for item in range(3)),
                    encoding="utf-8",
                )
                output = part / "packed_action.jsonl"
                marker = part / "PACK_ACTION_COMPLETE.json"
                packed = pack_action(
                    argparse.Namespace(input=str(source), output=str(output), marker=str(marker), seq_length=16)
                )
                self.assertEqual(packed["input_records"], 3)
                self.assertEqual(packed["represented_samples"], 3)
                self.assertEqual(packed["dropped_overlong"], 0)
                verify_part(marker)

            output = root / "packed_action_train.jsonl"
            marker = root / "ACTION_DATA_COMPLETE.json"
            assembled = assemble_action(
                argparse.Namespace(
                    parts_root=str(parts),
                    shard_start=0,
                    shard_count=2,
                    seq_length=16,
                    output=str(output),
                    marker=str(marker),
                )
            )
            self.assertEqual(assembled["represented_samples"], 6)
            self.assertEqual(len(load_index(output) or []), assembled["packed_records"])
            verify_assembly(marker)

            schedule = root / "training_schedule.env"
            result = generate_schedule(
                argparse.Namespace(
                    assembly_marker=str(marker),
                    output=str(schedule),
                    global_batch_size=2,
                    epochs=1.0,
                    warmup_fraction=0.05,
                )
            )
            self.assertGreaterEqual(result["train_iters"], 1)
            self.assertIn("ACTION_SEQ_LENGTH=\"16\"", schedule.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
