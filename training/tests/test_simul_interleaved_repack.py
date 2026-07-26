from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from training import constants_uniss as c
from training.simul_uniss import SAMPLE_SCHEMA_VERSION
from training.simul_uniss.jsonl_index import load_index
from training.simul_uniss.repack_interleaved import (
    assemble_interleaved,
    finalize,
    generate_schedule,
    pack_interleaved,
    verify_assembly,
    verify_part,
)


def sample(name: str, length: int, weight: float = 1.0) -> dict[str, object]:
    middle = [c.TOKEN_ENG] * max(0, length - 3)
    ids = [c.TOKEN_START_CONTENT, *middle, c.TOKEN_WRITE_GENERATE, c.TOKEN_EOS]
    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "id": name,
        "task": "simul_s2st",
        "input_ids": ids,
        "token_weights": [weight] * len(ids),
    }


class InterleavedRepackTests(unittest.TestCase):
    def test_pack_assemble_schedule_and_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            parts = root / "parts"
            for index in range(2):
                part = parts / f"train-{index:05d}"
                source = root / f"samples-{index}.jsonl"
                source.write_text(
                    "".join(
                        json.dumps(sample(f"{index}-{item}", 5 + item, 0.5 + item)) + "\n"
                        for item in range(3)
                    ),
                    encoding="utf-8",
                )
                output = part / "packed_interleaved.jsonl"
                marker = part / "PACK_INTERLEAVED_COMPLETE.json"
                packed = pack_interleaved(
                    argparse.Namespace(
                        input=str(source), output=str(output), marker=str(marker), seq_length=16
                    )
                )
                self.assertEqual(packed["input_records"], 3)
                self.assertEqual(packed["represented_samples"], 3)
                self.assertEqual(packed["dropped_overlong"], 0)
                first = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
                self.assertTrue(any(value not in (0.0, 1.0) for value in first["loss_mask"]))
                verify_part(marker)

            output = root / "packed_interleaved_train.jsonl"
            marker = root / "INTERLEAVED_DATA_COMPLETE.json"
            assembled = assemble_interleaved(
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
                    stage4_epochs=1.0,
                    stage6_epochs=0.25,
                    warmup_fraction=0.05,
                )
            )
            self.assertGreaterEqual(result["stage4_train_iters"], result["stage6_train_iters"])
            schedule_text = schedule.read_text(encoding="utf-8")
            self.assertIn('INTERLEAVED_SEQ_LENGTH="16"', schedule_text)
            self.assertIn("STAGE4_TRAIN_ITERS", schedule_text)
            self.assertIn("STAGE6_TRAIN_ITERS", schedule_text)

            validation_source = root / "validation.jsonl"
            validation_source.write_text(json.dumps(sample("valid", 6)) + "\n", encoding="utf-8")
            validation_output = root / "packed_interleaved_valid.jsonl"
            validation_marker = root / "PACK_INTERLEAVED_VALID_COMPLETE.json"
            pack_interleaved(
                argparse.Namespace(
                    input=str(validation_source),
                    output=str(validation_output),
                    marker=str(validation_marker),
                    seq_length=16,
                )
            )
            ready = root / "FULL_DATA_READY.json"
            finalized = finalize(
                argparse.Namespace(
                    assembly_marker=str(marker),
                    validation_marker=str(validation_marker),
                    schedule=str(schedule),
                    output=str(ready),
                )
            )
            self.assertEqual(finalized["seq_length"], 16)
            self.assertTrue(ready.is_file())

    def test_overlong_accounting_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            source = root / "samples.jsonl"
            source.write_text(
                json.dumps(sample("short", 5)) + "\n" + json.dumps(sample("long", 20)) + "\n",
                encoding="utf-8",
            )
            packed = pack_interleaved(
                argparse.Namespace(
                    input=str(source),
                    output=str(root / "packed.jsonl"),
                    marker=str(root / "PACK_INTERLEAVED_COMPLETE.json"),
                    seq_length=8,
                )
            )
            self.assertEqual(packed["input_records"], 2)
            self.assertEqual(packed["represented_samples"], 1)
            self.assertEqual(packed["dropped_overlong"], 1)


if __name__ == "__main__":
    unittest.main()
