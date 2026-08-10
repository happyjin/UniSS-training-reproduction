from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.scripts import (
    build_phase3_fingerprint as fingerprint,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.scripts.build_phase3_fingerprint import (
    DEFAULT_COLUMNS,
    DEFAULT_ROWS,
    SCHEMA,
    build,
)


class Phase3FingerprintTest(unittest.TestCase):
    def test_build_reads_only_frozen_embedding_coordinates(self) -> None:
        class FakeSlice:
            def get_shape(self):
                return [180_480, 896]

            def __getitem__(self, key):
                rows, full_columns = key
                self.assert_full = full_columns
                values = torch.zeros(len(rows), 896, dtype=torch.bfloat16)
                for row_index in range(len(rows)):
                    for column_index, column in enumerate(DEFAULT_COLUMNS):
                        values[row_index, column] = row_index * 10 + column_index
                return values

        class FakeHandle:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def get_slice(self, name):
                if name != "model.embed_tokens.weight":
                    raise KeyError(name)
                return FakeSlice()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.safetensors"
            output = root / "fingerprint.json"
            source.write_bytes(b"fake")
            with patch.object(fingerprint, "safe_open", return_value=FakeHandle()):
                payload = build(source, output)
            self.assertEqual(payload["schema_version"], SCHEMA)
            self.assertTrue(output.is_file())
            self.assertEqual(payload["values"][2][3], 23.0)


if __name__ == "__main__":
    unittest.main()
