import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPAIR = ROOT / "step1_repair_balanced_v1"
sys.path.insert(0, str(REPAIR))

from build_direction_indices import collect_split


class DirectionIndexTest(unittest.TestCase):
    def test_collects_global_rows_across_manifest_parts(self) -> None:
        with tempfile.TemporaryDirectory(
            dir="/opt/dlami/nvme/jasonleeeli/tmp"
        ) as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            first.write_text(
                "\n".join(
                    json.dumps({"direction": value})
                    for value in ("eng->cmn", "cmn->eng")
                )
                + "\n",
                encoding="utf-8",
            )
            second.write_text(
                json.dumps({"direction": "eng->cmn"}) + "\n",
                encoding="utf-8",
            )
            index = {
                "parts": {
                    "train": [
                        {"manifest": str(first), "records": 2},
                        {"manifest": str(second), "records": 1},
                    ]
                }
            }
            values = collect_split(index, "train")
            self.assertEqual(values[0].tolist(), [0, 2])
            self.assertEqual(values[1].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
