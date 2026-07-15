import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from training import mix_sample_jsonl as mix


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class MixSampleJsonlTest(unittest.TestCase):
    def test_weighted_round_robin_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.jsonl"
            b = Path(tmp) / "b.jsonl"
            write_rows(a, [{"id": "a1"}, {"id": "a2"}, {"id": "a3"}])
            write_rows(b, [{"id": "b1"}, {"id": "b2"}])

            groups = [
                mix.MixGroup("unist", 2, (a,)),
                mix.MixGroup("phase1", 1, (b,)),
            ]
            rows = list(mix.mix_groups(groups))
            self.assertEqual([row["id"] for row in rows], ["a1", "a2", "b1", "a3", "b2"])
            self.assertEqual([row["mix_group"] for row in rows], ["unist", "unist", "phase1", "unist", "phase1"])

    def test_parse_group_and_multiple_paths(self):
        group = mix.parse_group_spec("phase1=1:a.jsonl,b.jsonl")
        self.assertEqual(group.name, "phase1")
        self.assertEqual(group.weight, 1)
        self.assertEqual([str(path) for path in group.paths], ["a.jsonl", "b.jsonl"])
        with self.assertRaises(ValueError):
            mix.parse_group_spec("bad")
        with self.assertRaises(ValueError):
            mix.parse_group_spec("x=0:a.jsonl")

    def test_write_counts_and_max_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.jsonl"
            b = Path(tmp) / "b.jsonl"
            output = Path(tmp) / "mixed.jsonl"
            write_rows(a, [{"id": "a1", "task": "quality"}, {"id": "a2", "task": "quality"}])
            write_rows(b, [{"id": "b1", "task": "asr"}])

            groups = [mix.MixGroup("unist", 2, (a,)), mix.MixGroup("phase1", 1, (b,))]
            counts = mix.write_jsonl(mix.mix_groups(groups, max_records=2), output)
            self.assertEqual(counts["total"], 2)
            self.assertEqual(counts["unist"], 2)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["id"] for row in rows], ["a1", "a2"])

    def test_cli_help(self):
        result = subprocess.run(
            [sys.executable, "training/mix_sample_jsonl.py", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--group", result.stdout)


if __name__ == "__main__":
    unittest.main()
