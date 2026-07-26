import tempfile
import unittest
from pathlib import Path

from evaluation.io_utils import iter_jsonl, write_jsonl
from evaluation.sharding import merge_jsonl_by_key, select_shard


class EvaluationShardingTest(unittest.TestCase):
    def test_index_modulo_shards_are_disjoint_and_complete(self):
        rows = [{"id": str(index), "mode": "quality"} for index in range(11)]
        shards = [list(select_shard(rows, num_shards=4, shard_index=index)) for index in range(4)]
        keys = [[row["id"] for row in shard] for shard in shards]
        self.assertEqual(keys[0], ["0", "4", "8"])
        self.assertEqual(keys[3], ["3", "7"])
        self.assertEqual(sorted(value for shard in keys for value in shard), sorted(row["id"] for row in rows))

    def test_merge_is_idempotent_and_keeps_first_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.jsonl"
            part = root / "part.jsonl"
            write_jsonl(base, [{"id": "a", "mode": "quality", "value": 1}])
            write_jsonl(
                part,
                [
                    {"id": "a", "mode": "quality", "value": 2},
                    {"id": "b", "mode": "performance", "value": 3},
                ],
            )
            report = merge_jsonl_by_key([base, part], base)
            rows = list(iter_jsonl(base))
        self.assertEqual(report, {"written": 2, "duplicates_skipped": 1})
        self.assertEqual(rows[0]["value"], 1)
        self.assertEqual(rows[1]["id"], "b")


if __name__ == "__main__":
    unittest.main()
