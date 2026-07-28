import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from evaluation.io_utils import iter_jsonl, write_jsonl
from evaluation.merge_evaluation_shards import merge_shards
from evaluation.shard_manifest import shard_manifest
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

    def test_manifest_shards_are_balanced_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "manifest.jsonl"
            write_jsonl(source, ({"id": str(index)} for index in range(10)))
            report = shard_manifest(source, root / "parts", 4)
            ids = [
                [row["id"] for row in iter_jsonl(path)]
                for path in report["paths"]
            ]
        self.assertEqual(report["counts"], [3, 3, 2, 2])
        self.assertEqual(ids[0], ["0", "4", "8"])
        self.assertEqual(ids[3], ["3", "7"])

    def test_merge_evaluation_shards_builds_canonical_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            output = root / "output"
            shards = output / "data_parallel_shards"
            write_jsonl(manifest, [{"id": "a"}, {"id": "b"}])
            for index, sample_id in enumerate(("a", "b")):
                generation_rows = []
                result_rows = []
                for mode in ("quality", "performance"):
                    generation = {
                        "id": sample_id,
                        "mode": mode,
                        "semantic_token_count": 1,
                        "dummy_token_count": 0,
                        "generated_translation": "ok",
                    }
                    if sample_id == "b" and mode == "quality":
                        generation["generated_translation"] = None
                    generation_rows.append(generation)
                    result_rows.append(
                        {
                            **generation,
                            "audio_path": "/audio.wav",
                            "source_audio_path": "/source.wav",
                            "reference_audio_path": "/reference.wav",
                            "error": None,
                            "source_audio_error": None,
                            "reference_audio_error": None,
                        }
                    )
                part = shards / f"shard_{index:03d}"
                write_jsonl(part / "vllm" / "generation_results.jsonl", generation_rows)
                write_jsonl(part / "results.jsonl", result_rows)
            report = merge_shards(
                Namespace(
                    manifest=str(manifest),
                    output_root=str(output),
                    shard_root=str(shards),
                    num_shards=2,
                    modes=["quality", "performance"],
                )
            )
            merged = list(iter_jsonl(output / "results.jsonl"))
        self.assertEqual(report["expected"], 4)
        self.assertEqual(report["generation_summary"]["dummy_generated_tokens"], 0)
        self.assertEqual(report["generation_summary"]["missing_translation"], 1)
        self.assertEqual(report["decode_summary"]["decoded"], 4)
        self.assertEqual(len(merged), 4)


if __name__ == "__main__":
    unittest.main()
