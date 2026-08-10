from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.build_trajectory_cache import (
    _save_bundle,
    build_records_for_row,
    causal_bundle_reference,
    process_shard,
    stable_prefix_length,
    teacher_bundle_reference,
    trim_decoded_waveforms,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.build_cache_smoke_index import (
    select_smoke_rows,
)


def _summary(tokens: list[int], confidence: float = 0.95) -> dict[str, np.ndarray]:
    length = len(tokens)
    return {
        "indices": np.zeros((length, 2), dtype=np.int32),
        "probabilities": np.full((length, 2), 0.5, dtype=np.float16),
        "top1": np.asarray(tokens, dtype=np.int32),
        "confidence": np.full(length, confidence, dtype=np.float16),
    }


class TrajectoryCacheBuilderTest(unittest.TestCase):
    def test_decoded_waveforms_are_trimmed_on_time_axis(self) -> None:
        waveform = torch.arange(2 * 1 * 12, dtype=torch.float32).reshape(2, 1, 12)
        first, second = trim_decoded_waveforms(
            waveform, [2, 3], samples_per_token=4
        )
        self.assertEqual(first.shape, (8,))
        self.assertEqual(second.shape, (12,))
        torch.testing.assert_close(first, waveform[0].reshape(-1)[:8])
        torch.testing.assert_close(second, waveform[1].reshape(-1))

    def test_process_shard_keeps_batch_row_cache_indices_distinct(self) -> None:
        class Decoder:
            def decode(self, _globals, semantics):
                return [torch.zeros(1, len(values) * 320) for values in semantics]

        class Whisper:
            def encode(self, audio):
                return [SimpleNamespace(tokens=torch.arange(20)) for _ in audio]

        class Teacher:
            def encode_text(self, _text):
                return [10, 11, 12]

            def prompt(self, _row, _source):
                return [1, 2]

            def summarize(self, requests):
                return [_summary(target) for _prompt, target in requests]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            index = root / "index"
            output = root / "output"
            raw.mkdir()
            index.mkdir()
            rows = {
                "id": ["a", "b"],
                "transcription": ["one", "two"],
                "translation": ["yi", "er"],
                "source_glm": [list(range(20)), list(range(20))],
                "source_bicodec": [list(range(80)), list(range(80))],
                "target_bicodec": [list(range(32)), list(range(32))],
                "bicodec_global": [list(range(32)), list(range(32))],
                "src_lang": ["eng", "eng"],
                "tgt_lang": ["cmn", "cmn"],
            }
            pq.write_table(pa.Table.from_pydict(rows), raw / "train-00000.parquet")
            np.save(index / "train-00000.eng.npy", np.asarray([0, 1], dtype=np.int64))
            np.save(index / "train-00000.cmn.npy", np.asarray([], dtype=np.int64))
            args = Namespace(
                output_root=str(output),
                raw_unist_dir=str(raw),
                index_root=str(index),
                batch_size=2,
                rank=0,
                confidence_threshold=0.7,
                progress_interval=0,
            )
            process_shard(args, 0, Decoder(), Whisper(), Teacher())
            cache = output / "part-000" / "trajectory_cache.jsonl"
            records = [json.loads(line) for line in cache.read_text().splitlines()]
            self.assertEqual(
                [record["frontend_token_cache"].rsplit(":", 1)[-1] for record in records],
                ["0", "0", "1", "1"],
            )

    def test_smoke_selection_preserves_direction_membership(self) -> None:
        eng, cmn = select_smoke_rows(
            np.asarray([1, 4, 9], dtype=np.int64),
            np.asarray([2, 3, 8], dtype=np.int64),
            4,
        )
        self.assertEqual(eng.tolist(), [1, 4])
        self.assertEqual(cmn.tolist(), [2, 3])

    def test_stable_prefix_stops_at_first_unsafe_token(self) -> None:
        reference = [10, 11, 12]
        predictions = [reference, reference, reference, [10, 99, 12]]
        confidences = [[0.9, 0.9, 0.9] for _ in range(4)]
        length, mask = stable_prefix_length(
            reference, predictions, confidences, threshold=0.7
        )
        self.assertEqual(length, 1)
        self.assertEqual(mask, (True, False, True))

    def test_cache_namespaces_do_not_alias(self) -> None:
        path = Path("bundle.npz")
        self.assertEqual(causal_bundle_reference(path, 3), "bundle.npz::causal:3")
        self.assertEqual(teacher_bundle_reference(path, 3), "bundle.npz::teacher:3")

    def test_bundle_offsets_and_record_references_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle-000000.npz"
            summaries = [_summary([10, 11, 12]) for _ in range(16)]
            _save_bundle(bundle, summaries, ([1, 2], [3, 4, 5]))
            with np.load(bundle) as values:
                self.assertEqual(values["causal_tokens"].tolist(), [1, 2, 3, 4, 5])
                self.assertEqual(values["causal_token_offsets"].tolist(), [0, 2, 5])
                self.assertIn("request_15_top1", values.files)

            row = {
                "id": "sample",
                "src_lang": "eng",
                "tgt_lang": "cmn",
                "source_glm": list(range(20)),
                "source_bicodec": list(range(80)),
                "target_bicodec": list(range(32)),
                "bicodec_global": list(range(32)),
            }
            records = build_records_for_row(
                shard=0,
                row_index=4,
                row=row,
                causal_tokens=list(range(20)),
                translation_ids=[10, 11, 12],
                summaries=summaries[:8],
                cache_file=bundle,
                cache_row_index=1,
                request_offset=8,
                confidence_threshold=0.7,
            )
            self.assertTrue(
                all(record.frontend_token_cache.endswith("::causal:1") for record in records)
            )
            self.assertTrue(records[0].teacher_prefix_topk_path.endswith("::teacher:8"))
            self.assertTrue(records[1].teacher_full_topk_path.endswith("::teacher:15"))


if __name__ == "__main__":
    unittest.main()
