from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace

import torch

from training.simul_uniss.jsonl_index import load_index, write_index
from training.simul_uniss.subsecond_v3.build_balanced_selection import build as build_selection
from training.simul_uniss.subsecond_v3.build_mixed_manifest import build as build_mixed
from training.simul_uniss.subsecond_v3.prefix_hidden_teacher import (
    PrefixTeacherOutput,
    build_exact_prefix_hidden_targets,
)


TMP_ROOT = Path("/opt/dlami/nvme/jasonleeeli/tmp/stage_b_v3_tests")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    positions = array("Q")
    position = 0
    with path.open("wb") as handle:
        for row in rows:
            encoded = (json.dumps(row) + "\n").encode()
            positions.append(position)
            handle.write(encoded)
            position += len(encoded)
    write_index(path, positions)


class FakePrefixTeacher:
    def __init__(self) -> None:
        self.model = SimpleNamespace(
            codebook=SimpleNamespace(weight=torch.zeros(8, 3))
        )

    def encode(self, audio):
        outputs = []
        for index, _ in enumerate(audio):
            count = index + 3
            outputs.append(
                PrefixTeacherOutput(
                    tokens=torch.arange(1, count + 1),
                    pre_vq_hidden=torch.arange(count * 3, dtype=torch.float32).reshape(
                        count, 3
                    ),
                )
            )
        return outputs


class StageBV3DataTest(unittest.TestCase):
    def setUp(self) -> None:
        TMP_ROOT.mkdir(parents=True, exist_ok=True)

    def test_exact_prefix_targets_keep_token_hidden_alignment(self) -> None:
        waveform = torch.zeros(1, 10_240)
        tokens, stability, hidden = build_exact_prefix_hidden_targets(
            FakePrefixTeacher(),
            waveform,
            [80, 160, 240, 320, 400, 480, 560, 640],
            chunk_ms=160,
            lookahead_ms=80,
        )
        self.assertEqual(tokens.tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(tuple(hidden.shape), (6, 3))
        self.assertEqual(len(stability), len(tokens))
        self.assertTrue(torch.equal(hidden[0], torch.tensor([0.0, 1.0, 2.0])))

    def test_balanced_selection_and_mixed_manifest(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_ROOT) as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source_rows = [
                {"id": "e0", "src_lang": "eng", "tgt_lang": "cmn"},
                {"id": "z0", "src_lang": "cmn", "tgt_lang": "eng"},
                {"id": "e1", "src_lang": "eng", "tgt_lang": "cmn"},
                {"id": "z1", "src_lang": "cmn", "tgt_lang": "eng"},
            ]
            write_jsonl(source, source_rows)
            selection = root / "selection.jsonl"
            result = build_selection(
                argparse.Namespace(
                    source_manifest=str(source),
                    output=str(selection),
                    per_direction=2,
                    all_records=False,
                    seed=7,
                )
            )
            self.assertEqual(result["directions"], {"eng->cmn": 2, "cmn->eng": 2})
            selection_offsets = load_index(selection)
            assert selection_offsets is not None
            selected_rows = []
            with selection.open("rb") as handle:
                for offset in selection_offsets:
                    handle.seek(offset)
                    selected_rows.append(json.loads(handle.readline()))
            prefix = root / "prefix.jsonl"
            prefix_rows = [
                {
                    "source_manifest_index": row["source_manifest_index"],
                    "source_manifest_offset": row["source_manifest_offset"],
                    "shard_path": f"prefix-{index}.pt",
                    "target_start": 0,
                    "target_end": 1,
                    "reference_start": 0,
                    "reference_end": 1,
                }
                for index, row in enumerate(selected_rows)
            ]
            write_jsonl(prefix, prefix_rows)
            clone = root / "clone.jsonl"
            clone_rows = [
                {
                    "source_manifest_index": index,
                    "source_manifest_offset": 0,
                    "shard_path": f"clone-{index}.pt",
                    "target_start": 0,
                    "target_end": 1,
                    "reference_start": 0,
                    "reference_end": 1,
                }
                for index in range(4)
            ]
            write_jsonl(clone, clone_rows)
            mixed = root / "mixed.jsonl"
            mixed_result = build_mixed(
                argparse.Namespace(
                    selection_manifest=str(selection),
                    prefix_manifest=str(prefix),
                    clone_manifest=str(clone),
                    output=str(mixed),
                )
            )
            self.assertEqual(mixed_result["records"], 8)
            self.assertEqual(
                mixed_result["supervision"],
                {"exact_prefix80_hidden": 4, "streaming_clone_hidden": 4},
            )


if __name__ == "__main__":
    unittest.main()
