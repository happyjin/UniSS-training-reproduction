#!/usr/bin/env python3
"""Checks the Megatron loader and the probe's scoring helpers.

Run directly: ``python experiments/simul_s2st_route_v1/step1_v6_bleu_recheck/tests/test_loader.py``
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.distributed.checkpoint as dcp  # noqa: E402
from torch import nn  # noqa: E402

from experiments.simul_s2st_route_v1.step1_v6_bleu_recheck.loader import (  # noqa: E402
    CHECKPOINT_PREFIX,
    backbone_drift,
    checkpoint_iteration,
    load_joint_checkpoint,
)


class Tiny(nn.Module):
    def __init__(self, fill: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=True)
        self.register_buffer("codebook", torch.full((2, 4), fill, dtype=torch.float32))
        with torch.no_grad():
            self.linear.weight.fill_(fill)
            self.linear.bias.fill_(fill)


def write_checkpoint(directory: Path, model: nn.Module, *, dtype: torch.dtype) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    state = {f"{CHECKPOINT_PREFIX}{name}": value.to(dtype) for name, value in model.state_dict().items()}
    state["optimizer.exp_avg"] = torch.zeros(3, dtype=dtype)
    dcp.save(state, checkpoint_id=directory)


class LoaderTest(unittest.TestCase):
    def test_iteration_parsing(self) -> None:
        self.assertEqual(checkpoint_iteration("/a/b/iter_0005000"), 5000)
        with self.assertRaises(ValueError):
            checkpoint_iteration("/a/b/latest")

    def test_roundtrip_through_bf16_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "iter_0000250"
            write_checkpoint(path, Tiny(fill=0.5), dtype=torch.bfloat16)
            model = Tiny(fill=0.0)
            report = load_joint_checkpoint(model, path)

            self.assertEqual(report.iteration, 250)
            self.assertEqual(report.loaded_tensors, 3)
            self.assertEqual(report.missing_in_checkpoint, [])
            self.assertEqual(report.unused_in_checkpoint, 0)
            # bfloat16 represents 0.5 exactly, so the upcast back to float32 is lossless.
            self.assertTrue(torch.allclose(model.linear.weight, torch.full((3, 4), 0.5)))
            self.assertTrue(torch.allclose(model.codebook, torch.full((2, 4), 0.5)))
            self.assertEqual(model.codebook.dtype, torch.float32)

    def test_reports_tensors_absent_from_the_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "iter_0000100"
            write_checkpoint(path, Tiny(fill=1.0), dtype=torch.float32)
            model = Tiny(fill=0.0)
            model.register_buffer("extra", torch.zeros(2))
            report = load_joint_checkpoint(model, path)
            self.assertEqual(report.missing_in_checkpoint, ["extra"])
            self.assertEqual(report.loaded_tensors, 3)

    def test_rejects_a_shape_mismatch_instead_of_silently_skipping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "iter_0000100"
            write_checkpoint(path, Tiny(fill=1.0), dtype=torch.float32)
            model = Tiny(fill=0.0)
            model.linear = nn.Linear(5, 3)
            with self.assertRaises(ValueError):
                load_joint_checkpoint(model, path)

    def test_missing_directory_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_joint_checkpoint(Tiny(), Path(directory) / "iter_0000100")

    def test_backbone_drift_detects_a_single_changed_tensor(self) -> None:
        left = Tiny(fill=0.5)
        right = Tiny(fill=0.5)
        self.assertEqual(backbone_drift(left, right)["changed_tensors"], 0)
        with torch.no_grad():
            left.linear.bias.add_(0.25)
        drift = backbone_drift(left, right)
        self.assertEqual(drift["changed_tensors"], 1)
        self.assertAlmostEqual(float(drift["max_abs_delta"]), 0.25, places=6)
        self.assertEqual(drift["max_abs_delta_tensor"], "linear.bias")


class ScoringTest(unittest.TestCase):
    def test_agreement_handles_length_drift(self) -> None:
        from experiments.simul_s2st_route_v1.step1_v6_bleu_recheck.evaluate import agreement

        self.assertEqual(agreement([], [1, 2])["position_agreement"], 0.0)
        exact = agreement([1, 2, 3], [1, 2, 3])
        self.assertEqual(exact["position_agreement"], 1.0)
        self.assertEqual(exact["length_ratio"], 1.0)
        short = agreement([1, 9], [1, 2, 3, 4])
        self.assertEqual(short["compared_positions"], 2)
        self.assertEqual(short["position_agreement"], 0.5)
        self.assertEqual(short["length_ratio"], 0.5)

    def test_selection_is_balanced_and_duration_filtered(self) -> None:
        from experiments.simul_s2st_route_v1.step1_v6_bleu_recheck.evaluate import read_records

        rows = []
        for index in range(40):
            direction = ("eng", "cmn") if index % 4 == 0 else ("cmn", "eng")
            rows.append(
                {
                    "id": f"sample-{index}",
                    "src_lang": direction[0],
                    "tgt_lang": direction[1],
                    "source_audio": f"/tmp/{index}.wav",
                    "source_duration_ms": 500 if index == 4 else 4000,
                    "source_glm": [index],
                    "bicodec_global": [0] * 32,
                    "target_bicodec": [1, 2, 3],
                    "translation": f"text {index}",
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "joint_valid.jsonl"
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            records = read_records(
                manifest, per_direction=3, max_audio_seconds=10.0, min_audio_seconds=2.0
            )
        self.assertEqual(len(records), 6)
        self.assertEqual(
            sorted({record.direction for record in records}), ["cmn->eng", "eng->cmn"]
        )
        self.assertEqual(len({record.sample_id for record in records}), 6)
        # sample-4 sits below the duration floor and must not be selected.
        self.assertNotIn("sample-4", {record.sample_id for record in records})

    def test_selection_refuses_an_impossible_budget(self) -> None:
        from experiments.simul_s2st_route_v1.step1_v6_bleu_recheck.evaluate import read_records

        row = {
            "id": "only",
            "src_lang": "eng",
            "tgt_lang": "cmn",
            "source_audio": "/tmp/only.wav",
            "source_duration_ms": 4000,
            "source_glm": [1],
            "bicodec_global": [0] * 32,
            "target_bicodec": [1],
            "translation": "text",
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "joint_valid.jsonl"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                read_records(
                    manifest, per_direction=2, max_audio_seconds=10.0, min_audio_seconds=2.0
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
