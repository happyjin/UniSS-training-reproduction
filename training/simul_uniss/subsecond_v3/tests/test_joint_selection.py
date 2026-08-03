from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from training.simul_uniss.jsonl_index import load_index
from training.simul_uniss.subsecond_v3.build_balanced_phase3_eval_manifest import (
    build as build_eval_manifest,
)
from training.simul_uniss.subsecond_v3.select_joint_checkpoint import select
from training.simul_uniss.subsecond_v3.tests.test_stage_b_v3_data import (
    TMP_ROOT,
    write_jsonl,
)


class StageBV3JointSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        TMP_ROOT.mkdir(parents=True, exist_ok=True)

    def test_materializes_interleaved_balanced_eval_manifest(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_ROOT) as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            write_jsonl(
                source,
                [
                    {"id": "e0", "src_lang": "eng", "tgt_lang": "cmn"},
                    {"id": "z0", "src_lang": "cmn", "tgt_lang": "eng"},
                    {"id": "e1", "src_lang": "eng", "tgt_lang": "cmn"},
                    {"id": "z1", "src_lang": "cmn", "tgt_lang": "eng"},
                ],
            )
            selection = root / "selection.jsonl"
            write_jsonl(
                selection,
                [
                    {"direction": "eng->cmn", "source_manifest_index": 0},
                    {"direction": "cmn->eng", "source_manifest_index": 1},
                    {"direction": "eng->cmn", "source_manifest_index": 2},
                    {"direction": "cmn->eng", "source_manifest_index": 3},
                ],
            )
            output = root / "eval.jsonl"
            result = build_eval_manifest(
                argparse.Namespace(
                    selection_manifest=str(selection),
                    source_manifest=str(source),
                    output=str(output),
                    per_direction=2,
                )
            )
            self.assertEqual(result["directions"], {"eng->cmn": 2, "cmn->eng": 2})
            offsets = load_index(output)
            assert offsets is not None
            rows = []
            with output.open("rb") as handle:
                for offset in offsets:
                    handle.seek(offset)
                    rows.append(json.loads(handle.readline()))
            self.assertEqual([row["id"] for row in rows], ["e0", "z0", "e1", "z1"])

    def test_joint_score_exports_bleu_balanced_winner(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_ROOT) as directory:
            root = Path(directory)
            checkpoints = root / "checkpoints"
            reports = root / "reports"
            checkpoints.mkdir()
            reports.mkdir()
            candidate_rows = []
            specifications = (
                ("step_000500", 0.40, 20.0),
                ("step_001000", 0.30, 40.0),
            )
            for stem, agreement, bleu in specifications:
                checkpoint = checkpoints / f"{stem}.pt"
                checkpoint.write_bytes(stem.encode())
                candidate_rows.append(
                    {
                        "score": agreement,
                        "checkpoint": str(checkpoint.resolve()),
                        "metrics": {"selection_score": agreement},
                    }
                )
                stream = f"candidate_{stem}"
                (reports / f"{stem}.json").write_text(
                    json.dumps(
                        {
                            "student_checkpoint": str(checkpoint.resolve()),
                            "student_stream_name": stream,
                            "text_bleu": {
                                "groups": {
                                    f"{stream}:eng->cmn": {"score": bleu},
                                    f"{stream}:cmn->eng": {"score": bleu},
                                }
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            candidates = checkpoints / "CANDIDATES.json"
            candidates.write_text(json.dumps({"candidates": candidate_rows}), encoding="utf-8")
            result = select(
                argparse.Namespace(
                    candidates=str(candidates),
                    phase3_result_dir=str(reports),
                    output_dir=str(checkpoints),
                    agreement_weight=0.5,
                    bleu_weight=0.5,
                    allow_replace=False,
                )
            )
            self.assertTrue(str(result["selected_checkpoint"]).endswith("step_001000.pt"))
            self.assertEqual((checkpoints / "best.pt").read_bytes(), b"step_001000")


if __name__ == "__main__":
    unittest.main()
