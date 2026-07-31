from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from training.simul_uniss.subsecond_v2.assemble_stage_a import assemble


class FormalStageAAssemblyTest(unittest.TestCase):
    def test_only_formal_pass_records_enter_accepted_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a45_root = root / "a45"
            a68_root = root / "a68"
            for index in range(2):
                left = a45_root / str(index)
                right = a68_root / str(index)
                left.mkdir(parents=True)
                right.mkdir(parents=True)
                a45 = left / "a45_manifest.jsonl"
                a68 = right / "formal_manifest.jsonl"
                a45.write_text(json.dumps({"id": index}) + "\n", encoding="utf-8")
                a68.write_text(
                    json.dumps({"id": index, "formal_a68_pass": index == 0}) + "\n",
                    encoding="utf-8",
                )
                (left / "STAGE_A_A45_COMPLETE.json").write_text(
                    json.dumps({"status": "complete", "output_manifest": str(a45)}),
                    encoding="utf-8",
                )
                (right / "STAGE_A_A68_COMPLETE.json").write_text(
                    json.dumps({"status": "complete", "output_manifest": str(a68)}),
                    encoding="utf-8",
                )
            value = assemble(
                argparse.Namespace(
                    a45_root=str(a45_root),
                    a68_root=str(a68_root),
                    output_dir=str(root / "output"),
                    expected_parts=2,
                    expected_records=2,
                )
            )
            self.assertEqual(value["a45"]["records"], 2)
            self.assertEqual(value["formal_all"]["records"], 2)
            self.assertEqual(value["formal_accepted"]["records"], 1)
            self.assertEqual(value["formal_acceptance_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
