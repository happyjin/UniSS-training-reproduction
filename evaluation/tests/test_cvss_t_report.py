from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.cvss_t import report


class CvssReportTest(unittest.TestCase):
    def make_run(self, root: Path, direction: str, value: float) -> Path:
        run = root / f"cvss_t_phase3_full_{direction.replace('->', '_to_')}"
        (run / "metrics").mkdir(parents=True)
        metrics = {
            "speech_bleu": ("score", value),
            "text_bleu": ("score", value + 1),
            "autopcp": ("mean", 2.5),
            "slc": None,
            "utmos": ("mean", 3.5),
        }
        for name, spec in metrics.items():
            groups = {}
            for mode in ("quality", "performance"):
                if name == "slc":
                    values = {"sample_count": 1, "slc_0_2": 0.9, "slc_0_4": 1.0}
                else:
                    field, metric_value = spec  # type: ignore[misc]
                    values = {"sample_count": 1, field: metric_value}
                groups[f"{mode}:{direction}"] = values
            (run / f"metrics/{name}.json").write_text(json.dumps({"groups": groups}), encoding="utf-8")
        return run

    def test_complete_report_compares_matching_uniss_modes_and_discloses_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = {
                path.name: report.collect_run(path)
                for path in (
                    self.make_run(root, "eng->cmn", 30.0),
                    self.make_run(root, "cmn->eng", 24.0),
                )
            }
            records = report.metric_records(runs)
            status = report.completeness(records, expected_pairs=1)
            paper = json.loads(report.PAPER_REFERENCE.read_text(encoding="utf-8"))
            deltas = report.matching_paper_deltas(records, paper)
            markdown = report.markdown_report(
                runs,
                records,
                status,
                deltas,
                paper,
                {"train_shard_count": 198, "train_row_count": 100, "id_match_count": 0, "matched_train_record_count": 3},
                expected_pairs=1,
            )
        self.assertTrue(status["formal_complete"])
        self.assertEqual(status["observed_metric_cells"], 24)
        quality_en_zh = [
            row
            for row in deltas
            if row["mode"] == "quality" and row["direction"] == "eng->cmn" and row["metric"] == "speech_bleu"
        ][0]
        self.assertAlmostEqual(quality_en_zh["delta_local_minus_paper"], -2.2)
        self.assertIn("归一化文本命中的训练记录：3", markdown)
        self.assertIn("主观 MOS", markdown)
        self.assertIn("EN→ZH / ZH→EN", markdown)


if __name__ == "__main__":
    unittest.main()
