import json
import tempfile
import unittest
from pathlib import Path

from evaluation import aggregate_report


class AggregateReportTest(unittest.TestCase):
    def make_run(self, root: Path, stage: str, value: float) -> Path:
        run = root / f"qwen_{stage}_unist_dev_full_test"
        (run / "metrics").mkdir(parents=True)
        (run / "vllm").mkdir()
        (run / "vllm/run_config.json").write_text(
            json.dumps({"manifest": "/data/manifests/unist_dev_all.jsonl"}), encoding="utf-8"
        )
        metric = {
            "groups": {
                "quality:cmn->eng": {"sample_count": 10, "score": value},
            }
        }
        (run / "metrics/text_bleu.json").write_text(json.dumps(metric), encoding="utf-8")
        return run

    def test_unist_report_compares_phases_but_blocks_paper_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase2 = self.make_run(root, "phase2", 20.0)
            phase3 = self.make_run(root, "phase3", 21.5)
            runs = {path.name: aggregate_report.collect_run(path) for path in (phase2, phase3)}
            records = [
                record
                for name, run in runs.items()
                for record in aggregate_report.metric_records(name, run)
            ]
            comparisons = aggregate_report.build_comparisons(records)
            comparability = aggregate_report.paper_comparability(records)
            paper = json.loads(aggregate_report.PAPER_REFERENCE.read_text(encoding="utf-8"))
            markdown = aggregate_report.markdown_report(runs, comparisons, paper, comparability)

        self.assertEqual(len(comparisons), 1)
        self.assertAlmostEqual(comparisons[0]["delta_phase3_minus_phase2"], 1.5)
        self.assertFalse(comparability["direct_numeric_comparison_allowed"])
        self.assertIn("不计算跨数据集差值", markdown)
        self.assertIn("UniSS (Q)", markdown)
        self.assertIn("32.2000 | 24.2800", markdown)

    def test_cvss_full_result_enables_guarded_paper_comparison(self):
        row = {
            "dataset": "cvss_t",
            "split": "test",
            "scope": "full",
            "direction": "cmn->eng",
        }
        result = aggregate_report.paper_comparability([row])
        self.assertTrue(result["direct_numeric_comparison_allowed"])
        self.assertEqual(result["available_cvss_t_directions"], ["cmn->eng"])

    def test_matching_cvss_metric_builds_selected_paper_deltas(self):
        paper = json.loads(aggregate_report.PAPER_REFERENCE.read_text(encoding="utf-8"))
        row = {
            "dataset": "cvss_t",
            "split": "test",
            "scope": "full",
            "stage": "phase3",
            "mode": "quality",
            "direction": "eng->cmn",
            "metric": "speech_bleu",
            "value": 30.0,
        }
        comparisons = aggregate_report.build_paper_comparisons([row], paper)
        uniss_q = [item for item in comparisons if item["paper_model"] == "UniSS (Q)"]
        self.assertEqual(len(uniss_q), 1)
        self.assertAlmostEqual(uniss_q[0]["delta_local_minus_paper"], -2.2)
        self.assertFalse(any(item["paper_model"] == "UniSS (P)" for item in comparisons))


if __name__ == "__main__":
    unittest.main()
