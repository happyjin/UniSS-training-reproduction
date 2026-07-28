from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPO_ROOT / "experiments/evaluation/cvss_t_zh_en_phase3_v1"


class CvssScriptsTest(unittest.TestCase):
    def test_inference_uses_paper_sampling_parameters(self) -> None:
        source = (SCRIPT_ROOT / "run_vllm_eval.sh").read_text(encoding="utf-8")
        for fragment in (
            "--temperature 0.7",
            "--top-p 0.8",
            "--top-k -1",
            "--repetition-penalty 1.1",
            "--mode quality performance",
        ):
            self.assertIn(fragment, source)

    def test_objective_runner_has_all_table1_metrics_and_integrity_gate(self) -> None:
        source = (SCRIPT_ROOT / "run_objective_metrics.sh").read_text(encoding="utf-8")
        for fragment in (
            "evaluation.cvss_t.validate_results",
            "evaluation.asr_transcribe",
            "metrics/speech_bleu.json",
            "evaluation.utmos_metrics",
            "evaluation.autopcp_metrics",
            "evaluation.slc_metrics",
            "metrics/text_bleu.json",
        ):
            self.assertIn(fragment, source)

    def test_cvss_scripts_export_repository_pythonpath(self) -> None:
        for name in ("run_vllm_eval.sh", "run_objective_metrics.sh", "build_report.sh"):
            source = (SCRIPT_ROOT / name).read_text(encoding="utf-8")
            self.assertIn('export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"', source, name)

    def test_full_runner_keeps_directions_separate_and_reports_both(self) -> None:
        source = (SCRIPT_ROOT / "run_full_evaluation.sh").read_text(encoding="utf-8")
        self.assertIn("cvss_t_phase3_full_cmn_to_eng", source)
        self.assertIn("cvss_t_phase3_full_eng_to_cmn", source)
        self.assertIn('"cmn->eng"', source)
        self.assertIn('"eng->cmn"', source)
        self.assertIn("build_report.sh", source)


if __name__ == "__main__":
    unittest.main()
