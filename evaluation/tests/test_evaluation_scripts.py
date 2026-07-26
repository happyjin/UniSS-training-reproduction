import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPO_ROOT / "experiments/evaluation/uniss_full198_phase2_phase3"


class EvaluationScriptsTest(unittest.TestCase):
    def test_python_entrypoints_export_repository_pythonpath(self):
        for name in ("run_hf_manifest.sh", "run_vllm_eval.sh", "run_objective_metrics.sh"):
            source = (SCRIPT_ROOT / name).read_text(encoding="utf-8")
            self.assertIn('export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"', source, name)

    def test_autopcp_uses_local_metric_encoder_cache(self):
        source = (REPO_ROOT / "evaluation/autopcp_metrics.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("local_files_only=True"), 2)

    def test_vllm_wrapper_uses_legacy_engine_for_vocab_mask(self):
        source = (SCRIPT_ROOT / "run_vllm_eval.sh").read_text(encoding="utf-8")
        self.assertIn('export VLLM_USE_V1="${VLLM_USE_V1:-0}"', source)


if __name__ == "__main__":
    unittest.main()
