from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.uniss_phase3_whisper_streamspeech_joint_v6.evaluation.summarize_fixed_chunk_eval import (
    CHUNKS,
    MODELS,
    collect,
    render_markdown,
)


METRICS = {
    "bicodec_ctc": 10.0,
    "ar_s2tt": 5.0,
    "asr_ctc": 20.0,
    "nar_s2tt_ctc": 19.0,
    "ctc/unit_infeasible": 0.01,
    "bridge/commitment_mse": 0.02,
    "bridge/teacher_glm_agreement": 0.17,
}


def _validation_line(metrics: dict[str, float]) -> str:
    body = " | ".join(f"{name} value: {value:.6E}" for name, value in metrics.items())
    return f"validation loss at iteration 500 on validation set | {body} |\n"


class FixedChunkEvaluationTest(unittest.TestCase):
    def test_collect_and_render_complete_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model in MODELS:
                for chunk in CHUNKS:
                    values = dict(METRICS)
                    if model == "stage_b":
                        values["asr_ctc"] -= 2.0
                        values["nar_s2tt_ctc"] -= 1.0
                    (root / f"{model}_{chunk}.log").write_text(
                        "ignored\n" + _validation_line(values)
                    )
            rows = collect(root)
            self.assertEqual(len(rows), 10)
            report = render_markdown(
                rows,
                stage_a_checkpoint=root / "stage_a",
                stage_b_checkpoint=root / "stage_b",
            )
            self.assertIn("Stage B 在 5/5 个 chunk 上改善", report)
            self.assertIn("没有通过 teacher agreement 改善门", report)
            self.assertIn("offline", report)

    def test_matrix_scripts_use_skip_train_and_refuse_overwrite(self) -> None:
        root = Path(__file__).resolve().parents[1]
        evaluator = (root / "evaluation/run_fixed_chunk_eval_8gpu.sh").read_text()
        matrix = (root / "evaluation/run_fixed_chunk_matrix.sh").read_text()
        self.assertIn("--skip-train", evaluator)
        self.assertIn("--no-load-optim", evaluator)
        self.assertIn('--lr "${BASE_LR}"', evaluator)
        self.assertIn('--joint-chunks "${CHUNK}"', evaluator)
        self.assertIn('refuse_existing "${OUTPUT_LOG}"', evaluator)
        self.assertIn('refuse_existing "${LOG_ROOT}" "${REPORT_ROOT}"', matrix)


if __name__ == "__main__":
    unittest.main()
