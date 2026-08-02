from __future__ import annotations

import unittest
from pathlib import Path


class StageBV2PipelineTest(unittest.TestCase):
    def test_pipeline_waits_for_idle_gpus_and_preserves_stage_order(self) -> None:
        root = Path(__file__).resolve().parents[2]
        source = (
            root / "scripts/simul_uniss_subsecond_v2/run_stage_b_v2_repair_pipeline.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("wait_for_idle_gpus", source)
        self.assertIn("nvidia-smi --query-compute-apps=pid", source)
        self.assertNotIn("kill -9", source)
        self.assertLess(source.index("run_sidecar clone"), source.index("clone pretraining"))
        self.assertLess(source.index("clone pretraining"), source.index("prefix-80 fine-tuning"))
        self.assertIn("stage_b_v2_prefix80_phase3_sensitivity.json", source)


if __name__ == "__main__":
    unittest.main()
