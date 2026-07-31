from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from training.simul_uniss.subsecond_v2.prepare_stage_d import formal_schedule


class FormalStageDScheduleTest(unittest.TestCase):
    def test_formal_config_overrides_historical_low_utilization_defaults(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = root / "configs/experiments/simul_uniss_subsecond_v2/stage_d_formal_15shard_v1.env"
        command = (
            f"source {config}; "
            "printf '%s %s %s %s %s' \"$SEQ_LENGTH\" \"$SIMUL_NPROC_PER_NODE\" "
            "\"$SIMUL_MICRO_BATCH_SIZE\" \"$SIMUL_GLOBAL_BATCH_SIZE\" \"$SIMUL_QWEN_LR\""
        )
        output = subprocess.run(
            ["bash", "-c", command], check=True, capture_output=True, text=True
        ).stdout
        self.assertEqual(output, "18000 8 2 128 5e-6")

    def test_formal_schedule_preserves_source_and_target_and_ends_on_write(self) -> None:
        record = {
            "formal_a68_pass": True,
            "alignment_kind": "formal",
            "id": "x",
            "src_lang": "eng",
            "tgt_lang": "cmn",
            "transcription": "hello",
            "translation": "你好",
            "source_duration_ms": 800,
            "teacher_source_glm": [1, 2, 3, 4],
            "teacher_source_glm_end_ms": [160, 320, 480, 640],
            "target_bicodec": list(range(24)),
            "bicodec_global": list(range(32)),
            "micro_write_events": [
                {
                    "text": "你",
                    "semantic_start": 0,
                    "semantic_end": 12,
                    "support_end_ms": 300,
                    "earliest_safe_ms": 480,
                },
                {
                    "text": "好",
                    "semantic_start": 12,
                    "semantic_end": 24,
                    "support_end_ms": 600,
                    "earliest_safe_ms": 640,
                },
            ],
        }
        schedule = formal_schedule(record, lambda text: [ord(value) for value in text])
        source = [value for event in schedule["events"] for value in event["source_glm"]]
        target = [
            value
            for event in schedule["events"]
            if event["action"] == "write"
            for value in event["target_semantic"]
        ]
        self.assertEqual(source, [1, 2, 3, 4])
        self.assertEqual(target, list(range(24)))
        self.assertEqual(schedule["events"][-1]["action"], "write")
        self.assertTrue(schedule["events"][-1]["source_is_final"])


if __name__ == "__main__":
    unittest.main()
