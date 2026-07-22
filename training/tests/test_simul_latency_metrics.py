from __future__ import annotations

import unittest

from training.simul_uniss.latency_metrics import schedule_latency_metrics


class LatencyMetricsTests(unittest.TestCase):
    def test_metrics_for_simple_schedule(self) -> None:
        schedule = {
            "source_glm_length": 4,
            "target_text_length": 4,
            "events": [
                {
                    "action": "wait",
                    "source_glm_end": 1,
                    "source_end_ms": 640,
                },
                {
                    "action": "write",
                    "source_glm_end": 2,
                    "source_end_ms": 1280,
                    "target_text_ids": [1, 2],
                },
                {
                    "action": "write",
                    "source_glm_end": 4,
                    "source_end_ms": 1920,
                    "target_text_ids": [3, 4],
                },
            ],
        }
        metrics = schedule_latency_metrics(schedule)
        self.assertEqual(metrics["first_write_ms"], 1280.0)
        self.assertEqual(metrics["num_wait"], 1.0)
        self.assertGreater(metrics["ap"], 0.0)
        self.assertGreaterEqual(metrics["atd_ms_proxy"], 0.0)


if __name__ == "__main__":
    unittest.main()
