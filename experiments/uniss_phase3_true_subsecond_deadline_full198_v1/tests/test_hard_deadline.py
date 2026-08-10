from __future__ import annotations

import unittest

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.inference import (
    DeadlineScheduler,
)


class HardDeadlineTest(unittest.TestCase):
    def test_natural_write_is_distinct_from_forced_write(self) -> None:
        scheduler = DeadlineScheduler()
        natural = scheduler.decide(
            elapsed_speech_ms=480,
            write_probability=0.9,
            supported_tokens=2,
            speech_active=True,
        )
        forced = scheduler.decide(
            elapsed_speech_ms=800,
            write_probability=0.1,
            supported_tokens=0,
            speech_active=True,
        )
        self.assertTrue(natural.natural_write)
        self.assertFalse(natural.deadline_forced)
        self.assertFalse(forced.natural_write)
        self.assertTrue(forced.deadline_forced)

    def test_silence_is_not_forced_to_write(self) -> None:
        decision = DeadlineScheduler().decide(
            elapsed_speech_ms=1000,
            write_probability=0.9,
            supported_tokens=0,
            speech_active=False,
        )
        self.assertEqual(decision.action, "READ")


if __name__ == "__main__":
    unittest.main()
