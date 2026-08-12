from __future__ import annotations

import hashlib
import unittest

from experiments.uniss_phase3_event_rollout_joint_pilot15_v1.data.audit_fixed15 import (
    classify_timing_provenance,
    split_for_id,
)


class Fixed15DataAuditTest(unittest.TestCase):
    def test_split_rule_matches_formal_stage_a(self) -> None:
        for sample_id in ("sample-a", "sample-b", "NCSSD_R_EN_0000000000"):
            value = int(hashlib.sha256(sample_id.encode()).hexdigest()[:16], 16)
            expected = "valid" if value % 100 == 0 else "train"
            self.assertEqual(split_for_id(sample_id, 100), expected)

    def test_oracle_forced_alignment_is_not_called_exact_natural_timing(self) -> None:
        value = {
            "source_alignment_kind": "qwen3_forced_aligner_word_time_v1",
            "target_alignment_kind": "qwen3_forced_aligner_word_time_v1",
            "micro_write_events": [
                {"safe_label_kind": "oracle_bilingual_support_future_monotonic_v2"}
            ],
        }
        classification = classify_timing_provenance(value)
        self.assertIn("pseudo_timing", classification)
        self.assertNotIn("natural_exact", classification)


if __name__ == "__main__":
    unittest.main()
