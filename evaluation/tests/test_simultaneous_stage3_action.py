import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from evaluation.simultaneous_streaming.stage3_action_eval import (
    EvaluationRecord,
    build_batches,
    load_records,
)
from evaluation.simultaneous_streaming.stage3_aggregate import (
    aggregate_events,
    aggregate_samples,
)
from training import constants_uniss as c


class Stage3ActionEvaluationTest(unittest.TestCase):
    def test_load_records_validates_schedule_actions(self):
        sample = {
            "id": "x",
            "input_ids": [1, c.TOKEN_WAIT_READ, 2, c.TOKEN_WRITE_GENERATE, 3],
            "token_weights": [0.0, 4.0, 0.0, 4.0, 0.0],
        }
        schedule = {
            "id": "x",
            "src_lang": "cmn",
            "tgt_lang": "eng",
            "dataset_name": "unit",
            "events": [
                {"action": "wait", "chunk_index": 0, "source_end_ms": 640},
                {
                    "action": "write",
                    "chunk_index": 1,
                    "source_end_ms": 1280,
                    "source_is_final": True,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = root / "samples.jsonl"
            schedules = root / "schedules.jsonl"
            samples.write_text(json.dumps(sample) + "\n")
            schedules.write_text(json.dumps(schedule) + "\n")
            records = load_records(samples, schedules, rank=0, world_size=1)
        self.assertEqual(records[0].action_positions, [1, 3])
        self.assertEqual(records[0].action_labels, [c.TOKEN_WAIT_READ, c.TOKEN_WRITE_GENERATE])

    def test_build_batches_never_exceeds_padded_token_budget(self):
        records = [
            EvaluationRecord(i, str(i), [0] * length, [1], [c.TOKEN_WAIT_READ], [{}], "a", "b", "d")
            for i, length in enumerate([10, 11, 20, 21, 22])
        ]
        batches = list(build_batches(records, max_batch_tokens=44, max_batch_size=4))
        self.assertEqual(sum(len(batch) for batch in batches), len(records))
        for batch in batches:
            self.assertLessEqual(max(record.length for record in batch) * len(batch), 44)

    def test_aggregate_action_metrics(self):
        events = [
            {
                "reference_action": "wait",
                "binary_prediction": "wait",
                "global_prediction_action": "wait",
                "target_ce": 0.1,
                "binary_write_probability": 0.1,
            },
            {
                "reference_action": "wait",
                "binary_prediction": "write",
                "global_prediction_action": "other",
                "target_ce": 1.0,
                "binary_write_probability": 0.8,
            },
            {
                "reference_action": "write",
                "binary_prediction": "write",
                "global_prediction_action": "write",
                "target_ce": 0.2,
                "binary_write_probability": 0.9,
            },
        ]
        result = aggregate_events(events)
        self.assertAlmostEqual(result["binary_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["premature_write_given_wait"], 0.5)
        self.assertAlmostEqual(result["invalid_global_top1_rate"], 1 / 3)

    def test_aggregate_first_write_and_flush(self):
        samples = [
            {
                "reference_first_write_ms": 1280,
                "predicted_first_write_ms": 640,
                "first_write_delta_ms": -640,
                "predicted_write_count": 2,
                "reference_write_count": 1,
                "final_flush_success": True,
            },
            {
                "reference_first_write_ms": 1280,
                "predicted_first_write_ms": 1280,
                "first_write_delta_ms": 0,
                "predicted_write_count": 1,
                "reference_write_count": 1,
                "final_flush_success": False,
            },
        ]
        result = aggregate_samples(samples)
        self.assertEqual(result["samples"], 2)
        self.assertAlmostEqual(result["final_flush_success_rate"], 0.5)
        self.assertAlmostEqual(result["first_write_absolute_error_ms_mean"], 320)


if __name__ == "__main__":
    unittest.main()
