#!/usr/bin/env python3
"""CPU-only contract tests for periodic AR validation selection/metrics."""
from __future__ import annotations

import math
import unittest

from custom.ar_eval_callback import select_ar_eval_indices, summarize_ar_records


class FakeDataset:
    def __init__(self):
        self.rows = []
        for direction in (('en', 'zh'), ('zh', 'en')):
            for frames in (50, 100, 220):
                for ordinal in range(8):
                    self.rows.append({
                        'source_lang': direction[0],
                        'target_lang': direction[1],
                        'target_codec_codes': [[ordinal] * 16 for _ in range(frames)],
                    })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class AREvalCallbackTest(unittest.TestCase):
    def test_selection_is_fixed_and_stratified(self):
        dataset = FakeDataset()
        first = select_ar_eval_indices(dataset, 12, seed=17, candidate_count=len(dataset))
        second = select_ar_eval_indices(dataset, 12, seed=17, candidate_count=len(dataset))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        strata = {
            (dataset[index]['source_lang'], len(dataset[index]['target_codec_codes']))
            for index in first
        }
        self.assertEqual(len(strata), 6)

    def test_summary_keeps_eos_length_boundary_and_text_metrics(self):
        records = [
            {
                'target_codec_frames': 100,
                'generated_frames': 105,
                'eos_triggered': True,
                'text_exact_match': True,
                'gt_boundary': {
                    'top1_is_eos': True,
                    'eos_probability': 0.8,
                    'eos_rank': 1,
                },
            },
            {
                'target_codec_frames': 100,
                'generated_frames': 384,
                'eos_triggered': False,
                'text_exact_match': False,
                'gt_boundary': {
                    'top1_is_eos': False,
                    'eos_probability': 0.001,
                    'eos_rank': 20,
                },
            },
            {'val_index': 3, 'error': 'synthetic'},
        ]
        summary = summarize_ar_records(records, pathological_frames=350)
        self.assertEqual(summary['ar_eval/sample_count'], 2.0)
        self.assertEqual(summary['ar_eval/error_count'], 1.0)
        self.assertEqual(summary['ar_eval/eos_trigger_rate'], 0.5)
        self.assertEqual(summary['ar_eval/timely_eos_rate'], 0.5)
        self.assertEqual(summary['ar_eval/pathological_long_rate'], 0.5)
        self.assertEqual(summary['ar_eval/text_exact_match'], 0.5)
        self.assertEqual(summary['ar_eval/gt_boundary_eos_top1_rate'], 0.5)
        self.assertTrue(math.isclose(
            summary['ar_eval/gt_boundary_eos_probability_median'], 0.4005))


if __name__ == '__main__':
    unittest.main()
