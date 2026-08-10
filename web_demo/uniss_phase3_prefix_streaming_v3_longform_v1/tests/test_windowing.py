import unittest

import numpy as np

from web_demo.uniss_phase3_prefix_streaming_v3_longform_v1.windowing import (
    place_target_without_overlap,
    plan_bounded_windows,
    render_target_timeline,
    stereo_waveform,
)


class WindowingTest(unittest.TestCase):
    def test_five_minute_plan_is_complete_and_bounded(self):
        sample_rate = 100
        waveform = np.ones(300 * sample_rate, dtype=np.float32) * 0.1
        spans = plan_bounded_windows(waveform, sample_rate)

        self.assertEqual(spans[0].start_sample, 0)
        self.assertEqual(spans[-1].end_sample, len(waveform))
        self.assertTrue(
            all(a.end_sample == b.start_sample for a, b in zip(spans, spans[1:]))
        )
        self.assertTrue(
            all(18.0 <= span.samples / sample_rate <= 30.0 for span in spans)
        )
        self.assertTrue(10 <= len(spans) <= 16)

    def test_boundary_search_prefers_nearby_silence(self):
        sample_rate = 100
        waveform = np.ones(65 * sample_rate, dtype=np.float32)
        waveform[26 * sample_rate : 27 * sample_rate] = 0.0
        spans = plan_bounded_windows(waveform, sample_rate)

        boundary_seconds = spans[0].end_sample / sample_rate
        self.assertGreaterEqual(boundary_seconds, 26.0)
        self.assertLessEqual(boundary_seconds, 27.0)

    def test_short_recording_is_one_window(self):
        spans = plan_bounded_windows(np.ones(700, dtype=np.float32), 100)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].samples, 700)

    def test_target_schedule_and_stereo_do_not_overlap(self):
        placements = []
        first_start, first_end = place_target_without_overlap(
            placements,
            np.ones(4, dtype=np.float32),
            available_sample=3,
            cursor=0,
        )
        second_start, second_end = place_target_without_overlap(
            placements,
            np.ones(2, dtype=np.float32) * 0.5,
            available_sample=4,
            cursor=first_end,
        )
        self.assertEqual((first_start, first_end), (3, 7))
        self.assertEqual((second_start, second_end), (7, 9))

        timeline = render_target_timeline(placements, minimum_samples=5)
        stereo = stereo_waveform(np.arange(6, dtype=np.float32), timeline)
        self.assertEqual(timeline.tolist(), [0, 0, 0, 1, 1, 1, 1, 0.5, 0.5])
        self.assertEqual(stereo.shape, (9, 2))
        self.assertTrue(np.array_equal(stereo[:6, 0], np.arange(6, dtype=np.float32)))
        self.assertTrue(np.array_equal(stereo[:, 1], timeline))


if __name__ == "__main__":
    unittest.main()
