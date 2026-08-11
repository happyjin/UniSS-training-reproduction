from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from web_demo.true_subsecond_pilot15_streaming_v1.causal_frontend import (
    BoundedCausalWhisperVQFrontend,
    SAMPLE_RATE,
)


class FakeEncoder:
    def encode(self, audio):
        output = []
        for waveform, _ in audio:
            waveform = waveform.reshape(-1).numpy()
            count = (len(waveform) + 1279) // 1280
            values = []
            for index in range(count):
                segment = waveform[index * 1280 : (index + 1) * 1280]
                values.append(int(round(float(segment.mean()) * 1000)) % 16384)
            output.append(SimpleNamespace(tokens=torch.tensor(values)))
        return output


class CausalFrontendTest(unittest.TestCase):
    def test_only_elapsed_right_context_is_committed(self) -> None:
        frontend = BoundedCausalWhisperVQFrontend(
            FakeEncoder(), right_context_ms=80, window_ms=640
        )
        chunk = np.ones(SAMPLE_RATE * 320 // 1000, dtype=np.float32) * 0.1
        first = frontend.push(chunk)
        self.assertEqual(len(first.new_tokens), 2)
        second = frontend.push(chunk)
        self.assertEqual(len(second.new_tokens), 4)
        final = frontend.push(chunk, is_final=True)
        self.assertEqual(final.committed_tokens, 12)

    def test_waveform_state_is_bounded_for_five_minutes(self) -> None:
        frontend = BoundedCausalWhisperVQFrontend(
            FakeEncoder(), right_context_ms=80, window_ms=640
        )
        chunk = np.zeros(SAMPLE_RATE * 320 // 1000, dtype=np.float32)
        for _ in range(5 * 60 * 1000 // 320):
            frontend.push(chunk)
        self.assertLessEqual(frontend.maximum_buffer_samples, SAMPLE_RATE * 640 // 1000)
        self.assertEqual(frontend.committed_revision_violations, 0)


if __name__ == "__main__":
    unittest.main()
