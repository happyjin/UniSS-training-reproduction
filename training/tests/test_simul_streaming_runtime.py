from __future__ import annotations

import unittest

import numpy as np

from uniss.streaming.bicodec_streamer import StreamingBiCodecDecoder
from uniss.streaming.controller import FrontendStep, StreamingController, WriteResult
from uniss.streaming.policy import PolicyDecision, PolicyGate
from uniss.streaming.stable_prefix import StablePrefixCommitter


def fake_decode(_speaker_tokens, semantic_tokens):
    values = np.asarray(semantic_tokens, dtype=np.float32)
    return np.repeat(values, 320)


class FakeModel:
    def __init__(self) -> None:
        self.source = []
        self.waits = 0

    def append_source(self, glm_tokens):
        self.source.extend(glm_tokens)

    def choose_action(self, eligible: bool, is_final: bool):
        return PolicyDecision.WRITE if eligible or is_final else PolicyDecision.WAIT

    def commit_wait(self):
        self.waits += 1

    def generate_write(self, is_final: bool):
        return WriteResult([1, 2], [3, 4, 5, 6, 7])


class StreamingRuntimeTests(unittest.TestCase):
    def test_stable_prefix_never_rolls_back(self) -> None:
        committer = StablePrefixCommitter(holdback_tokens=1)
        self.assertEqual(committer.update([1, 2, 3]), [])
        self.assertEqual(committer.update([1, 2, 4]), [1])
        self.assertEqual(committer.update([9, 9, 9]), [])
        self.assertEqual(committer.committed, [1])
        self.assertEqual(committer.revision_events, 1)

    def test_policy_forces_final_write(self) -> None:
        gate = PolicyGate(min_write_tokens=2)
        self.assertFalse(
            gate.eligible(source_count=1, target_supported_count=1, target_committed_count=0)
        )
        self.assertTrue(
            gate.eligible(
                source_count=1,
                target_supported_count=1,
                target_committed_count=0,
                is_final=True,
            )
        )

    def test_streaming_codec_returns_exact_final_length(self) -> None:
        codec = StreamingBiCodecDecoder(
            fake_decode,
            left_context_tokens=10,
            holdback_tokens=2,
            overlap_ms=40,
        )
        speaker = list(range(32))
        first = codec.push(list(range(10)), speaker_tokens=speaker)
        second = codec.push(list(range(10, 20)), is_final=True)
        self.assertEqual(len(first) + len(second), 20 * 320)
        self.assertEqual(codec.emitted_samples, 20 * 320)

    def test_controller_wait_then_write(self) -> None:
        model = FakeModel()
        codec = StreamingBiCodecDecoder(fake_decode, holdback_tokens=1, overlap_ms=20)
        controller = StreamingController(model=model, codec=codec)
        speaker = list(range(32))
        action1, audio1 = controller.process_step(
            FrontendStep([1, 2], source_count=1, target_supported_count=0),
            speaker_tokens=speaker,
        )
        action2, audio2 = controller.process_step(
            FrontendStep([1, 2, 3], source_count=2, target_supported_count=2),
            speaker_tokens=speaker,
            is_final=True,
        )
        self.assertEqual(action1, PolicyDecision.WAIT)
        self.assertEqual(action2, PolicyDecision.WRITE)
        self.assertEqual(len(audio1), 0)
        self.assertGreater(len(audio2), 0)
        self.assertEqual(controller.wait_count, 1)
        self.assertEqual(controller.write_count, 1)


if __name__ == "__main__":
    unittest.main()
