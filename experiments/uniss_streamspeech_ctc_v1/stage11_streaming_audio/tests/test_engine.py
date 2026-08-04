import unittest
from types import SimpleNamespace

from experiments.uniss_streamspeech_ctc_v1.stage11_streaming_audio.engine import (
    Stage11Session,
)


class RejectionTest(unittest.TestCase):
    def session(self):
        value = Stage11Session.__new__(Stage11Session)
        value.engine = SimpleNamespace(
            config=SimpleNamespace(semantic_max_run=16, semantic_unique_ratio_min=0.1)
        )
        return value

    def test_rejects_invalid_structure(self):
        write = SimpleNamespace(
            structurally_valid=False,
            semantic_values=[1, 2],
            semantic_max_identical_run=1,
            semantic_unique_ratio=1.0,
        )
        self.assertEqual(self.session()._rejection(write), "invalid_structure")

    def test_accepts_diverse_semantic(self):
        write = SimpleNamespace(
            structurally_valid=True,
            semantic_values=list(range(20)),
            semantic_max_identical_run=1,
            semantic_unique_ratio=1.0,
        )
        self.assertIsNone(self.session()._rejection(write))


if __name__ == "__main__":
    unittest.main()
