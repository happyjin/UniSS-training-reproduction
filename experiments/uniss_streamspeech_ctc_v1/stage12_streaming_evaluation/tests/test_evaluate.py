import unittest

from experiments.uniss_streamspeech_ctc_v1.stage12_streaming_evaluation.evaluate import row


class ImportTest(unittest.TestCase):
    def test_row_callable(self):
        self.assertTrue(callable(row))


if __name__ == "__main__":
    unittest.main()
