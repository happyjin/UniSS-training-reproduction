import sys
import unittest
from pathlib import Path


STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))

from ctc_utils import ctc_minimum_frames, deterministic_split, normalize_text


class CTCUtilsTest(unittest.TestCase):
    def test_normalization_is_language_aware(self) -> None:
        self.assertEqual(normalize_text("  Hello   WORLD  ", "eng"), "hello world")
        self.assertEqual(normalize_text(" 你 好 \n 世 界 ", "cmn"), "你好世界")

    def test_ctc_minimum_frames_includes_repeat_blanks(self) -> None:
        self.assertEqual(ctc_minimum_frames([]), 0)
        self.assertEqual(ctc_minimum_frames([1, 2, 3]), 3)
        self.assertEqual(ctc_minimum_frames([1, 1, 2, 2, 2]), 8)

    def test_split_is_deterministic(self) -> None:
        self.assertEqual(
            deterministic_split("sample-42"), deterministic_split("sample-42")
        )
        self.assertEqual(deterministic_split("sample-42", 0), "train")


if __name__ == "__main__":
    unittest.main()
