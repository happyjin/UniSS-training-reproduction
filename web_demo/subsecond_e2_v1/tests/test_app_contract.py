from __future__ import annotations

import unittest

from web_demo.subsecond_e2_v1.app_gradio import DEFAULT_CHECKPOINT, DEFAULT_TOKENIZER


class E2AppContractTest(unittest.TestCase):
    def test_assets_exist(self) -> None:
        self.assertTrue(DEFAULT_CHECKPOINT.is_file())
        self.assertTrue(DEFAULT_TOKENIZER.is_file())


if __name__ == "__main__":
    unittest.main()
