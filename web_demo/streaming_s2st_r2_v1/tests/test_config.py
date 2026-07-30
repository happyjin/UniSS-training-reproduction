import unittest
from dataclasses import replace

from web_demo.streaming_s2st_r2_v1.config import StreamingDemoConfig


class StreamingDemoConfigTest(unittest.TestCase):
    def test_frozen_assets_and_model_labels(self):
        config = StreamingDemoConfig()
        config.validate()
        self.assertIn("R2 explicit-latency", config.model_label)
        fallback = replace(config, model_name="r3")
        fallback.validate()
        self.assertIn("R3 bilingual-adaptive", fallback.model_label)

    def test_invalid_browser_selectable_model_is_rejected(self):
        with self.assertRaises(ValueError):
            replace(StreamingDemoConfig(), model_name="arbitrary").validate()


if __name__ == "__main__":
    unittest.main()
