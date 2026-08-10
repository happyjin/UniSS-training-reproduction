from __future__ import annotations

import unittest

from web_demo.uniss_phase3_prefix_streaming_v3_stereo_v1.app_gradio import timeline_html


class AppContractTest(unittest.TestCase):
    def test_timeline_escapes_generated_text(self) -> None:
        value = timeline_html(
            [
                {
                    "index": 0,
                    "source_end_ms": 480,
                    "action": "write",
                    "new_glm_tokens": 2,
                    "new_text_tokens": 1,
                    "semantic_tokens": 8,
                    "committed_text": "<script>",
                }
            ]
        )
        self.assertIn("&lt;script&gt;", value)
        self.assertNotIn("<script>", value)


if __name__ == "__main__":
    unittest.main()

