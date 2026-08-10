import unittest

from web_demo.uniss_phase3_prefix_streaming_v3_longform_v1.app_gradio import (
    window_table_html,
)
from web_demo.uniss_phase3_prefix_streaming_v3_longform_v1.engine import (
    LongFormWindowRecord,
)


class AppContractTest(unittest.TestCase):
    def test_window_table_escapes_translation(self):
        record = LongFormWindowRecord(
            index=0,
            plan_index=0,
            depth=0,
            source_start_seconds=0.0,
            source_end_seconds=25.0,
            status="completed",
            boundary_rms=0.01,
            chunk_ms=480,
            first_audio_global_ms=4160.0,
            rtf=0.8,
            translation="<script>alert(1)</script>",
        )
        output = window_table_html([record])
        self.assertIn("&lt;script&gt;", output)
        self.assertNotIn("<script>", output)


if __name__ == "__main__":
    unittest.main()
