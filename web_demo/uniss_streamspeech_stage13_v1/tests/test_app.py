import subprocess
import unittest
from pathlib import Path

from web_demo.uniss_streamspeech_stage13_v1.app_gradio import timeline_html
from web_demo.uniss_streamspeech_stage13_v1.config import Stage13Config


class AppContractTest(unittest.TestCase):
    def test_timeline_escapes_text(self):
        value = timeline_html(
            [{"index": 1, "source_end_ms": 160, "policy_action": "WRITE", "qwen_text_delta": "<x>", "audio_samples": 0}]
        )
        self.assertIn("&lt;x&gt;", value)
        self.assertNotIn("<x>", value)

    def test_fixed_speaker_has_32_tokens(self):
        self.assertEqual(len(Stage13Config().fixed_speaker_tokens()), 32)

    def test_recovered_media_wrappers_are_available(self):
        root = Path(__file__).resolve().parents[3]
        media_bin = root / "web_demo" / "streaming_s2st_r2_v1" / "bin"
        for name in ("ffmpeg", "ffprobe"):
            tool = media_bin / name
            self.assertTrue(tool.is_file(), tool)
            completed = subprocess.run(
                [str(tool), "-version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
