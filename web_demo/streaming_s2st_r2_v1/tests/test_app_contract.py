import json
import tempfile
import unittest
from pathlib import Path

from web_demo.streaming_s2st_r2_v1.app_gradio import (
    build_demo,
    event_timeline_html,
    parse_args,
    write_access_files,
)
from web_demo.streaming_s2st_r2_v1.config import StreamingDemoConfig
from web_demo.streaming_s2st_r2_v1.engine import StreamingDemoEngine
from web_demo.streaming_s2st_r2_v1.session_manager import SessionRegistry


class GradioAppContractTest(unittest.TestCase):
    def test_public_defaults_do_not_accept_credentials(self):
        args = parse_args([])
        self.assertEqual(args.port, 7862)
        self.assertFalse(hasattr(args, "auth_user"))
        self.assertFalse(hasattr(args, "auth_password"))

    def test_access_metadata_is_public_no_login(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            access = root / "access.json"
            public = root / "url.txt"
            write_access_files(
                public_url_file=public,
                access_info_file=access,
                local_url="http://127.0.0.1:7862",
                public_url="https://example.gradio.live",
                config=StreamingDemoConfig(),
            )
            value = json.loads(access.read_text(encoding="utf-8"))
            self.assertEqual(value["auth_mode"], "public_no_login")
            self.assertIsNone(value["username"])
            self.assertIsNone(value["password"])
            self.assertEqual(public.read_text(encoding="utf-8").strip(), value["public_url"])

    def test_timeline_escapes_generated_text(self):
        value = event_timeline_html(
            [{"event_index": 0, "source_end_ms": 640, "action": "write", "generated_text": "<x>"}]
        )
        self.assertIn("&lt;x&gt;", value)
        self.assertNotIn("<x>", value)

    def test_upload_and_microphone_expose_stereo_players(self):
        config = StreamingDemoConfig()
        demo = build_demo(
            config,
            StreamingDemoEngine(config),
            SessionRegistry(config.output_root, config.microphone_max_audio_seconds),
        )
        labels = {
            component.get("props", {}).get("label")
            for component in demo.get_config_file()["components"]
        }
        self.assertIn("同步双声道播放（左=源语言，右=翻译语言）", labels)
        self.assertIn("完成后的双声道延迟对比（左=源语言，右=翻译语言）", labels)
        demo.close(verbose=False)


if __name__ == "__main__":
    unittest.main()
