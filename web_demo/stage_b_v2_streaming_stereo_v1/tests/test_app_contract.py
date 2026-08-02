from __future__ import annotations

import unittest

from web_demo.streaming_s2st_r2_v1.session_manager import SessionRegistry
from web_demo.stage_b_v2_streaming_stereo_v1.app_gradio import build_demo, parse_args
from web_demo.stage_b_v2_streaming_stereo_v1.config import StudentV2StreamingConfig
from web_demo.stage_b_v2_streaming_stereo_v1.engine import StudentV2StreamingEngine


class StudentV2AppContractTest(unittest.TestCase):
    def test_public_defaults_and_frozen_assets(self) -> None:
        args = parse_args([])
        self.assertEqual(args.port, 7864)
        config = StudentV2StreamingConfig()
        config.validate()
        self.assertIn("Student", config.model_label)
        self.assertEqual(config.frontend_feed_ms, 160)
        self.assertEqual(config.chunk_ms, 640)

    def test_upload_and_microphone_expose_stereo_players(self) -> None:
        config = StudentV2StreamingConfig()
        demo = build_demo(
            config,
            StudentV2StreamingEngine(config),
            SessionRegistry(config.output_root, config.microphone_max_audio_seconds),
        )
        labels = {
            component.get("props", {}).get("label")
            for component in demo.get_config_file()["components"]
        }
        self.assertIn("双声道播放（左=源语言，右=翻译语言）", labels)
        self.assertIn("完成后的双声道对比（左=源语言，右=翻译语言）", labels)
        demo.close(verbose=False)


if __name__ == "__main__":
    unittest.main()
