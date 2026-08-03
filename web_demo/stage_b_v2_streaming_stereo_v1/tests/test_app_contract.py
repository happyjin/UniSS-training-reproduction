from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from web_demo.streaming_s2st_r2_v1.session_manager import SessionRegistry
from web_demo.stage_b_v2_streaming_stereo_v1.app_gradio import (
    build_demo,
    format_student_final_status,
    parse_args,
)
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

    def test_final_status_distinguishes_nca_from_ca(self) -> None:
        temporary_root = Path("/opt/dlami/nvme/jasonleeeli/tmp")
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=temporary_root
        ) as directory:
            result_path = Path(directory) / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "latency_metrics": {
                            "first_write_source_timeline_nca_ms": 5460.0,
                            "first_audio_timeline_placement_nca_ms": 5460.0,
                            "first_write_decision_ca_estimate_ms": 5481.0,
                            "first_audio_ready_ca_estimate_ms": 10930.0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            status = format_student_final_status(
                SimpleNamespace(
                    result_json_path=str(result_path),
                    model_label="student",
                    mode="causal",
                    source_duration_seconds=5.46,
                    translation_duration_seconds=4.4,
                    total_seconds=6.7,
                    fallback_used=False,
                    fallback_reason=None,
                    forced_actions=0,
                    structural_recoveries=0,
                )
            )
        self.assertIn("源时间线/NCA", status)
        self.assertIn("服务端就绪（CA估算）", status)
        self.assertIn("10930 ms", status)


if __name__ == "__main__":
    unittest.main()
