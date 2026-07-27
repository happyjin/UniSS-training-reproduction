from __future__ import annotations

import unittest

import gradio as gr

from web_demo.offline_s2st_phase3_v1.app_gradio import (
    append_history,
    build_demo,
    format_status,
)
from web_demo.offline_s2st_phase3_v1.config import DemoConfig
from web_demo.offline_s2st_phase3_v1.inference_engine import InferenceResult


class AppContractTest(unittest.TestCase):
    def sample_result(self) -> InferenceResult:
        return InferenceResult(
            request_dir="/tmp/request",
            input_audio_path="/tmp/input.wav",
            output_audio_path="/tmp/output.wav",
            result_json_path="/tmp/result.json",
            direction="中文 → 英文",
            model_label="Phase3 full198 iter_0009075",
            mode="Quality",
            transcription="你好世界",
            translation="Hello world",
            input_duration_seconds=1.2,
            output_duration_seconds=1.0,
            total_seconds=2.5,
            warnings=[],
            chunks=[],
        )

    def test_history_contains_transcription_translation_and_audio_notice(self):
        history = append_history([], self.sample_result())
        self.assertEqual(len(history), 2)
        content = history[-1]["content"]
        self.assertIn("你好世界", content)
        self.assertIn("Hello world", content)
        self.assertIn("翻译语音", content)

    def test_status_identifies_frozen_model_and_quality(self):
        status = format_status(self.sample_result())
        self.assertIn("Phase3 full198 iter_0009075", status)
        self.assertIn("Quality", status)

    def test_microphone_keeps_browser_format_for_backend_decoder(self):
        demo = build_demo(DemoConfig(), object())
        inputs = [
            block
            for block in demo.blocks.values()
            if isinstance(block, gr.Audio)
            and block.label == "输入语音 / Record or upload"
        ]
        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0].type, "filepath")
        self.assertIsNone(inputs[0].format)
        self.assertEqual(set(inputs[0].sources), {"microphone", "upload"})

    def test_generated_audio_is_playable_and_downloadable(self):
        demo = build_demo(DemoConfig(), object())
        players = [
            block
            for block in demo.blocks.values()
            if isinstance(block, gr.Audio)
            and block.label == "翻译语音 / Generated speech"
        ]
        self.assertEqual(len(players), 1)
        self.assertEqual(players[0].format, "wav")
        self.assertTrue(players[0].autoplay)
        self.assertTrue(players[0].show_download_button)
        downloads = [
            block
            for block in demo.blocks.values()
            if isinstance(block, gr.File) and block.label == "下载翻译语音 WAV"
        ]
        self.assertEqual(len(downloads), 1)


if __name__ == "__main__":
    unittest.main()
