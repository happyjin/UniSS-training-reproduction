import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from training import constants_uniss as c
from training import generate_unist_eval_audio as eval_audio


class GenerateUniSTEvalAudioTest(unittest.TestCase):
    def test_truncate_at_eos(self):
        self.assertEqual(eval_audio.truncate_at_eos([1, 2, c.TOKEN_EOS, 3]), [1, 2, c.TOKEN_EOS])
        self.assertEqual(eval_audio.truncate_at_eos([1, 2, 3]), [1, 2, 3])

    def test_extract_bicodec_semantic_values(self):
        ids = [
            1,
            c.BICODEC_SEMANTIC_SPAN.id_for(7),
            c.BICODEC_SEMANTIC_SPAN.id_for(8191),
            c.TOKEN_END_SEMANTIC,
        ]
        self.assertEqual(eval_audio.extract_bicodec_semantic_values(ids), [7, 8191])

    def test_clean_generated_text(self):
        text = "<|task_asr|> hello <|end_content|>"
        self.assertEqual(eval_audio.clean_generated_text(text), "hello")

    def test_safe_sample_name(self):
        name = eval_audio.safe_sample_name(3, "dataset/item:1", "quality")
        self.assertTrue(name.startswith("00003_quality_"))
        self.assertNotIn("/", name)
        self.assertNotIn(":", name)

    def test_parse_args_supports_source_audio(self):
        args = eval_audio.parse_args(
            [
                "--input",
                "dev.parquet",
                "--model",
                "hf-model",
                "--output-dir",
                "out",
                "--save-source-audio",
            ]
        )
        self.assertTrue(args.save_source_audio)

    def test_maybe_decode_audio_passes_global_then_semantic_tokens(self):
        class FakeSpeechTokenizer:
            def __init__(self):
                self.decoded = None
                self.saved = None

            def decode(self, tokens):
                self.decoded = tokens.detach().cpu().tolist()
                return [0.0, 0.1]

            def save_audio(self, audio, output_path, sample_rate):
                self.saved = (audio, Path(output_path), sample_rate)

        with TemporaryDirectory() as tmpdir:
            fake = FakeSpeechTokenizer()
            output_path = Path(tmpdir) / "sample.wav"
            audio_path, error = eval_audio.maybe_decode_audio(
                speech_tokenizer=fake,
                global_values=[1, 2],
                semantic_values=[7, 8, 9],
                output_path=output_path,
                device=torch.device("cpu"),
            )

        self.assertEqual(audio_path, str(output_path))
        self.assertIsNone(error)
        self.assertEqual(fake.decoded, [1, 2, 7, 8, 9])
        self.assertEqual(fake.saved, ([0.0, 0.1], output_path, 16000))

    def test_tts_mode_uses_phase1_tts_layout_and_source_reference(self):
        record = {
            "id": "dev-1",
            "src_lang": "eng",
            "tgt_lang": "cmn",
            "transcription": "hi",
            "translation": "你好",
            "source_glm": [1, 2],
            "source_bicodec": [3, 4],
            "target_bicodec": [5, 6],
            "bicodec_global": list(range(32)),
        }

        sample = eval_audio.build_eval_sample(record, mode="tts", text_encoder=lambda text: [1000 + ord(text[0])])

        self.assertEqual(sample.task, "tts")
        self.assertEqual(eval_audio.reference_bicodec_values(record, "tts"), [3, 4])
        self.assertEqual(eval_audio.reference_bicodec_values(record, "quality"), [5, 6])


if __name__ == "__main__":
    unittest.main()
