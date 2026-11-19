"""Lance duration_ms → source_duration_ms for --group_by_length lengths."""

import os
import unittest

from custom.lance_manifest import MANIFEST_COLUMNS, build_s2st_record


class LanceGroupByLengthTest(unittest.TestCase):

    def test_manifest_columns_include_duration_ms(self):
        self.assertIn('duration_ms', MANIFEST_COLUMNS)

    def test_s2st_record_renames_duration_ms(self):
        rec = build_s2st_record({
            '_rowid': 42,
            'tgt_lang': 'en',
            'src_lang': 'zh',
            'target_text': 'hello',
            'duration_ms': 1120,
        }, use_mel=True, use_speaker_embedding=True)
        self.assertEqual(rec['source_duration_ms'], 1120)
        self.assertEqual(rec['target_codec_codes'], 'lance://42/codes')
        self.assertEqual(rec['source_input_features'], 'lance://42/features')

    def test_audio_tokens_from_ms_matches_omni_tower(self):
        # Same formula as custom.dataset._audio_tokens_from_ms (no swift import).
        def audio_tokens_from_ms(duration_ms):
            mel = int(duration_ms) * 16000 // 1000 // 160
            if mel <= 2:
                return 0
            conv1 = (mel - 1) // 2 + 1
            return max((conv1 - 2) // 2 + 1, 0)

        self.assertEqual(audio_tokens_from_ms(1120), 28)
        self.assertEqual(audio_tokens_from_ms(3740), 93)
        self.assertEqual(audio_tokens_from_ms(5580), 139)

        os.environ.setdefault('OMNISPEAK_LENGTH_BASE_TOKENS', '48')
        base = int(os.environ['OMNISPEAK_LENGTH_BASE_TOKENS'])
        length = base + audio_tokens_from_ms(1120) + len('hello')
        self.assertEqual(length, 48 + 28 + 5)


if __name__ == '__main__':
    unittest.main()
