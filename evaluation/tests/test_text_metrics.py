import unittest

from evaluation import text_metrics


class TextMetricsTest(unittest.TestCase):
    def test_normalize_english_keeps_apostrophes(self):
        self.assertEqual(text_metrics.normalize_english("It's—Fine, RIGHT?!"), "it's fine right")

    def test_normalize_chinese_simplifies_and_splits_characters(self):
        value = text_metrics.normalize_chinese("繁體，測試！", simplify=lambda text: text.replace("繁體", "繁体").replace("測試", "测试"))
        self.assertEqual(value, "繁 体 测 试")

    def test_identical_corpora_score_one_hundred(self):
        rows = [
            {
                "id": "en-1",
                "mode": "quality",
                "src_lang": "cmn",
                "tgt_lang": "eng",
                "generated_translation": "It's completely ready right now.",
                "translation_ref": "It's completely ready right now!",
            },
            {
                "id": "zh-1",
                "mode": "performance",
                "src_lang": "eng",
                "tgt_lang": "cmn",
                "generated_translation": "繁體測試。",
                "translation_ref": "繁体测试！",
            },
        ]
        report = text_metrics.compute_grouped_bleu(
            rows,
            hypothesis_field="generated_translation",
            reference_field="translation_ref",
        )
        self.assertEqual(report["scored_count"], 2)
        for group in report["groups"].values():
            self.assertAlmostEqual(group["score"], 100.0)


if __name__ == "__main__":
    unittest.main()
