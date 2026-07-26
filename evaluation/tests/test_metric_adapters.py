import unittest

from evaluation import asr_transcribe, autopcp_metrics, utmos_metrics


class MetricAdaptersTest(unittest.TestCase):
    def test_asr_backend_routing(self):
        self.assertEqual(asr_transcribe.target_asr_backend("eng"), "whisper-large-v3")
        self.assertEqual(asr_transcribe.target_asr_backend("cmn"), "paraformer-zh")

    def test_score_aggregation(self):
        rows = [
            {"mode": "quality", "src_lang": "eng", "tgt_lang": "cmn", "utmos_score": 3.0},
            {"mode": "quality", "src_lang": "eng", "tgt_lang": "cmn", "utmos_score": 4.0},
        ]
        report = utmos_metrics.aggregate_scores(rows)
        self.assertEqual(report["groups"]["quality:eng->cmn"]["mean"], 3.5)
        autopcp_rows = [
            {"mode": "performance", "src_lang": "cmn", "tgt_lang": "eng", "autopcp_score": 2.0},
            {"mode": "performance", "src_lang": "cmn", "tgt_lang": "eng", "autopcp_score": 4.0},
        ]
        report = autopcp_metrics.aggregate_scores(autopcp_rows)
        self.assertEqual(report["groups"]["performance:cmn->eng"]["mean"], 3.0)


if __name__ == "__main__":
    unittest.main()
