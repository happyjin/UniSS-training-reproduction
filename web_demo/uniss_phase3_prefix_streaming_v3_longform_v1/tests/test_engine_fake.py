import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from web_demo.uniss_phase3_prefix_streaming_v3_longform_v1.config import (
    LongFormDemoConfig,
)
from web_demo.uniss_phase3_prefix_streaming_v3_longform_v1.engine import (
    BoundedLongFormEngine,
)


class _FakeBaseEngine:
    adapter_manifest = {"selected_iteration": 8000}

    def __init__(self, fail_quality_first=False):
        self.calls = 0
        self.fail_quality_first = fail_quality_first

    def stream(self, path, *, direction, chunk_ms):
        self.calls += 1
        source, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        duration = len(source) / sample_rate
        target_path = Path(path).with_name("fake_translation.wav")
        target = np.ones(int(sample_rate * min(1.0, duration / 4.0)), dtype=np.float32) * 0.1
        sf.write(target_path, target, sample_rate, subtype="PCM_16")
        result_path = Path(path).with_name("fake_result.json")
        result_path.write_text("{}\n", encoding="utf-8")
        bad = self.fail_quality_first and self.calls == 1
        yield SimpleNamespace(
            result=SimpleNamespace(
                translation_path=str(target_path),
                result_path=str(result_path),
                first_write_source_ms=1000.0,
                first_audio_source_ms=1200.0,
                processing_seconds=0.1,
                rtf=0.01,
                wait_events=1,
                write_events=1,
                committed_text_tokens=160 if bad else 3,
                semantic_tokens=50 if bad else 5,
                translation=f"window-{self.calls}-{direction}-{chunk_ms}",
                events=[],
            )
        )


class LongFormEngineFakeTest(unittest.TestCase):
    def test_multiple_windows_produce_one_auditable_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.wav"
            sample_rate = 16_000
            # 65 seconds forces three bounded source windows.
            sf.write(
                source_path,
                np.ones(65 * sample_rate, dtype=np.float32) * 0.05,
                sample_rate,
                subtype="PCM_16",
            )
            fake = _FakeBaseEngine()
            config = LongFormDemoConfig(output_root=root / "outputs")
            engine = BoundedLongFormEngine(config, base_engine=fake)
            updates = list(engine.run(source_path, direction="zh-en", chunk_ms=480))
            result = updates[-1].result

            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.failed_windows, 0)
            self.assertEqual(result.completed_windows, 3)
            self.assertEqual(fake.calls, 3)
            self.assertLessEqual(result.maximum_observed_window_seconds, 30.0)
            self.assertAlmostEqual(result.source_duration_seconds, 65.0, places=3)
            self.assertTrue(Path(result.stereo_path).is_file())
            self.assertTrue(Path(result.result_path).is_file())
            self.assertEqual(result.selected_iteration, 8000)

    def test_quality_gate_bisects_saturated_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.wav"
            sample_rate = 16_000
            sf.write(
                source_path,
                np.ones(20 * sample_rate, dtype=np.float32) * 0.05,
                sample_rate,
                subtype="PCM_16",
            )
            fake = _FakeBaseEngine(fail_quality_first=True)
            engine = BoundedLongFormEngine(
                LongFormDemoConfig(output_root=root / "outputs"), base_engine=fake
            )
            result = list(
                engine.run(source_path, direction="zh-en", chunk_ms=480)
            )[-1].result

            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(fake.calls, 3)
            self.assertEqual(result.completed_windows, 2)
            self.assertEqual(result.retry_windows, 2)
            self.assertTrue(all(record.depth == 1 for record in result.records))
            self.assertTrue(
                all("text_token_saturation" in (record.retry_reason or "") for record in result.records)
            )


if __name__ == "__main__":
    unittest.main()
