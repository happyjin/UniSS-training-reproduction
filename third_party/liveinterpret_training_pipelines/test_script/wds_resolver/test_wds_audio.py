"""CPU tests for indexed WDS byte-range audio resolution."""

import io
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from custom.speaker_encoder import load_reference_waveform
from custom.wds_audio import (
    WDSAudioError,
    parse_wds_uri,
    read_wds_audio_bytes,
    resolve_audio_reference,
)


def _wav_bytes(sample_rate=8000, num_samples=800):
    samples = (np.sin(np.arange(num_samples) * 2 * np.pi * 220 / sample_rate) * 12000).astype('<i2')
    output = io.BytesIO()
    with wave.open(output, 'wb') as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(samples.tobytes())
    return output.getvalue()


class WDSAudioResolverTest(unittest.TestCase):

    def setUp(self):
        # Create a temp dir that mimics the container mount layout:
        #   <root>/audio/processed/shards/process_version=test_process_v1/shard-00000.tar
        # The URI netloc will be set to <root> so wds://<root>/audio/... resolves
        # to /<root>/audio/... on disk.
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        shard_dir = root / 'audio' / 'processed' / 'shards' / 'process_version=test_process_v1'
        shard_dir.mkdir(parents=True)
        self.payload = _wav_bytes()
        prefix = b'not-a-tar-header-but-byte-range-semantics-are-identical'
        suffix = b'trailing-shard-data'
        shard = shard_dir / 'shard-00000.tar'
        shard.write_bytes(prefix + self.payload + suffix)
        # netloc = root path (absolute), so wds://<root>/audio/... → /<root>/audio/...
        self.uri = (
            f'wds://{root}/audio/processed/shards/'
            'process_version=test_process_v1/shard-00000.tar'
            f'?offset={len(prefix)}&length={len(self.payload)}&format=wav')
        self.env = mock.patch.dict(os.environ, {
            'WDS_AUDIO_MODE': 'auto',
            'WDS_AUDIO_STRICT': 'true',
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp_dir.cleanup()

    def test_parse_and_read_exact_range(self):
        reference = parse_wds_uri(self.uri)
        self.assertEqual(reference.audio_format, 'wav')
        self.assertEqual(read_wds_audio_bytes(self.uri), self.payload)
        self.assertEqual(resolve_audio_reference(self.uri), self.payload)

    def test_resolved_bytes_decode_for_speaker(self):
        waveform, sample_rate = load_reference_waveform(self.uri)
        self.assertEqual(sample_rate, 8000)
        self.assertEqual(waveform.dtype, np.float32)
        self.assertEqual(waveform.shape, (800,))
        self.assertTrue(np.isfinite(waveform).all())

    def test_auto_preserves_legacy_path(self):
        self.assertEqual(
            resolve_audio_reference('/fsx/legacy/audio.wav'),
            '/fsx/legacy/audio.wav')

    def test_modes_reject_unexpected_reference(self):
        with mock.patch.dict(os.environ, {'WDS_AUDIO_MODE': 'off'}):
            with self.assertRaisesRegex(WDSAudioError, 'WDS_AUDIO_MODE=off'):
                resolve_audio_reference(self.uri)

        with mock.patch.dict(os.environ, {'WDS_AUDIO_MODE': 'required'}):
            with self.assertRaisesRegex(WDSAudioError, 'WDS_AUDIO_MODE=required'):
                resolve_audio_reference('/fsx/legacy/audio.wav')

    def test_rejects_out_of_bounds_range(self):
        bad_uri = self.uri.replace('offset=', 'offset=999999999&ignored=')
        with self.assertRaises(WDSAudioError):
            read_wds_audio_bytes(bad_uri)


if __name__ == '__main__':
    unittest.main()
