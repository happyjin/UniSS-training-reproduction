"""Run with the Omni transformers 4.52 environment; CPU suffices for cache tests."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from transformers import DynamicCache

path = Path(__file__).resolve().parents[3] / 'pipelines/streaming_si/omni_talker_accel.py'
spec = importlib.util.spec_from_file_location('omni_talker_accel', path)
accel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accel)


class DummyTalker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.model = SimpleNamespace(config=SimpleNamespace(
            num_attention_heads=2, num_key_value_heads=2, hidden_size=8,
            num_hidden_layers=2, max_position_embeddings=32,
        ))


def dynamic(length, value=1.):
    cache = DynamicCache()
    for layer in range(2):
        k = torch.full((1, 2, length, 4), value + layer)
        cache.update(k, k + 10, layer)
    return cache


class CacheTransitions(unittest.TestCase):
    def setUp(self):
        self.runner = accel.TalkerAccelerator(DummyTalker())
        self.runner.buckets = (8, 16)

    def assert_prefix(self, left, right, length):
        for lk, rk, lv, rv in zip(left.key_cache, right.key_cache, left.value_cache, right.value_cache):
            torch.testing.assert_close(lk[:, :, :length], rk[:, :, :length], rtol=0, atol=0)
            torch.testing.assert_close(lv[:, :, :length], rv[:, :, :length], rtol=0, atol=0)

    def test_growth_and_overflow_preserve_valid_kv(self):
        original = dynamic(6)
        small = self.runner.adopt(original, 7)
        self.assertEqual(small.get_seq_length(), 6)
        self.assert_prefix(original, small, 6)
        large = self.runner.adopt(small, 9)
        self.assertEqual(large.max_cache_len, 16)
        self.assert_prefix(original, large, 6)
        fallback = self.runner.adopt(large, 17)
        self.assertIsInstance(fallback, DynamicCache)
        self.assertTrue(fallback._omni_accel_fallback)
        self.assertEqual(fallback.get_seq_length(), 6)
        self.assert_prefix(original, fallback, 6)
        # Fallback owns its tensors; pool reset for a later session cannot erase it.
        large.reset()
        self.assert_prefix(original, fallback, 6)

    def test_new_sequence_clears_old_tail_and_reuses_storage(self):
        first = self.runner.adopt(dynamic(6), 7)
        address = first.key_cache[0].data_ptr()
        second = self.runner.adopt(dynamic(2, 5.), 3)
        self.assertEqual(second.key_cache[0].data_ptr(), address)
        self.assertEqual(second.get_seq_length(), 2)
        self.assert_prefix(dynamic(2, 5.), second, 2)
        for k, v in zip(second.key_cache, second.value_cache):
            self.assertEqual(torch.count_nonzero(k[:, :, 2:]).item(), 0)
            self.assertEqual(torch.count_nonzero(v[:, :, 2:]).item(), 0)


if __name__ == '__main__':
    unittest.main()
