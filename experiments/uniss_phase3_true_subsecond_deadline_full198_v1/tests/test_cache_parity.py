from __future__ import annotations

import unittest

import torch

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.model import (
    ChunkCausalWhisperVQAdapter,
)


class CacheParityTest(unittest.TestCase):
    def test_chunked_output_matches_full_output(self) -> None:
        torch.manual_seed(11)
        model = ChunkCausalWhisperVQAdapter(
            hidden_size=24, layers=4, kernel_size=5, expansion=2
        ).eval()
        value = torch.randn(2, 37, 24)
        with torch.no_grad():
            full = model(value)
            state = None
            chunks = []
            for start, end in ((0, 3), (3, 11), (11, 12), (12, 29), (29, 37)):
                output, state = model.forward_chunk(value[:, start:end], state)
                chunks.append(output)
            streamed = torch.cat(chunks, dim=1)
        torch.testing.assert_close(full, streamed, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
