import unittest
from types import SimpleNamespace

import torch

from experiments.uniss_streamspeech_ctc_v1.stage09_online_runtime.runtime import (
    Stage09OnlineRuntime,
)


class FakeProcessor:
    def vocab_size(self):
        return 3

    def id_to_piece(self, value):
        return ("▁a", "b", "▁c")[value]

    def decode(self, values):
        return "".join(str(value) for value in values)


class RuntimeGeometryTest(unittest.TestCase):
    def runtime(self):
        runtime = Stage09OnlineRuntime.__new__(Stage09OnlineRuntime)
        runtime.base = SimpleNamespace(
            config=SimpleNamespace(stack_factor=4),
            mel_lengths=lambda lengths: torch.tensor([int(lengths[0])]),
        )
        return runtime

    def test_nonfinal_uses_only_complete_stacks(self):
        runtime = self.runtime()
        self.assertEqual(
            runtime._valid_projected_frames(torch.tensor([23]), 10, False), 5
        )

    def test_final_flush_keeps_partial_stack(self):
        runtime = self.runtime()
        self.assertEqual(
            runtime._valid_projected_frames(torch.tensor([23]), 10, True), 6
        )


if __name__ == "__main__":
    unittest.main()
