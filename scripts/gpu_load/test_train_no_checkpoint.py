"""CPU-only checks for memory accounting, process ownership and real backward updates."""

import importlib.util
from pathlib import Path
import signal
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("training_load", Path(__file__).with_name("train_mist_workload.py"))
load = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load)


class Allocation:
    def __init__(self, cuda, size):
        self.cuda, self.size = cuda, size
        cuda.padding += size

    def numel(self):
        return self.size

    def zero_(self):
        return self

    def __del__(self):
        self.cuda.padding -= self.size


class FakeCuda:
    class OutOfMemoryError(RuntimeError):
        pass

    def __init__(self):
        self.padding = 0
        self.base = 4000 * load.MIB
        self.training_peak = 8000 * load.MIB
        self.cache = 4000 * load.MIB
        self.outside = 500 * load.MIB
        self.total = 16000 * load.MIB

    def max_memory_allocated(self):
        return self.training_peak + self.padding

    def memory_reserved(self):
        return self.base + self.padding + self.cache

    def mem_get_info(self):
        return self.total - self.outside - self.memory_reserved(), self.total

    def empty_cache(self):
        self.cache = 0


class TrainingLoadTests(unittest.TestCase):
    def test_memory_fill_reserves_activation_peak_and_headroom(self):
        a = load.parse_args(["--devices", "0", "--warmup-steps", "1", "--min-free-mib", "1024"])
        cuda = FakeCuda()
        torch = SimpleNamespace(cuda=cuda, uint8="uint8", empty=lambda n, **kw: Allocation(cuda, n))
        fill = load.MemoryFill(torch, a)
        fill.after_step()
        predicted_peak = cuda.outside + cuda.training_peak + fill.held
        self.assertLessEqual(predicted_peak, cuda.total - 1024 * load.MIB)
        self.assertGreater(fill.held, 6000 * load.MIB)
        # Larger sequences must displace padding rather than permanently raising total use.
        cuda.training_peak += 1000 * load.MIB
        fill.after_step()
        self.assertLessEqual(cuda.outside + cuda.training_peak + fill.held, cuda.total - 1024 * load.MIB)
        old_held = fill.held
        fill.release_after_oom()
        self.assertEqual(cuda.padding, 0)
        fill.after_step()
        self.assertLess(fill.held, old_held)

    def test_active_pid_prevents_start_even_with_little_memory(self):
        a = load.parse_args(["--devices", "0"])
        cards = {0: {"pids": {123}, "used": 1}}
        self.assertEqual(load.busy_cards(a, cards), [0])
        with patch.object(load, "gpu_snapshot", return_value=cards):
            with self.assertRaisesRegex(RuntimeError, "GPUs busy"):
                load.wait_until_ready(a)

    def test_completion_marker_is_required_before_any_gpu_check(self):
        a = load.parse_args(["--devices", "0", "--wait-for-file", "/tmp/not-created-by-this-test.ready"])
        with patch.object(Path, "is_file", return_value=False), \
                patch.object(load, "gpu_snapshot") as snapshot, \
                patch.object(load.time, "sleep", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                load.wait_until_ready(a)
            snapshot.assert_not_called()

    def test_new_foreign_process_stops_only_own_worker_group(self):
        a = load.parse_args(["--devices", "0"])
        idle = {0: {"uuid": "GPU-test", "used": 0, "total": 16000, "pids": set()}}
        occupied = {0: {**idle[0], "pids": {9001, 9002}}}
        child = SimpleNamespace(pid=9001, poll=lambda: None, wait=lambda **kw: 0)
        with patch.object(load, "wait_until_ready", return_value=idle), \
                patch.object(load, "gpu_snapshot", side_effect=[idle, occupied]), \
                patch.object(load.subprocess, "Popen", return_value=child), \
                patch.object(load.os, "killpg") as killpg:
            with self.assertRaisesRegex(RuntimeError, "another GPU process"):
                load.supervise(a)
            killpg.assert_called_once_with(9001, signal.SIGTERM)

    def test_real_gstar_loss_backward_and_optimizer_without_saving(self):
        import torch
        from torch import nn

        sys.path[:0] = [str(load.REPO), str(load.REPO / "third_party/Megatron-LM")]
        from megatron.core.transformer.transformer_config import TransformerConfig
        from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import (
            EOS_ID, NEW0, SPK0, ThinkerGStarModel,
        )

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(8, 8)

        class TextModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Block()])

            def forward(self, inputs_embeds, **kwargs):
                return SimpleNamespace(last_hidden_state=self.layers[0].q_proj(inputs_embeds))

        class Thinker(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8))
                self.embedding = nn.Embedding(151936, 8)
                self.lm_head = nn.Linear(8, 151936, bias=False)
                self.model = TextModel()

            def get_input_embeddings(self):
                return self.embedding

        torch.manual_seed(123)
        tiny = SimpleNamespace(thinker=Thinker().bfloat16())
        config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=2)
        # The installed torch accelerator probe raises without driver access.
        # This test intentionally exercises only CPU math and tiny model weights.
        with patch.object(torch.accelerator, "current_accelerator", return_value=torch.device("cpu")), \
                patch("transformers.Qwen3OmniMoeForConditionalGeneration.from_pretrained", return_value=tiny), \
                patch.object(torch, "save", side_effect=AssertionError("checkpoint write attempted")):
            model = ThinkerGStarModel(config, qwen3_path="unused", lora_rank=2, lora_alpha=4,
                                     lora_from=0, recompute=False, init_merged=False)
            ids = torch.tensor([[1, NEW0, SPK0, EOS_ID]])
            batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
            before = model.new_head.detach().clone()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            metrics = load.backward_update(torch, model, iter([batch, batch]), 2, device="cpu")
            self.assertTrue(bool(torch.isfinite(metrics).all()))
            self.assertEqual(len(metrics), 6)
            self.assertGreater(model.new_head.grad.abs().sum().item(), 0)
            optimizer.step()
            self.assertFalse(torch.equal(model.new_head.detach(), before))
            self.assertIsNone(model.thinker.embedding.weight.grad)


if __name__ == "__main__":
    unittest.main()
