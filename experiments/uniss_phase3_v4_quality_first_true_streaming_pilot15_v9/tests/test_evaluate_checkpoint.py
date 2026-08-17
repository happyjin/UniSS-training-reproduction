import torch

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v9.stage_a_causal_whisper_asr.evaluate_checkpoint import (
    adapt_pooled_hidden,
)


class FakeAdapter:
    def __call__(self, hidden: torch.Tensor):
        residual = torch.full_like(hidden, 2.0)
        return hidden + residual, residual


class FakeObjective:
    code_adapter = FakeAdapter()


def test_v9_evaluator_applies_trained_adapter_before_quantization() -> None:
    hidden = torch.zeros(3, 4)
    adapted = adapt_pooled_hidden(FakeObjective(), hidden)
    assert torch.equal(adapted, torch.full_like(hidden, 2.0))
