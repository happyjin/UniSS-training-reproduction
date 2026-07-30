from __future__ import annotations

import torch

from training.simul_uniss.subsecond_v1.model import CausalAudioStudentV2


def _reference_pack(
    projected: torch.Tensor,
    total_lengths: torch.Tensor,
    utterance_lengths: torch.Tensor,
    right: int,
) -> torch.Tensor:
    batch = projected.shape[0]
    max_utterance = int(utterance_lengths.max())
    result = projected.new_zeros(batch, max_utterance + right, projected.shape[-1])
    for row in range(batch):
        utterance = int(utterance_lengths[row])
        total = int(total_lengths[row])
        result[row, :utterance] = projected[row, :utterance]
        available_right = min(right, max(0, total - utterance))
        if available_right:
            result[row, max_utterance : max_utterance + available_right] = projected[
                row, utterance : utterance + available_right
            ]
    return result


def test_vectorized_emformer_pack_matches_reference_forward_and_gradient() -> None:
    generator = torch.Generator().manual_seed(20260730)
    reference_input = torch.randn(4, 12, 7, generator=generator, requires_grad=True)
    vector_input = reference_input.detach().clone().requires_grad_(True)
    total_lengths = torch.tensor([7, 12, 5, 9])
    utterance_lengths = torch.tensor([5, 8, 5, 7])

    reference = _reference_pack(reference_input, total_lengths, utterance_lengths, right=2)
    vectorized = CausalAudioStudentV2._pack_emformer_input(
        vector_input, total_lengths, utterance_lengths, right=2
    )
    torch.testing.assert_close(vectorized, reference, rtol=0, atol=0)

    reference.square().sum().backward()
    vectorized.square().sum().backward()
    torch.testing.assert_close(vector_input.grad, reference_input.grad, rtol=0, atol=0)


def test_vectorized_emformer_pack_supports_zero_right_context() -> None:
    projected = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
    total_lengths = torch.tensor([6, 4])
    utterance_lengths = torch.tensor([5, 3])
    reference = _reference_pack(projected, total_lengths, utterance_lengths, right=0)
    vectorized = CausalAudioStudentV2._pack_emformer_input(
        projected, total_lengths, utterance_lengths, right=0
    )
    torch.testing.assert_close(vectorized, reference, rtol=0, atol=0)
