from __future__ import annotations

import pytest
import torch
from torch import nn

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.packing import (
    LOSS_STREAMING_ASR,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.frontend import (
    CausalWhisperOutput,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.objective import (
    StageAObjective,
    chunk_pair_for_progress,
    distributed_stage_a_objective,
    stable_multichunk_mask,
    terminal_codec_extension_deficit_samples,
)


class TinyFrontend(nn.Module):
    hidden_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.register_buffer(
            "tiny_codebook",
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ),
        )

    @property
    def codebook(self) -> torch.Tensor:
        return self.tiny_codebook

    def forward(
        self,
        waveform: torch.Tensor,
        waveform_lengths: torch.Tensor,
        *,
        chunk_ms: int,
    ) -> CausalWhisperOutput:
        del waveform, chunk_ms
        batch = len(waveform_lengths)
        frame = torch.zeros(batch, 8, 4, device=waveform_lengths.device)
        frame[:, :4, 0] = self.scale
        frame[:, 4:, 1] = self.scale
        pooled = frame.reshape(batch, 2, 4, 4).mean(dim=2)
        return CausalWhisperOutput(
            frame,
            torch.full((batch,), 8, device=frame.device, dtype=torch.long),
            pooled,
            torch.full((batch,), 2, device=frame.device, dtype=torch.long),
        )


def test_curriculum_reaches_deployment_chunk_without_dropping_larger_view() -> None:
    assert chunk_pair_for_progress(0.0, 0) == (1280, 1280)
    assert chunk_pair_for_progress(0.2, 1) == (960, 1280)
    assert chunk_pair_for_progress(0.9, 1) == (160, 320)


def test_stable_multichunk_mask_compares_only_equal_visibility_endpoints() -> None:
    mask = stable_multichunk_mask(
        torch.tensor([2560, 5120]),
        sequence_length=16,
        first_chunk_ms=160,
        second_chunk_ms=320,
    )
    assert mask[0, :8].all()
    assert not mask[1, :8].any()
    assert mask[1, 8:16].all()


def test_objective_replaces_offline_glm_embeddings_and_has_finite_losses() -> None:
    objective = StageAObjective(
        TinyFrontend(),
        qwen_hidden_size=6,
        ctc_output_size=5,
        ctc_blank_id=4,
        glm_semantic_offset=0,
    )
    decoder = torch.randn(5, 1, 6)
    embeddings = torch.randn(8, 6)
    logits = torch.randn(5, 8, requires_grad=True)
    labels = torch.tensor([1, 2, 3, 4, 5])
    loss_mask = torch.ones(5)
    loss_kinds = torch.full((5,), LOSS_STREAMING_ASR)
    batch = {
        "waveform": torch.zeros(1, 2560),
        "waveform_lengths": torch.tensor([2560]),
        "glm_ids": torch.tensor([[0, 1]]),
        "glm_positions": torch.tensor([[1, 2]]),
        "glm_lengths": torch.tensor([2]),
        "acoustic_batch": torch.tensor([0]),
        "ctc_ids": torch.tensor([[0, 1]]),
        "ctc_lengths": torch.tensor([2]),
        "disabled_acoustics": torch.tensor([0]),
    }
    output = objective(
        decoder,
        embeddings,
        logits,
        labels,
        loss_mask,
        loss_kinds,
        batch,
        original_seq_length=5,
        chunk_ms=160,
        consistency_chunk_ms=320,
    )
    assert torch.equal(output.decoder_input[1, 0], embeddings[0])
    assert torch.equal(output.decoder_input[2, 0], embeddings[1])
    total, metrics = distributed_stage_a_objective(output)
    assert torch.isfinite(total)
    assert all(torch.isfinite(value) for value in metrics.values())
    total.backward()
    assert objective.ctc_head.weight.grad is not None
    assert objective.frontend.scale.grad is not None


def test_terminal_extension_accepts_only_formally_audited_pcm_geometries() -> None:
    assert terminal_codec_extension_deficit_samples(2560, 2, 3) == 0
    assert terminal_codec_extension_deficit_samples(2240, 2, 3) == 320
    assert terminal_codec_extension_deficit_samples(2239, 2, 3) is None
    assert terminal_codec_extension_deficit_samples(2560, 2, 4) is None
    assert terminal_codec_extension_deficit_samples(2560, 3, 4) is None


def test_objective_extends_only_formally_audited_terminal_codec_boundary() -> None:
    objective = StageAObjective(
        TinyFrontend(),
        qwen_hidden_size=6,
        ctc_output_size=5,
        ctc_blank_id=4,
        glm_semantic_offset=0,
    )
    decoder = torch.randn(5, 1, 6)
    embeddings = torch.randn(8, 6)
    batch = {
        "waveform": torch.zeros(1, 2560),
        "waveform_lengths": torch.tensor([2560]),
        "glm_ids": torch.tensor([[0, 1, 1]]),
        "glm_positions": torch.tensor([[1, 2, 3]]),
        "glm_lengths": torch.tensor([3]),
        "acoustic_batch": torch.tensor([0]),
        "ctc_ids": torch.tensor([[0, 1]]),
        "ctc_lengths": torch.tensor([2]),
        "disabled_acoustics": torch.tensor([0]),
    }
    prepared = objective.prepare(
        decoder,
        embeddings,
        batch,
        original_seq_length=5,
        chunk_ms=160,
        consistency_chunk_ms=160,
    )
    assert prepared.causal_glm_terminal_extensions.item() == 1
    assert torch.equal(prepared.decoder_input[2], prepared.decoder_input[3])

    batch["waveform_lengths"] = torch.tensor([2240])
    prepared = objective.prepare(
        decoder,
        embeddings,
        batch,
        original_seq_length=5,
        chunk_ms=160,
        consistency_chunk_ms=160,
    )
    assert prepared.causal_glm_terminal_extensions.item() == 1

    batch["waveform_lengths"] = torch.tensor([2239])
    batch["acoustic_sample_ids"] = ["boundary-regression"]
    batch["source_audio_paths"] = ["/tmp/boundary.flac"]
    with pytest.raises(ValueError, match="boundary-regression"):
        objective.prepare(
            decoder,
            embeddings,
            batch,
            original_seq_length=5,
            chunk_ms=160,
            consistency_chunk_ms=160,
        )
