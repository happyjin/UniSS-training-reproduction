from __future__ import annotations

import torch

from experiments.uniss_phase3_runtime_parity_streaming_v2.frontend.audio_cached_frontend import (
    BLOCK_SAMPLES,
    PCM_LEFT_CONTEXT,
    StreamingCachedWhisperVQFrontend,
)


def test_causal_stft_geometry_has_exactly_sixteen_frames() -> None:
    analysis_samples = PCM_LEFT_CONTEXT + BLOCK_SAMPLES
    assert (analysis_samples - 400) // 160 + 1 == 16


def test_causal_conv_helper_retains_only_left_context() -> None:
    module = torch.nn.Conv1d(3, 4, kernel_size=3, stride=1, padding=0)
    current = torch.randn(1, 3, 16)
    tail = torch.randn(1, 3, 2)
    output, next_tail = StreamingCachedWhisperVQFrontend._causal_conv_block(
        module, current, tail
    )
    assert output.shape == (1, 4, 16)
    torch.testing.assert_close(next_tail, current[:, :, -2:])
