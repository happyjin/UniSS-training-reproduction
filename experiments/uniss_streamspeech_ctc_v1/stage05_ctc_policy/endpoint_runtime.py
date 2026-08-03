"""Streaming runtime for the real Stage03/Stage03b CTC endpoint heads."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F


def load_endpoint_model(
    checkpoint_path: str | Path,
    *,
    eng_vocab_size: int,
    cmn_vocab_size: int,
    device: torch.device,
):
    from endpoint_model import EndpointCTCStudent
    from training.simul_uniss.subsecond_v2.stage_b_latent_model import (
        LatentStageBModelConfig,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = LatentStageBModelConfig.from_dict(checkpoint["model_config"])
    model = EndpointCTCStudent(
        config, eng_vocab_size=eng_vocab_size, cmn_vocab_size=cmn_vocab_size
    )
    state = checkpoint["model"]
    if any(name.startswith("base.") for name in state):
        state = {
            name[len("base.") :]: value
            for name, value in state.items()
            if name.startswith("base.") and name[len("base.") :] in model.state_dict()
        }
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def streaming_ctc_paths(
    model,
    waveform: torch.Tensor,
    waveform_length: torch.Tensor,
    *,
    source_head: str,
    target_head: str,
):
    """Yield accumulated causal CTC paths after each Emformer segment.

    Features are computed once for evaluation efficiency. Mel extraction uses
    ``center=False`` and every Emformer call receives only its current segment
    plus the configured right context, so no future model frames leak into a
    decision. The final segment is zero-padded only to flush real tail frames.
    """

    projected = model.extract_projected(waveform)
    valid = int(model.stacked_lengths(waveform_length)[0])
    projected = projected[:, :valid]
    segment = int(model.config.segment_frames)
    right = int(model.config.right_context_frames)
    source_path: list[int] = []
    target_path: list[int] = []
    states = None
    for start in range(0, valid, segment):
        real_segment = min(segment, valid - start)
        end = min(valid, start + segment + right)
        chunk = projected[:, start:end]
        expected = segment + right
        if chunk.shape[1] < expected:
            chunk = F.pad(chunk, (0, 0, 0, expected - chunk.shape[1]))
        lengths = torch.full(
            (chunk.shape[0],), expected, dtype=torch.long, device=chunk.device
        )
        hidden, _, states = model.encoder.infer(chunk, lengths, states)
        hidden = model.output_norm(hidden[:, :real_segment])
        source_path.extend(
            model.heads[source_head](hidden)[0].argmax(dim=-1).cpu().tolist()
        )
        target_path.extend(
            model.heads[target_head](hidden)[0].argmax(dim=-1).cpu().tolist()
        )
        consumed_frames = min(valid, start + segment + right)
        # Return snapshots: callers may retain observations for diagnostics and
        # must not see an earlier prefix mutate when the next chunk is decoded.
        yield list(source_path), list(target_path), consumed_frames, start + segment >= valid
