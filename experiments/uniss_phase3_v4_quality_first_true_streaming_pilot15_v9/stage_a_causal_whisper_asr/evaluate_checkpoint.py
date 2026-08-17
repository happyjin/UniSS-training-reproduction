#!/usr/bin/env python3
"""V9 free-running ASR diagnosis with the trained causal-code adapter."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.distributed.checkpoint as dcp

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr import (
    evaluate_checkpoint as base,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.frontend import (
    TrainableSharedCausalWhisperVQ,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v9.stage_a_causal_whisper_asr.training.objective import (
    StageAObjective,
)


def load_objective(
    checkpoint,
    model_path,
    device: torch.device,
) -> StageAObjective:
    objective = StageAObjective(
        TrainableSharedCausalWhisperVQ(model_path, gradient_checkpointing=False),
        qwen_hidden_size=896,
    ).to(device=device, dtype=torch.bfloat16).eval()
    state = {
        f"stage_a_objective.{name}": value
        for name, value in objective.state_dict().items()
    }
    dcp.load(state, checkpoint_id=str(checkpoint))
    return objective


def adapt_pooled_hidden(
    objective: StageAObjective,
    hidden: torch.Tensor,
) -> torch.Tensor:
    adapted, _ = objective.code_adapter(hidden)
    return adapted


@torch.inference_mode()
def acoustic_outputs(
    objective: StageAObjective,
    waveform: torch.Tensor,
    source_glm: Sequence[int],
    *,
    chunk_ms: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, object]]:
    device = next(objective.parameters()).device
    waveform = waveform.unsqueeze(0).to(device)
    lengths = torch.tensor([waveform.shape[1]], dtype=torch.long, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = objective.frontend(waveform, lengths, chunk_ms=chunk_ms)
    hidden = output.pooled_hidden[0, : int(output.pooled_lengths[0])]
    if len(hidden) + 1 == len(source_glm):
        deficit = base.terminal_codec_extension_deficit_samples(
            int(lengths[0]), len(hidden), len(source_glm)
        )
        if deficit is None:
            raise ValueError(
                "unaudited terminal causal-token extension during V9 diagnosis"
            )
        hidden = torch.cat((hidden, hidden[-1:]), dim=0)
    if len(hidden) != len(source_glm):
        raise ValueError(
            f"causal GLM length mismatch: {len(hidden)} vs {len(source_glm)}"
        )
    adapted = adapt_pooled_hidden(objective, hidden)
    codes = objective._nearest_codes(adapted)
    residual = objective.bridge_projection(objective.bridge_norm(adapted))
    ctc_logits = objective.ctc_head(output.frame_hidden)[
        0, : int(output.frame_lengths[0])
    ]
    raw_ctc = ctc_logits.float().argmax(dim=-1).tolist()
    collapsed = base.collapse_ctc(raw_ctc, objective.ctc_blank_id)
    ctc_text = bytes(value for value in collapsed if 0 <= value < 256).decode(
        "utf-8", errors="replace"
    )
    diagnostic = {
        "input_frames": len(raw_ctc),
        "raw_nonblank_frames": sum(
            value != objective.ctc_blank_id for value in raw_ctc
        ),
        "collapsed_nonblank_tokens": len(collapsed),
        "blank_ratio": sum(value == objective.ctc_blank_id for value in raw_ctc)
        / max(1, len(raw_ctc)),
        "text": ctc_text,
    }
    return (codes, residual), diagnostic


def main() -> None:
    base.load_objective = load_objective
    base.acoustic_outputs = acoustic_outputs
    base.main()


if __name__ == "__main__":
    main()


__all__ = ["acoustic_outputs", "adapt_pooled_hidden", "load_objective", "main"]
