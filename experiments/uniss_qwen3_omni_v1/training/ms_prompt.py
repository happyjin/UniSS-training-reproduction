"""E2E-v0's prompt, shared by training (pretrain_thinker_ms.py) and serving (evaluation/e2e_ms_stream.py) without
pulling Megatron into the vLLM environment."""

from __future__ import annotations

MS_SUFFIX = (" Several people may be speaking. Each time the speaker changes, first write the label [S<n>@<t>]:"
             " speakers are numbered in the order they first speak (S1, S2, ...), and t is the second where that"
             " speech starts.")


def ms_instruction(direction: str) -> str:
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction

    return inference_instruction()["cmn" if direction.endswith("zh") else "eng"] + MS_SUFFIX
