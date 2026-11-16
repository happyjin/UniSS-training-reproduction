"""E9 cascade decode (reports/mist_w3po_v1/multispk_s2st_baselines_prereg_v2.zh-CN.md §1 S3, §4): online_decode_finalpost
(B's decode protocol: whole-meeting c2g06 online windows, 1.0 s reads, policy wait, final-read-post close_detect, loop
guard, 8 shards by meeting) with the speaker stream ZEROED, for the v8 speaker-blind Simul models
(eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dlnosd_{en2zh,zh2en}_v8, trained with NOSD=1).

Zeroing is the one used in training (training/thinker_gstar_model.py: use_sd=False -> torch.zeros_like of the speaker
projector's output) and in the v8 overlap study's decode of these models (gstar_si_stream_online_v6s --zero-sd:
sp = torch.zeros_like(sd_proj(...))): Engine.from_merged is wrapped so that the loaded SDProjector returns zeros of its
own output's shape and dtype. The speaker-token POSITIONS stay (interleaved K:1 with the audio, as in training); the
diarizer posteriors therefore reach the model in no way (the final-read-post P extension is a no-op). Every other
argument, the prompt, the read schedule and the output format are online_decode_finalpost's / online_decode's.

    python -m experiments.mist_w3po_v1.evaluation.online_decode_finalpost_zerosd --final-read-post close_detect \
        --thinker <speaker-blind merged Thinker> <online_decode arguments>
"""
from __future__ import annotations

import torch

from experiments.mist_w3po_v1.core import engine as E
from experiments.mist_w3po_v1.evaluation import online_decode_finalpost as fp


class ZeroSD(torch.nn.Module):
    """the speaker projector with its output replaced by zeros (same shape / dtype / device)."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(self.inner(x))


def install():
    """wrap Engine.from_merged (class attribute, shared by online_decode's import); returns the original."""
    orig = E.Engine.from_merged.__func__

    def from_merged(cls, path: str, direction: str, max_gib: int = 40):
        eng = orig(cls, path, direction, max_gib)
        eng.sd_proj = ZeroSD(eng.sd_proj)
        print("ZERO_SD: speaker-stream projector output zeroed (speaker-blind model, NOSD)", flush=True)
        return eng

    E.Engine.from_merged = classmethod(from_merged)
    return orig


def main() -> None:
    install()
    fp.main()


if __name__ == "__main__":
    main()
