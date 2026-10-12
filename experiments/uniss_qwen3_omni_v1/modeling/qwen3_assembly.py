"""One model: Qwen3-Omni's Thinker driving the Talker UniSS trained.

Phase B ran two Thinkers. Qwen3 turned source audio into a text string,
that string was written to a file, and Qwen2.5's Thinker read it back as a
TTS prompt to produce the hidden states the Talker consumes. That works
without any training precisely because the Talker's input distribution
never changed -- and it cannot stream, because the text has to be complete
before the second half begins.

Here there is one Thinker. Qwen3's hidden states reach the Talker
directly, which is what streaming requires and what Phase D has to train
the Talker to read. Qwen3's own Talker and code2wav are deleted, the way
Qwen2.5's token2wav already is: this project's output representation is
UniSS's single-stream BiCodec, not either model's codec.

What carries across unchanged is the dimension. Both Thinkers' text models
have hidden_size 2048, so global_embed, thinker_to_talker_proj and
build_text_stream need no change at all. What does not carry across is
meaning: two models' 2048-dimensional spaces have nothing to do with each
other, which is the whole reason Phase D exists.

**The stream terminators are learned here rather than looked up.**
Qwen2.5-Omni reserves embedding rows 151859 and 151861 as the Talker's
text pad and eos, and build_text_stream ends a reply with one eos then
pad. Those two rows hold the *same vector* in Qwen2.5-Omni -- checked,
torch.equal is True -- so the distinction never carried a signal, and in
Qwen3-Omni the same rows are two unrelated vectors that were never trained
for this role at all (cosine -0.038). Borrowing them would feed the Talker
two arbitrary directions as "the text is finished". They are two vectors
of 2048 numbers whose only job is to mark an ending, so they are
parameters, initialised from Qwen3's rows and fitted in Phase D.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from experiments.uniss_omni_s2st_v1.modeling.talker_surgery import (
    read_codec_layout,
    retarget_talker_codebook,
    valid_output_mask,
)

# Qwen2.5-Omni's reserved rows, kept only to initialise from the same
# positions in Qwen3's table. Nothing downstream looks them up.
TEXT_PAD_ROW = 151_859
TEXT_EOS_ROW = 151_861
THINKER_HIDDEN = 2048


@dataclass
class Assembly:
    thinker: nn.Module
    talker: nn.Module
    global_embed: nn.Embedding
    markers: nn.Module           # holds eos/pad, see StreamMarkers
    processor: object            # Qwen3's, for audio and chat templating
    tokenizer: object
    layout: object
    output_mask: torch.Tensor
    device: torch.device
    dtype: torch.dtype


class StreamMarkers(nn.Module):
    """The two vectors that end a reply's text stream, as parameters."""

    def __init__(self, hidden: int, eos_init=None, pad_init=None):
        super().__init__()
        self.eos = nn.Parameter(
            eos_init.clone().float() if eos_init is not None else torch.zeros(hidden)
        )
        self.pad = nn.Parameter(
            pad_init.clone().float() if pad_init is not None else torch.zeros(hidden)
        )

    def embeds(self, dtype, device):
        return (
            self.eos.to(dtype=dtype, device=device).view(1, 1, -1),
            self.pad.to(dtype=dtype, device=device).view(1, 1, -1),
        )


def build_assembly(
    *,
    qwen3_dir: str = "pretrained_models/Qwen3-Omni-30B-A3B-Instruct",
    qwen25_dir: str = "pretrained_models/Qwen2.5-Omni-3B",
    talker_checkpoint: str | None = None,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    head_scale: float = 0.1,
    attn: str = "sdpa",
    thinker_from: str = "qwen3",
) -> Assembly:
    """``thinker_from='qwen25'`` builds the control arm.

    Everything else is identical -- the same Talker weights, the same
    surgery, the same prefix, the same marker parameters -- so a probe run
    against it measures the Talker on the representation it was trained
    for. Without that line, a near-zero score on the Qwen3 arm cannot be
    told apart from a bug in the probe.
    """
    from transformers import (
        Qwen2_5OmniForConditionalGeneration,
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeProcessor,
    )

    dev = torch.device(device)

    # The Talker comes off CPU and only its own half moves to the GPU, so
    # the 3B's Thinker never shares a card with the 30B's.
    donor = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        qwen25_dir, dtype=dtype
    )
    talker = donor.talker
    layout = read_codec_layout(talker.config)
    retarget_talker_codebook(talker, head_scale=head_scale)
    global_embed = nn.Embedding(4096, THINKER_HIDDEN)
    if talker_checkpoint:
        state = torch.load(talker_checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = talker.load_state_dict(state["talker"], strict=False)
        if state.get("global_embed") is not None:
            with torch.no_grad():
                global_embed.weight.copy_(state["global_embed"])
        else:
            raise SystemExit(
                f"{talker_checkpoint} has no global_embed; the speaker prefix"
                " would be random and nothing downstream would say so"
            )
        # A trained RMSNorm sits near 1; near zero is an optimiser moment
        # exported in its place, which produces a constant-output Talker.
        norm = talker.state_dict().get("model.norm.weight")
        if norm is not None and float(norm.float().abs().mean()) < 0.01:
            raise SystemExit("model.norm.weight is near zero; bad export")
    talker = talker.to(device=dev, dtype=dtype)
    global_embed = global_embed.to(device=dev, dtype=dtype)
    del donor.thinker
    if hasattr(donor, "token2wav"):
        del donor.token2wav
    del donor

    if thinker_from == "qwen25":
        donor2 = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            qwen25_dir, dtype=dtype
        )
        for attr in ("talker", "token2wav"):
            if hasattr(donor2, attr):
                delattr(donor2, attr)
        thinker = donor2.thinker.to(dev)
        del donor2
    else:
        omni3 = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            qwen3_dir, dtype=dtype, attn_implementation=attn
        )
        for attr in ("talker", "code2wav", "token2wav"):
            if hasattr(omni3, attr):
                delattr(omni3, attr)
        thinker = omni3.thinker.to(dev)
        del omni3
    torch.cuda.empty_cache()

    embed = thinker.get_input_embeddings().weight
    if embed.shape[1] != THINKER_HIDDEN:
        raise SystemExit(
            f"Qwen3 thinker hidden is {embed.shape[1]}, not {THINKER_HIDDEN};"
            " the bridge would need redimensioning"
        )
    proj = talker.thinker_to_talker_proj.weight
    if tuple(proj.shape) != (talker.config.hidden_size, THINKER_HIDDEN):
        raise SystemExit(f"thinker_to_talker_proj is {tuple(proj.shape)}, not (896, 2048)")

    markers = StreamMarkers(
        THINKER_HIDDEN,
        eos_init=embed[TEXT_EOS_ROW].detach().cpu(),
        pad_init=embed[TEXT_PAD_ROW].detach().cpu(),
    ).to(dev)

    if thinker_from == "qwen25":
        from experiments.uniss_omni_s2st_v1.runtime.omni_processor import (
            load_audio_text_processor,
        )
        processor = load_audio_text_processor(qwen25_dir)
    else:
        processor = Qwen3OmniMoeProcessor.from_pretrained(qwen3_dir)
    return Assembly(
        thinker=thinker, talker=talker, global_embed=global_embed, markers=markers,
        processor=processor, tokenizer=processor.tokenizer, layout=layout,
        output_mask=valid_output_mask(layout, device=dev), device=dev, dtype=dtype,
    )
