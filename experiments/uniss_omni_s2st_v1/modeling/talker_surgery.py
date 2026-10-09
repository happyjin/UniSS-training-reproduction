"""Point Omni's Talker at this project's BiCodec codebook.

The shapes already fit, which was not a given. Omni's Talker carries a
codec vocabulary of 8,448 whose special tokens sit at 8,292 / 8,293 / 8,294
/ 8,296, so slots 0..8,191 are exactly the codec codes -- and this project's
BiCodec semantic codebook is 8,192. Nothing is resized, nothing is appended,
and the four special ids keep their meanings.

What does change is what the slots *mean*. Omni's row k encodes its own
codec's code k, which has no relationship to BiCodec's code k, so those rows
have to be re-drawn. Only two tensors are touched:

    model.embed_tokens.weight  [8448, 2048]   codec embedding (Thinker's dim)
    codec_head.weight          [8448,  896]   output head

Every transformer layer is left alone, and so is ``thinker_to_talker_proj``
[896, 2048] -- the cross-conditioning pathway the plan listed as a component
to build turns out to ship pretrained, because the Talker already consumes
Thinker hidden states this way.

Rows are re-drawn at the scale of the *code* rows specifically, not of the
whole tensor. The two populations differ by an order of magnitude --
measured on this checkpoint, embed_tokens code rows have element-wise std
0.1096 against the reserved rows' 0.0102, and codec_head 0.0204 against
0.0109 -- so a whole-tensor statistic is neither.

``head_scale`` and ``embed_scale`` then multiply those draws. Both exist
because the right answer is not obvious and is better measured than argued.

For the head: drawn at the pretrained scale it is as *confident* as a
trained head while being random, which is a worse starting point than a
flat one (``reports/uniss_omni_s2st_v1/stage0/``).

For the embedding: the Talker sums the code embedding with the Thinker
hidden state *before* projecting, and those hidden states have row norm
173 against the code rows' 4.78. At Omni's own scale the code a step was
given survives that sum at a ratio of 0.028, and measurably so -- see
``reports/uniss_omni_s2st_v1/stage1/CONDITIONING_SCALE.zh-CN.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

BICODEC_SEMANTIC_SIZE = 8192


@dataclass(frozen=True)
class CodecLayout:
    """Which slots are codes and which are special, read from the config."""

    vocab_size: int
    code_size: int
    special_ids: dict[str, int] = field(default_factory=dict)

    @property
    def reserved_start(self) -> int:
        return self.code_size


def read_codec_layout(talker_config, *, code_size: int = BICODEC_SEMANTIC_SIZE) -> CodecLayout:
    specials = {
        name: int(getattr(talker_config, attr))
        for name, attr in (
            ("pad", "tts_codec_pad_token_id"),
            ("bos", "tts_codec_start_token_id"),
            ("eos", "tts_codec_end_token_id"),
            ("mask", "tts_codec_mask_token_id"),
        )
        if getattr(talker_config, attr, None) is not None
    }
    vocab_size = int(talker_config.vocab_size)
    if code_size > vocab_size:
        raise ValueError(f"codebook {code_size} exceeds talker vocab {vocab_size}")
    clashing = {name: i for name, i in specials.items() if i < code_size}
    if clashing:
        # A special id inside the code range would make that code
        # unreachable and silently truncate the codebook.
        raise ValueError(
            f"special tokens {clashing} fall inside the first {code_size} code slots"
        )
    return CodecLayout(vocab_size=vocab_size, code_size=code_size, special_ids=specials)


def _redraw_(
    weight: torch.Tensor,
    rows: int,
    generator: torch.Generator,
    *,
    scale: float = 1.0,
) -> dict:
    """Re-draw ``rows`` leading rows at those rows' own scale, in place."""
    with torch.no_grad():
        reference = weight[:rows].to(torch.float32)
        mean = reference.mean().item()
        std = reference.std().item() * scale
        if std <= 0.0:
            weight[:rows] = 0.0
            return {"mean": 0.0, "std": 0.0, "rows": rows, "scale": scale}
        sample = torch.empty(
            (rows, weight.shape[1]), dtype=torch.float32, device="cpu"
        ).normal_(mean=mean, std=std, generator=generator)
        weight[:rows] = sample.to(dtype=weight.dtype, device=weight.device)
    return {"mean": mean, "std": std, "rows": rows, "scale": scale}


def retarget_talker_codebook(
    talker: nn.Module,
    *,
    code_size: int = BICODEC_SEMANTIC_SIZE,
    seed: int = 20261009,
    head_scale: float = 1.0,
    embed_scale: float = 1.0,
) -> dict:
    """Re-draw the code rows of the Talker's embedding and output head."""
    layout = read_codec_layout(talker.config, code_size=code_size)

    embedding = talker.get_input_embeddings()
    head = talker.codec_head
    for name, tensor, expected in (
        ("embed_tokens", embedding.weight, layout.vocab_size),
        ("codec_head", head.weight, layout.vocab_size),
    ):
        if tensor.shape[0] != expected:
            raise ValueError(
                f"{name} has {tensor.shape[0]} rows, expected {expected};"
                " the checkpoint does not match its config"
            )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    return {
        "layout": layout,
        "embed_tokens": _redraw_(
            embedding.weight, layout.code_size, generator, scale=embed_scale
        ),
        "codec_head": _redraw_(
            head.weight, layout.code_size, generator, scale=head_scale
        ),
        "untouched_rows": layout.vocab_size - layout.code_size,
    }


def valid_output_mask(
    layout: CodecLayout,
    *,
    allow: tuple[str, ...] = ("eos",),
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Boolean mask over the Talker's 8,448 slots: what may ever be emitted.

    Of the 256 reserved rows only eos is reachable in this setup; the other
    255 are Omni's own codec bookkeeping, which no BiCodec target contains
    and no vocoder of ours has an entry for.

    Measured at Stage 0 with freshly drawn code rows, those 255 rows hold
    0.36% of the mass, and masking them moves cross-entropy by 0.004 nats.
    So this is not a quality fix and should not be sold as one -- the 19.4%
    that *does* sit outside the code range is eos, which is legitimate and
    stays. What the mask buys is the guarantee: decoding cannot emit a token
    that has no acoustic meaning, however the head drifts during training.
    """
    mask = torch.zeros(layout.vocab_size, dtype=torch.bool, device=device)
    mask[: layout.code_size] = True
    for name in allow:
        token = layout.special_ids.get(name)
        if token is None:
            raise KeyError(f"no special token called {name!r} in {layout.special_ids}")
        mask[token] = True
    return mask


def apply_output_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Drive the masked slots to -inf, in a dtype-safe way."""
    if logits.shape[-1] != mask.shape[-1]:
        raise ValueError(
            f"logits have {logits.shape[-1]} slots, mask has {mask.shape[-1]}"
        )
    return logits.masked_fill(~mask.to(logits.device), torch.finfo(logits.dtype).min)
