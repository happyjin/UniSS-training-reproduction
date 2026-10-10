"""Teacher-forced assembly of Thinker -> Talker, for Stage 0 and training.

This reproduces the construction Omni's own ``generate`` performs
incrementally, but laid out in full so a whole target sequence can be scored
in one forward pass.

Two facts from the Talker's ``forward`` drive everything here:

    inputs_embeds   = embed_tokens(codec_ids) + thinker_reply_part[:, :1, :]
    talker_lm_input = thinker_to_talker_proj(inputs_embeds)

so (a) the conditioning is a *sum in Thinker's 2,048-dim space*, not a cross
attention, and (b) exactly **one text hidden state is consumed per codec
step**. Targets have far more codes than text tokens -- a four-character
Chinese reply against fifty-odd codes is typical -- so the text stream is run
out with an eos embedding followed by pad embeddings, which is what
``generate`` does one step at a time.

``thinker_reply_part`` is the last-layer hidden state *plus* the token's own
input embedding, not the hidden state alone. That is Omni's construction, and
a model conditioned on only one of the two terms would train against a signal
the pretrained projection was never fitted to.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TalkerBatch:
    """Everything the Talker needs for one teacher-forced forward pass."""

    codec_input_ids: torch.Tensor      # [B, T]   bos + codes[:-1]
    codec_labels: torch.Tensor         # [B, T]   codes, -100 where padded
    thinker_reply_part: torch.Tensor   # [B, T, 2048]
    attention_mask: torch.Tensor       # [B, T]


def build_text_stream(
    reply_hidden: torch.Tensor,
    reply_embeds: torch.Tensor,
    *,
    steps: int,
    eos_embed: torch.Tensor,
    pad_embed: torch.Tensor,
) -> torch.Tensor:
    """One conditioning vector per codec step, run out with eos then pad.

    ``reply_hidden`` and ``reply_embeds`` are [B, M, D] over the gold reply
    tokens; the result is [B, steps, D].
    """
    if reply_hidden.shape != reply_embeds.shape:
        raise ValueError(
            f"hidden {tuple(reply_hidden.shape)} and embeds"
            f" {tuple(reply_embeds.shape)} must agree"
        )
    batch, length, dim = reply_hidden.shape
    stream = reply_hidden + reply_embeds
    if steps <= length:
        # A target shorter than its own text is not a real case, but
        # truncating silently would hide whichever bug produced it.
        return stream[:, :steps, :]

    tail = [eos_embed.expand(batch, 1, dim)]
    remaining = steps - length - 1
    if remaining > 0:
        tail.append(pad_embed.expand(batch, remaining, dim))
    return torch.cat([stream] + tail, dim=1)[:, :steps, :]


def build_codec_sequence(
    codes: list[torch.Tensor],
    *,
    bos_id: int,
    pad_id: int,
    eos_id: int | None = None,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad a batch of code sequences into inputs, labels and a mask.

    The Talker is trained to emit ``codes`` given ``bos + codes[:-1]``, so
    inputs and labels are the same length and already aligned: position t
    predicts ``codes[t]``. No further shifting belongs downstream.

    Pass ``eos_id`` to append a stop target. Training without it produces a
    Talker that never learns to stop, which matters more here than it looks:
    a four-character Chinese reply carries fifty-odd codes, so the text
    stream is exhausted long before the codes are, and the only thing that
    can tell the model where the utterance ends is the eos label itself.
    It is optional because Stage 0 scores gold code sequences, where an
    appended stop target would score a step the data does not describe.
    """
    if not codes:
        raise ValueError("no sequences")
    extra = 1 if eos_id is not None else 0
    steps = max(int(c.numel()) for c in codes) + extra
    batch = len(codes)
    device = codes[0].device

    inputs = torch.full((batch, steps), pad_id, dtype=torch.long, device=device)
    labels = torch.full((batch, steps), ignore_index, dtype=torch.long, device=device)
    mask = torch.zeros((batch, steps), dtype=torch.long, device=device)

    for row, sequence in enumerate(codes):
        length = int(sequence.numel())
        inputs[row, 0] = bos_id
        if length > 1:
            inputs[row, 1:length] = sequence[:-1]
        labels[row, :length] = sequence
        mask[row, :length] = 1
        if eos_id is not None:
            inputs[row, length] = int(sequence[-1])
            labels[row, length] = eos_id
            mask[row, length] = 1
    return inputs, labels, mask


def prefix_labels(labels: torch.Tensor, width: int, ignore_index: int = -100) -> torch.Tensor:
    """Pad labels on the left for a conditioning prefix that is not predicted."""
    if width <= 0:
        return labels
    pad = torch.full(
        (labels.shape[0], width), ignore_index, dtype=labels.dtype, device=labels.device
    )
    return torch.cat([pad, labels], dim=1)


def prefix_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
    """Extend an attention mask to cover a conditioning prefix."""
    if width <= 0:
        return mask
    ones = torch.ones(
        (mask.shape[0], width), dtype=mask.dtype, device=mask.device
    )
    return torch.cat([ones, mask], dim=1)


def reply_token_span(
    prompt_length: int, reply_length: int
) -> slice:
    """Positions of the gold reply inside a teacher-forced Thinker forward."""
    return slice(prompt_length, prompt_length + reply_length)


def build_text_stream_batched(
    reply_hidden: torch.Tensor,
    reply_embeds: torch.Tensor,
    reply_lengths: torch.Tensor,
    *,
    steps: int,
    eos_embed: torch.Tensor,
    pad_embed: torch.Tensor,
) -> torch.Tensor:
    """``build_text_stream`` for a whole batch, with no host synchronisation.

    The per-row form needs each row's own text length, and reading it as a
    Python int (``int(mask[row].sum())``) synchronises the device. Done
    once per row per micro-batch that drains the pipeline forty-eight times
    a step, which is most of the gap between 62% and full utilisation.

    Here the lengths stay on the device and the layout is expressed as a
    gather plus two selects: positions below a row's length take its text,
    the position at its length takes eos, everything beyond takes pad.
    """
    if reply_hidden.shape != reply_embeds.shape:
        raise ValueError(
            f"hidden {tuple(reply_hidden.shape)} and embeds"
            f" {tuple(reply_embeds.shape)} must agree"
        )
    batch, width, dim = reply_hidden.shape
    stream = reply_hidden + reply_embeds
    device = stream.device

    positions = torch.arange(steps, device=device)
    index = positions.clamp(max=max(0, width - 1)).view(1, steps, 1).expand(batch, steps, dim)
    gathered = stream.gather(1, index)

    lengths = reply_lengths.view(batch, 1).to(device)
    grid = positions.view(1, steps)
    in_text = (grid < lengths).unsqueeze(-1)
    at_eos = (grid == lengths).unsqueeze(-1)
    tail = torch.where(at_eos, eos_embed.to(stream.dtype), pad_embed.to(stream.dtype))
    return torch.where(in_text, gathered, tail)
