#!/usr/bin/env python3
"""Anchor the streaming collator against the inference implementation.

`StreamingCollator` emits index tensors that `_build_talker_inputs_embeds_scatter`
turns into talker input embeddings. Those same embeddings are built explicitly,
position by position, inside `generate_streaming_semantic_codes`
(`src/inference/infer_omni_talker_tts.py:1361-1450`) -- code that demonstrably
works with the released weights.

This test rebuilds the inference construction here as an independent reference and
asserts the collator's indices reproduce it elementwise. Correctness is therefore
anchored on a working implementation rather than on a reading of the paper.

Scope note: the reference covers embedding assembly only. Label placement has no
counterpart in the inference path (generation needs no supervision signal) and is
validated separately by `scripts/probe_stage1_overfit.py`.

Run: pytest tests/test_streaming_collator_parity.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]  # custom/：`import simuls2st_omni` 的根
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simuls2st_omni.train.collators.streaming import StreamingCollator, StreamingChunk

HIDDEN = 16
N_CODES = 64
CODEC_PAD, CODEC_BOS, CODEC_EOS, CODEC_MASK = 64, 65, 66, 67
TEXT_BOS, TEXT_EOS, TEXT_PAD = 200, 201, 202


class _FakeEmbedding:
    """Deterministic embedding: row i is filled with a value derived from i.

    Distinct rows stay far apart, so an off-by-one in indexing shows up as a
    large numeric difference rather than a subtle one.
    """

    def __init__(self, offset: float):
        self.offset = offset

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1).to(torch.float32)
        out = flat.unsqueeze(-1).expand(-1, HIDDEN) * 0.01 + self.offset
        return out.reshape(*ids.shape, HIDDEN).clone()


def _thinker_cond(total_len: int) -> torch.Tensor:
    """Stand-in for `thinker_hidden + thinker_inputs_embeds`, one row per token."""
    base = torch.arange(total_len, dtype=torch.float32).unsqueeze(-1)
    return (base * 0.1 + 7.0).expand(total_len, HIDDEN).clone().unsqueeze(0)


def _reference_embeds(
    chunks: list[StreamingChunk],
    thinker_cond: torch.Tensor,
    talker_embed,
    thinker_embed,
) -> tuple[list[int], torch.Tensor]:
    """Reimplementation of the inference-side construction.

    Mirrors `infer_omni_talker_tts.py:1366-1430`, with every chunk built in the
    complete `[mask][pad][bos][codes][eos]` form that training requires. (The
    inference path leaves the final chunk's `[pad][bos]` without the codec
    additives because the talker model adds them on its first forward; that
    special case does not apply to teacher forcing.)
    """
    hidden = thinker_cond.shape[-1]
    text_bos_embed = thinker_embed(torch.tensor([TEXT_BOS]))[0]
    text_eos_embed = thinker_embed(torch.tensor([TEXT_EOS]))[0]
    text_pad_embed = thinker_embed(torch.tensor([TEXT_PAD]))[0]
    codec_pad_embed = talker_embed(torch.tensor([CODEC_PAD]))[0]
    codec_bos_embed = talker_embed(torch.tensor([CODEC_BOS]))[0]
    codec_eos_embed = talker_embed(torch.tensor([CODEC_EOS]))[0]

    seg_ids: list[int] = []
    seg_embeds: list[torch.Tensor] = []

    for chunk in chunks:
        t_start, t_end = chunk.thinker_span
        r_start, r_end = chunk.reply_span
        assert t_end > t_start and r_end > r_start

        # mask region: codec_mask ids, embeddings taken straight from thinker_cond
        seg_ids.extend([CODEC_MASK] * (t_end - t_start))
        seg_embeds.append(thinker_cond[0, t_start:t_end])

        codes = torch.as_tensor(chunk.codes, dtype=torch.long).reshape(-1)
        n_codes = int(codes.shape[0])

        seg_ids.append(CODEC_PAD)
        seg_embeds.append((codec_pad_embed + text_bos_embed).unsqueeze(0))

        seg_ids.append(CODEC_BOS)
        seg_embeds.append((codec_bos_embed + thinker_cond[0, r_start]).unsqueeze(0))

        # tail conditioning for the L code slots plus the trailing eos slot
        tail = torch.zeros((n_codes + 1, hidden), dtype=thinker_cond.dtype)
        base_tail_len = r_end - r_start - 1
        for k in range(n_codes + 1):
            if k < base_tail_len:
                tail[k] = thinker_cond[0, r_start + 1 + k]
            elif k == base_tail_len:
                tail[k] = text_eos_embed
            else:
                tail[k] = text_pad_embed

        if n_codes > 0:
            seg_ids.extend(codes.tolist())
            seg_embeds.append(talker_embed(codes) + tail[:n_codes])

        seg_ids.append(CODEC_EOS)
        seg_embeds.append((codec_eos_embed + tail[n_codes]).unsqueeze(0))

    return seg_ids, torch.cat(seg_embeds, dim=0).unsqueeze(0)


def _scatter_from_indices(
    thinker_cond: torch.Tensor,
    talker_input_ids: torch.Tensor,
    scatter_indices: torch.Tensor,
    text_cond_indices: torch.Tensor,
    talker_embed,
    thinker_embed,
) -> torch.Tensor:
    """Port of `_build_talker_inputs_embeds_scatter` with the same sentinels.

    Kept local so the test does not need a 3B checkpoint on the GPU; the sentinel
    contract (-2/-3/-4 -> text_bos/text_eos/text_pad) is read directly from
    `src/train/modeling_omni_talker.py:340-344`.
    """
    dtype = thinker_cond.dtype
    codec_emb = talker_embed(talker_input_ids).to(dtype)
    hidden = thinker_cond.shape[-1]

    valid = scatter_indices >= 0
    gather_idx = scatter_indices.clamp(min=0).unsqueeze(-1).expand(-1, -1, hidden)
    scattered = torch.gather(thinker_cond, dim=1, index=gather_idx)

    text_cond = torch.zeros_like(codec_emb)
    text_mask = text_cond_indices >= 0
    if text_mask.any():
        idx = text_cond_indices.clamp(min=0).unsqueeze(-1).expand(-1, -1, hidden)
        text_cond = torch.where(text_mask.unsqueeze(-1), torch.gather(thinker_cond, 1, idx), text_cond)

    for sentinel, token_id in ((-2, TEXT_BOS), (-3, TEXT_EOS), (-4, TEXT_PAD)):
        mask = text_cond_indices == sentinel
        if mask.any():
            embed = thinker_embed(torch.tensor([token_id]))[0].to(dtype)
            text_cond = torch.where(mask.unsqueeze(-1), embed, text_cond)

    return torch.where(valid.unsqueeze(-1), scattered, codec_emb + text_cond)


def _collator() -> StreamingCollator:
    return StreamingCollator(
        codec_pad=CODEC_PAD,
        codec_bos=CODEC_BOS,
        codec_eos=CODEC_EOS,
        codec_mask=CODEC_MASK,
        text_bos=TEXT_BOS,
        text_eos=TEXT_EOS,
        text_pad=TEXT_PAD,
    )


def _make_sample(chunk_specs: list[tuple[int, int, int, int]]) -> dict:
    """Build a sample from (thinker_len, reply_offset, reply_len, n_codes) specs."""
    chunks: list[StreamingChunk] = []
    cursor = 0
    code_value = 3
    for thinker_len, reply_offset, reply_len, n_codes in chunk_specs:
        assert reply_offset + reply_len <= thinker_len
        t_start, t_end = cursor, cursor + thinker_len
        r_start = t_start + reply_offset
        codes = list(range(code_value, code_value + n_codes))
        code_value += n_codes
        chunks.append(
            StreamingChunk(
                thinker_span=(t_start, t_end),
                reply_span=(r_start, r_start + reply_len),
                codes=codes,
            )
        )
        cursor = t_end
    return {
        "id": "probe",
        "task_type": "streaming_s2s_translate",
        "thinker_token_ids": list(range(cursor)),
        "chunks": chunks,
    }


CASES = {
    # (thinker_len, reply_offset, reply_len, n_codes) per chunk
    "two_chunks": [(6, 3, 3, 4), (5, 2, 3, 5)],
    "idle_chunk_L0": [(6, 3, 3, 4), (5, 2, 3, 0), (4, 1, 3, 3)],
    "codes_exceed_reply": [(6, 2, 2, 9)],
    "codes_shorter_than_reply": [(9, 2, 6, 2)],
    "single_token_reply": [(5, 4, 1, 4)],
    "all_idle": [(5, 2, 3, 0), (4, 1, 3, 0)],
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_streaming_collator_matches_inference_reference(case: str) -> None:
    sample = _make_sample(CASES[case])
    total_thinker = len(sample["thinker_token_ids"])
    cond = _thinker_cond(total_thinker)
    talker_embed = _FakeEmbedding(offset=0.0)
    thinker_embed = _FakeEmbedding(offset=100.0)

    ref_ids, ref_embeds = _reference_embeds(
        sample["chunks"], cond, talker_embed, thinker_embed
    )

    batch = _collator()([sample])
    got_ids = batch["talker_input_ids"][0].tolist()
    valid_len = int(batch["talker_attention_mask"][0].sum())

    assert valid_len == len(ref_ids), (
        f"{case}: talker length {valid_len} != reference {len(ref_ids)}"
    )
    assert got_ids[:valid_len] == ref_ids, f"{case}: talker_input_ids differ"

    got_embeds = _scatter_from_indices(
        cond,
        batch["talker_input_ids"][:, :valid_len],
        batch["talker_thinker_scatter_indices"][:, :valid_len],
        batch["talker_text_cond_indices"][:, :valid_len],
        talker_embed,
        thinker_embed,
    )
    torch.testing.assert_close(got_embeds, ref_embeds, rtol=0, atol=0)


def test_labels_supervise_codes_and_eos_only() -> None:
    """Mask region, pad and bos must never be supervised; codes and eos must be."""
    from simuls2st_omni.train.modeling_dual_head import IGNORE_INDEX

    sample = _make_sample(CASES["idle_chunk_L0"])
    batch = _collator()([sample])
    ids = batch["talker_input_ids"][0]
    labels = batch["talker_labels"][0]
    valid_len = int(batch["talker_attention_mask"][0].sum())

    supervised = (labels[:valid_len] != IGNORE_INDEX).nonzero().flatten().tolist()
    for pos in supervised:
        assert int(ids[pos]) != CODEC_MASK, f"mask position {pos} is supervised"
        assert int(ids[pos]) != CODEC_PAD, f"pad position {pos} is supervised"
        assert int(ids[pos]) != CODEC_BOS, f"bos position {pos} is supervised"
        assert int(labels[pos]) == int(ids[pos]), f"label != input at {pos}"

    # Every chunk contributes exactly one eos, plus one label per code.
    n_eos = sum(1 for c in sample["chunks"])
    n_codes = sum(len(c.codes) for c in sample["chunks"])
    assert len(supervised) == n_codes + n_eos


def test_padding_rows_are_inert() -> None:
    """Shorter rows in a batch must be fully masked and unsupervised."""
    from simuls2st_omni.train.modeling_dual_head import IGNORE_INDEX

    short = _make_sample(CASES["single_token_reply"])
    long = _make_sample(CASES["idle_chunk_L0"])
    batch = _collator()([short, long])

    lens = batch["talker_attention_mask"].sum(dim=1)
    assert lens[0] < lens[1], "expected differing row lengths"
    tail = slice(int(lens[0]), batch["talker_input_ids"].shape[1])
    assert (batch["talker_labels"][0, tail] == IGNORE_INDEX).all()
    assert (batch["talker_thinker_scatter_indices"][0, tail] == -1).all()
    assert (batch["talker_text_cond_indices"][0, tail] == -1).all()
