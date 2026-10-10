"""Initialise Omni's Talker from this project's trained UniSS decoder.

Omni's Talker and the UniSS Phase-3 model are the same transformer: 24
layers, hidden 896, 14 heads over 2 KV groups, FFN 4864, RMSNorm eps 1e-6,
rope base 1e6. 289 of the Talker's 293 tensors match by name and shape.

That matters because the UniSS model has already been trained to emit
BiCodec semantic codes -- it is what produced this project's published
AutoPCP and UTMOS -- while Omni's Talker has only ever seen its own codec.
Starting Stage 1 from random code rows asks it to learn from scratch
something a shape-identical model on this disk already knows.

What transfers and what cannot:

    model.layers.*            288 tensors, exact            -> copied
    model.norm.weight                                       -> copied
    codec_head.weight[:8192]  <- UniSS tied head rows
                                 155,761..163,952 (896-dim) -> copied
    model.embed_tokens        Omni [8448, 2048] vs UniSS
                              [180480, 896]                 -> left alone
    thinker_to_talker_proj    Omni-only, 2048 -> 896        -> left alone

The embedding cannot come across because Omni sums the code embedding with
the Thinker hidden state in the Thinker's 2,048-dim space before
projecting, where UniSS fed 896-dim code embeddings straight into the
layers. So the transferred layers arrive expecting UniSS's input
distribution and the input path has to be learned -- a far smaller problem
than learning the layers and the codebook as well.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import torch
from safetensors import safe_open

# From training/constants_uniss.py: the bicodec_semantic block.
UNISS_SEMANTIC_OFFSET = 155_761
UNISS_SEMANTIC_SIZE = 8_192


class TransferError(ValueError):
    """The donor checkpoint does not match the Talker it is meant to seed."""


def read_uniss_tensors(hf_dir: str | Path) -> dict[str, torch.Tensor]:
    """Every tensor in an exported UniSS HF checkpoint."""
    hf_dir = Path(hf_dir)
    shards = sorted(glob.glob(str(hf_dir / "*.safetensors")))
    if not shards:
        raise TransferError(f"no safetensors in {hf_dir}")
    tensors: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(shard, "pt") as handle:
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key)
    return tensors


def check_geometry(hf_dir: str | Path, talker_config) -> dict:
    """Refuse a donor whose transformer is not the Talker's."""
    config = json.loads((Path(hf_dir) / "config.json").read_text(encoding="utf-8"))
    fields = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
    )
    mismatched = {
        name: (config.get(name), getattr(talker_config, name, None))
        for name in fields
        if config.get(name) != getattr(talker_config, name, None)
    }
    if mismatched:
        raise TransferError(
            f"donor geometry differs from the Talker: {mismatched}"
        )
    if not config.get("tie_word_embeddings"):
        # The head is read off the embedding; an untied donor would need its
        # lm_head instead, and silently reading the embedding would seed the
        # head with the wrong matrix.
        raise TransferError(
            "donor does not tie its embeddings; the output head would be wrong"
        )
    return config


def transfer_into_talker(
    talker,
    hf_dir: str | Path,
    *,
    code_size: int = UNISS_SEMANTIC_SIZE,
    copy_head: bool = True,
) -> dict:
    """Copy the layers, the final norm and the code head, in place."""
    check_geometry(hf_dir, talker.config)
    donor = read_uniss_tensors(hf_dir)
    target = dict(talker.named_parameters())
    target.update(dict(talker.named_buffers()))

    copied: list[str] = []
    skipped: dict[str, str] = {}
    with torch.no_grad():
        for name, parameter in target.items():
            if name in ("model.embed_tokens.weight", "codec_head.weight"):
                continue
            if name.startswith("thinker_to_talker_proj"):
                skipped[name] = "Omni-only bridge"
                continue
            source = donor.get(name)
            if source is None:
                skipped[name] = "absent from the donor"
                continue
            if tuple(source.shape) != tuple(parameter.shape):
                skipped[name] = f"shape {tuple(source.shape)} vs {tuple(parameter.shape)}"
                continue
            parameter.copy_(source.to(parameter.dtype))
            copied.append(name)

        head_rows = 0
        if copy_head:
            embedding = donor.get("model.embed_tokens.weight")
            if embedding is None:
                raise TransferError("donor has no embedding to read the head from")
            start, stop = UNISS_SEMANTIC_OFFSET, UNISS_SEMANTIC_OFFSET + code_size
            if embedding.shape[0] < stop:
                raise TransferError(
                    f"donor embedding has {embedding.shape[0]} rows,"
                    f" needs {stop} to cover the semantic block"
                )
            rows = embedding[start:stop]
            head = talker.codec_head.weight
            if rows.shape[1] != head.shape[1]:
                raise TransferError(
                    f"donor rows are {rows.shape[1]}-dim, head is {head.shape[1]}-dim"
                )
            head[:code_size].copy_(rows.to(head.dtype))
            head_rows = code_size

    return {
        "donor": str(hf_dir),
        "copied_tensors": len(copied),
        "head_rows": head_rows,
        "skipped": skipped,
    }
