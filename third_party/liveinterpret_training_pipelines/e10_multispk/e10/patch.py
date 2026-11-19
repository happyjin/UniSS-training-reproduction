"""E10 runtime patches on the LiveInterpret S5 code (installed by plugin_e10.py; the S5 files are not edited).

Data side (records carry an `e10` block, written by build_li_records.py):
  * `S4StreamingDataset._prepare_record`: the instruction comes from `rec["e10"]["instruction"]` (B's G-STAR rows
    get a multi-speaker instruction, B's single-speaker `sd` rows keep the S5 one);
  * `S4StreamingDataset._build_view`: passes the record's posterior path on;
  * `S4Encoder.encode`: loads the Nemotron posteriors, applies B's perturbation on training rows (p=0.2, N(0, 0.1),
    roll +-1 frame; seeded by the row's s4_seed) and the 13-frame delay (e10.common.delayed_post) -> `spk_post`
    (n_audio_frames, 8); rows without posteriors (sd) get zeros;
  * `collate_s4`: `spk_posts` = list of the per-sample tensors.
Model side:
  * `StreamingSIForConditionalGeneration.forward(spk_posts=...)` keeps them for the call;
  * `SI.encode_audio_inputs` adds `e10_sd_proj(spk_post)` to each sample's projected audio frames (training path;
    the streaming inference path adds the same term itself, see run_e10.py).
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np
import torch

from .common import delayed_post, perturb

_INSTALLED = False


def _read_rec(ds, flat_index):
    fi, li = ds.items[flat_index]
    return ds.indexes[fi].read(li)


def _patch_dataset():
    from streaming_si.data import s4_streaming_dataset as M
    cls = M.S4StreamingDataset
    orig_prepare, orig_view = cls._prepare_record, cls._build_view

    def _prepare_record(self, flat_index, forced_m=None):
        prepared = orig_prepare(self, flat_index, forced_m)
        if prepared is None:
            return None
        e10 = _read_rec(self, flat_index).get("e10") or {}
        text = e10.get("instruction")
        if text:
            key = ("e10", text)
            if key not in self._instr_cache:
                self._instr_cache[key] = self.tokenizer.encode(text, add_special_tokens=False)
            prepared = dataclasses.replace(prepared, instr_ids=list(self._instr_cache[key]))
        return prepared

    def _build_view(self, prepared, kind, live_turn=None, include_view_fields=False):
        item = orig_view(self, prepared, kind, live_turn=live_turn, include_view_fields=include_view_fields)
        if item is None:
            return None
        e10 = _read_rec(self, prepared.flat_index).get("e10")
        if e10 is not None:
            if kind != "full":
                raise RuntimeError("E10 records support only full views (no prefix-drop)")
            item["e10_post"] = e10.get("post") or ""
        return item

    cls._prepare_record, cls._build_view = _prepare_record, _build_view


def _patch_pipeline():
    from streaming_si.data import s4_pipeline as P
    orig_encode, orig_collate = P.S4Encoder.encode, P.collate_s4
    p_perturb = float(os.environ.get("E10_POST_PERTURB", "0.2"))

    def encode(self, row):
        item = orig_encode(self, row)
        if "e10_post" not in item:
            return item
        path = item.pop("e10_post")
        n = int(torch.as_tensor(item["audio_seg_lens"]).sum())
        post = np.load(path).astype(np.float32) if path else None
        if post is not None and row["s4_split"] == "train" and p_perturb > 0:
            post = perturb(post, np.random.default_rng(int(row["s4_seed"]) & 0xFFFFFFFF), p_perturb)
        item["spk_post"] = torch.from_numpy(delayed_post(post, n))
        return item

    def collate_s4(batch):
        posts = [b.pop("spk_post", None) for b in batch]
        out = orig_collate(batch)
        if any(p is not None for p in posts):
            if any(p is None for p in posts):
                raise ValueError("E10: a batch mixes records with and without spk_post")
            out["spk_posts"] = posts
        return out

    P.S4Encoder.encode, P.collate_s4 = encode, collate_s4


def add_sd(model, feats: torch.Tensor, spk_post: torch.Tensor) -> torch.Tensor:
    """feats (n, hidden) + e10_sd_proj(spk_post[:n]) (zero rows past spk_post's end)."""
    sd = getattr(model, "e10_sd_proj", None)
    if sd is None:
        raise RuntimeError("E10: model has no e10_sd_proj (load it through e10.tuner)")
    n = feats.size(0)
    p = spk_post[:n].to(device=feats.device, dtype=torch.float32)
    if p.size(0) < n:
        p = torch.cat([p, p.new_zeros(n - p.size(0), p.size(1))])
    w_dtype = next(sd.parameters()).dtype
    return feats + sd(p.to(w_dtype)).to(feats.dtype)


def _patch_model():
    from streaming_si.model import model_si as S
    from streaming_si.model import modeling_streaming_si as H
    orig_forward = H.StreamingSIForConditionalGeneration.forward
    orig_encode = S.SI.encode_audio_inputs

    def forward(self, *args, spk_posts=None, **kwargs):
        if spk_posts is None:
            return orig_forward(self, *args, **kwargs)
        self.__dict__["_e10_spk"] = spk_posts
        try:
            return orig_forward(self, *args, **kwargs)
        finally:
            self.__dict__.pop("_e10_spk", None)

    def encode_audio_inputs(self, audio_inputs, audio_lens=None):
        feats = orig_encode(self, audio_inputs, audio_lens)
        spk = self.__dict__.get("_e10_spk")
        if spk is None or feats is None:
            return feats
        if len(spk) != len(feats):
            raise RuntimeError(f"E10: {len(spk)} spk_posts for {len(feats)} samples")
        return [f if (f is None or p is None) else add_sd(self, f, p) for f, p in zip(feats, spk)]

    H.StreamingSIForConditionalGeneration.forward = forward
    S.SI.encode_audio_inputs = encode_audio_inputs


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    _patch_dataset()
    _patch_pipeline()
    _patch_model()
    _INSTALLED = True
