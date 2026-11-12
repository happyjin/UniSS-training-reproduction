"""Test of joint/identity_ckpt.py, CPU only: a synthetic merged directory (2 safetensors shards + index, sd_proj.pt)
-> the checkpoint's rows are the stored rows bit for bit; crf/decode_crf_ckpt.apply_ckpt on a tiny engine-like object
holding those weights leaves every tensor bit-identical (no adapter folded, rows / projector rewritten with equal
values); refuses to overwrite.

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.joint.test_identity_ckpt
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from safetensors.torch import save_file

from experiments.mist_w3po_v1.crf import decode_crf_ckpt as dc
from experiments.mist_w3po_v1.joint import identity_ckpt as I


def main():
    torch.manual_seed(0)
    V, H, n0, ne = 40, 6, 30, 37
    emb, head = torch.randn(V, H).to(torch.bfloat16), torch.randn(V, H).to(torch.bfloat16)
    sd = nn.Sequential(nn.Linear(4, 3))
    with tempfile.TemporaryDirectory() as td:
        b = Path(td) / "merged"
        b.mkdir()
        save_file({I.EMB: emb, "x.w": torch.zeros(2)}, str(b / "model-00001-of-00002.safetensors"))
        save_file({I.HEAD: head}, str(b / "model-00002-of-00002.safetensors"))
        (b / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            I.EMB: "model-00001-of-00002.safetensors", "x.w": "model-00001-of-00002.safetensors",
            I.HEAD: "model-00002-of-00002.safetensors"}}))
        torch.save(sd.state_dict(), b / "sd_proj.pt")
        o = Path(td) / "ck"
        I.build(str(b), str(o), n0, ne)
        ex = torch.load(o / "extra.pt")
        assert torch.equal(ex["new_emb"], emb[n0:ne]) and torch.equal(ex["new_head"], head[n0:ne])
        assert ex["new_emb"].dtype == torch.bfloat16 and torch.load(o / "lora.pt") == {}
        assert (o / "DONE").exists() and ex["lora_rank"] == 32 and ex["objective"] == "identity"

        class Th(nn.Module):
            def __init__(self):
                super().__init__()
                self.e = nn.Embedding(V, H)
                self.lm_head = nn.Linear(H, V, bias=False)
                self.e.weight.data, self.lm_head.weight.data = emb.clone(), head.clone()

            def get_input_embeddings(self):
                return self.e

        th = Th()
        sp = nn.Sequential(nn.Linear(4, 3))
        sp.load_state_dict(torch.load(b / "sd_proj.pt"))
        before = {k: v.clone() for k, v in list(th.state_dict().items()) + [("sp." + k, v) for k, v in sp.state_dict().items()]}
        old0, old1 = dc.NEW0, dc.NEW_END
        dc.NEW0, dc.NEW_END = n0, ne
        try:
            info = dc.apply_ckpt(SimpleNamespace(thinker=th, sd_proj=sp), o)
        finally:
            dc.NEW0, dc.NEW_END = old0, old1
        after = dict(list(th.state_dict().items()) + [("sp." + k, v) for k, v in sp.state_dict().items()])
        assert info["adapters_folded"] == 0
        for k in before:
            assert torch.equal(before[k], after[k]), k
        try:
            I.build(str(b), str(o), n0, ne)
            raise AssertionError("overwrote")
        except SystemExit:
            pass
    print("ALL PASSED (identity checkpoint: rows bit for bit, apply_ckpt leaves the model unchanged)")


if __name__ == "__main__":
    main()
