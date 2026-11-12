"""Test of joint/megatron_to_crf_ckpt.py, CPU only: a synthetic Megatron dist checkpoint (2 layers x q/k/v/o LoRA,
sd_proj, new rows) and a synthetic release base; merge_thinker_gstar.py's merged weights == the base folded in memory
with the converted checkpoint by crf/decode_crf_ckpt.apply_ckpt (adapted weights, both row blocks, projector) bit for
bit; refuses to overwrite.

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.joint.test_megatron_to_crf_ckpt
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from safetensors.torch import load_file, save_file

from experiments.mist_w3po_v1.crf import decode_crf_ckpt as dc
from experiments.mist_w3po_v1.joint import megatron_to_crf_ckpt as M
from experiments.uniss_qwen3_omni_v1.training import merge_thinker_gstar as MG
from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import NEW0, NEW_END


def main():
    torch.manual_seed(0)
    H, r, V = 16, 4, NEW_END + 3
    base_w = {f"thinker.model.layers.{l}.self_attn.{p}_proj.weight": torch.randn(H, H).to(torch.bfloat16)
              for l in range(2) for p in "qkvo"}
    base_w["thinker.model.embed_tokens.weight"] = torch.randn(V, 2).to(torch.bfloat16)
    base_w["thinker.lm_head.weight"] = torch.randn(V, 2).to(torch.bfloat16)
    ck = {}
    for l in range(2):
        for p in "qkvo":
            s = f"module.thinker.model.layers.{l}.self_attn.{p}_proj"
            ck[s + ".a"], ck[s + ".b"] = torch.randn(r, H) * 0.1, torch.randn(H, r) * 0.1
    ck["module.new_emb"], ck["module.new_head"] = torch.randn(NEW_END - NEW0, 2), torch.randn(NEW_END - NEW0, 2)
    ck["module.sd_proj.conv.weight"], ck["module.sd_proj.conv.bias"] = torch.randn(3, 8, 5), torch.randn(3)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "base").mkdir()
        save_file(base_w, str(d / "base" / "model-00001-of-00001.safetensors"))
        (d / "base" / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {k: "model-00001-of-00001.safetensors" for k in base_w}}))
        dcp.save(ck, checkpoint_id=str(d / "iter_0000250"))
        old = sys.argv
        sys.argv = ["m", "--checkpoint-dir", str(d / "iter_0000250"), "--base", str(d / "base"), "--rank", str(r),
                    "--alpha", "8", "--out", str(d / "merged")]
        try:
            MG.main()
        finally:
            sys.argv = old
        merged = load_file(str(d / "merged" / "model-00001-of-00001.safetensors"))
        M.convert(str(d / "iter_0000250"), str(d / "base"), r, 8.0, str(d / "conv"))

        class Lin(nn.Module):
            def __init__(self, w):
                super().__init__()
                self.weight = nn.Parameter(w.clone(), requires_grad=False)

        class Th(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList()
                for l in range(2):
                    m = nn.Module()
                    m.self_attn = nn.Module()
                    for p in "qkvo":
                        setattr(m.self_attn, f"{p}_proj", Lin(base_w[f"thinker.model.layers.{l}.self_attn.{p}_proj.weight"]))
                    self.model.layers.append(m)
                self.e = Lin(base_w["thinker.model.embed_tokens.weight"])
                self.lm_head = Lin(base_w["thinker.lm_head.weight"])

            def get_input_embeddings(self):
                return self.e

        th = Th()
        sp = nn.Module()
        sp.conv = nn.Conv1d(8, 3, 5)
        info = dc.apply_ckpt(SimpleNamespace(thinker=th, sd_proj=sp), d / "conv")
        assert info["adapters_folded"] == 8, info
        for l in range(2):
            for p in "qkvo":
                k = f"thinker.model.layers.{l}.self_attn.{p}_proj.weight"
                got = getattr(th.model.layers[l].self_attn, f"{p}_proj").weight
                assert torch.equal(got, merged[k]), k
        assert torch.equal(th.e.weight, merged["thinker.model.embed_tokens.weight"])
        assert torch.equal(th.lm_head.weight, merged["thinker.lm_head.weight"])
        sdm = torch.load(d / "merged" / "sd_proj.pt")
        assert all(torch.equal(sp.state_dict()[k], sdm[k]) for k in sdm)
        try:
            M.convert(str(d / "iter_0000250"), str(d / "base"), r, 8.0, str(d / "conv"))
            raise AssertionError("overwrote")
        except SystemExit:
            pass
    print("ALL PASSED (base + converted checkpoint folded in memory == merge_thinker_gstar output, bit for bit)")


if __name__ == "__main__":
    main()
