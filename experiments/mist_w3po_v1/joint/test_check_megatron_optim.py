"""Test of joint/check_megatron_optim.py, CPU only, on synthetic torch dist checkpoints: a complete optimizer state
passes; no optimizer state, a missing exp_avg_sq, a wrong shape, an all-zero second moment fail.

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.joint.test_check_megatron_optim
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from experiments.mist_w3po_v1.joint import check_megatron_optim as C


def model_ck() -> dict:
    ck = {}
    for p in "qk":
        s = f"module.thinker.model.layers.0.self_attn.{p}_proj"
        ck[s + ".a"], ck[s + ".b"] = torch.randn(4, 16).bfloat16(), torch.randn(16, 4).bfloat16()
    ck["module.new_emb"], ck["module.new_head"] = torch.randn(3, 2).bfloat16(), torch.randn(3, 2).bfloat16()
    ck["module.sd_proj.conv.weight"] = torch.randn(3, 8, 5).bfloat16()
    ck["module.other_frozen_buffer"] = torch.randn(2)
    return ck


def with_optim(ck: dict, drop: str = "", zero_sq: bool = False, bad_shape: str = "") -> dict:
    out = dict(ck)
    for k, v in ck.items():
        if not C.is_model_key(k):
            continue
        for s in ("exp_avg", "exp_avg_sq", "fp32_param"):
            key = f"optimizer.state.{s}.{k}"
            if key == drop:
                continue
            t = torch.rand(v.shape) + 0.1
            if s == "exp_avg_sq" and zero_sq:
                t = torch.zeros(v.shape)
            if key == bad_shape:
                t = torch.rand(7)
            out[key] = t
    return out


def main():
    torch.manual_seed(0)
    ck = model_ck()
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        a = "module.thinker.model.layers.0.self_attn.q_proj.a"
        cases = {
            "ok": (with_optim(ck), True),
            "no_optim": (ck, False),
            "missing_sq": (with_optim(ck, drop=f"optimizer.state.exp_avg_sq.{a}"), False),
            "zero_sq": (with_optim(ck, zero_sq=True), False),
            "bad_shape": (with_optim(ck, bad_shape=f"optimizer.state.exp_avg.{a}"), False),
        }
        for name, (sd, want) in cases.items():
            dcp.save(sd, checkpoint_id=str(d / name))
            r = C.check(str(d / name))
            assert r["ok"] is want, (name, r)
            assert r["model_keys"] == 7, r
            if name == "ok":
                assert r["states"] == ["exp_avg", "exp_avg_sq", "fp32_param"] and r["exp_avg_sq_keys"] == 7, r
                assert r["exp_avg_sq_abs_sum"] > 0 and r["bytes"] > 0, r
            if name == "missing_sq":
                assert r["n_missing"] == 1, r
        try:
            C.main([str(d / "no_optim")])
            raise AssertionError("no exit")
        except SystemExit as e:
            assert e.code == 1
        try:
            C.main([str(d / "ok")])
            raise AssertionError("no exit")
        except SystemExit as e:
            assert e.code == 0
    print("ALL PASSED (complete optimizer state ok; none / missing exp_avg_sq / wrong shape / zero second moment fail)")


if __name__ == "__main__":
    main()
