"""Robust wave 11 (reports/mist_w3po_v1/robust_wave11_laplace_prereg.zh-CN.md): clean-anchored Laplace posterior merge
(Fisher-weighted WiSE) of the base Thinker W0 and ERM's W_e = W0 + Delta into a full Qwen3-Omni copy that
si_thinker_stream's vLLM serves (--thinker).

For every adapted matrix (q/k/v/o of every text layer; key thinker.model.<name>.weight):
  Delta  = (alpha / rank) * b @ a                       (ERM adapters_final.pt, fp32)
  Lambda = F_n / (F_c + F_n) elementwise; 0.5 where F_c + F_n = 0 (falls back to WiSE)
  W*     = W0 + Lambda * Delta, in fp32, cast back to bf16 (as wave5_m_merge.py)
F_c, F_n: wave11_fisher.py outputs (F_c at W0 on clean, F_n at W_e on ERM's training mixture, N = 512 each).
Files without an adapted weight are hard-linked from the release (never modified); shards holding one are rewritten.
Refuses to overwrite. Auxiliary statistics (no gate) go to <out>.laplace_merge.json next to the model directory:
Lambda per layer / per projection (mean), global quantiles (10^4-bin histogram on [0, 1]), effective step
||Lambda*Delta|| / ||Delta|| (global, per layer, per projection), Pearson corr(Lambda, |Delta|) (global, exact streaming
sums) and Spearman corr on a fixed subsample (2 * 10^4 elements per matrix, torch.Generator seed 0).
    <qwen3omni python> experiments/mist_w3po_v1/voicecond/wave11_laplace_merge.py --fc <F_c.pt> --fn <F_n.pt> \
        --adapters <run_erm>/adapters_final.pt --out <dir>
    <qwen3omni python> experiments/mist_w3po_v1/voicecond/wave11_laplace_merge.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"
NBIN, NSUB = 10000, 20000
QS = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


def deltas(path: str) -> dict:
    meta = json.load(open(Path(path).with_suffix(".json")))
    assert meta["arm"] == "erm", meta
    scale = meta["alpha"] / meta["rank"]
    st = torch.load(path, map_location="cpu")
    return {f"thinker.model.{k[:-2]}.weight": scale * (st[k[:-2] + ".b"].float() @ st[k].float())
            for k in st if k.endswith(".a")}


def laplace_lambda(fc: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    s = fc + fn
    pos = s > 0
    return torch.where(pos, fn / torch.where(pos, s, torch.ones_like(s)), torch.full_like(s, 0.5))


def rankdata(x: torch.Tensor) -> torch.Tensor:
    """average ranks (ties averaged), float64"""
    v, inv, cnt = torch.unique(x, return_inverse=True, return_counts=True)
    csum = torch.cumsum(cnt, 0).double()
    avg = csum - (cnt.double() - 1) / 2
    return avg[inv]


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double() - x.double().mean(), y.double() - y.double().mean()
    return float((x * y).sum() / ((x * x).sum().sqrt() * (y * y).sum().sqrt()).clamp(min=1e-300))


def selftest() -> None:
    g = torch.Generator().manual_seed(1)
    fc = torch.rand(64, 32, generator=g) * 1e-6
    lam = laplace_lambda(fc, fc.clone())
    assert torch.equal(lam, torch.full_like(lam, 0.5)), "F_c = F_n must give Lambda = 0.5"
    z = torch.zeros(4, 4)
    assert torch.equal(laplace_lambda(z, z), torch.full_like(z, 0.5)), "F_c + F_n = 0 must give 0.5"
    fn = torch.tensor([0.0, 1.0, 3.0, 2.0])
    fc2 = torch.tensor([1.0, 0.0, 1.0, 6.0])
    assert torch.allclose(laplace_lambda(fc2, fn), torch.tensor([0.0, 1.0, 0.75, 0.25])), "hand-computed Lambda"
    w0 = torch.randn(64, 32, generator=g).to(torch.bfloat16)
    d = torch.randn(64, 32, generator=g) * 1e-2
    a = (w0.float() + lam * d).to(torch.bfloat16)
    b = (w0.float() + 0.5 * d).to(torch.bfloat16)
    assert torch.equal(a, b), "F_c = F_n merge must equal WiSE 0.5"
    r = rankdata(torch.tensor([3.0, 1.0, 1.0, 2.0]))
    assert torch.allclose(r, torch.tensor([4.0, 1.5, 1.5, 3.0], dtype=torch.float64))
    print(json.dumps({"selftest": "ok", "lambda_equal_fisher": 0.5, "lambda_zero_fisher": 0.5,
                      "merge_equals_wise05": True}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fc")
    ap.add_argument("--fn")
    ap.add_argument("--adapters")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    out = Path(a.out)
    stats_p = out.parent / f"{out.name}.laplace_merge.json"
    for p in (out, stats_p):
        if p.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {p}")
    for p, w in ((a.fc, "c"), (a.fn, "n")):
        m = json.load(open(Path(p).with_suffix(".json")))
        assert m["which"] == w and m["N"] == 512, (p, m)
    D = deltas(a.adapters)
    Fc, Fn = torch.load(a.fc, map_location="cpu"), torch.load(a.fn, map_location="cpu")
    assert set(D) == set(Fc) == set(Fn), "adapted matrices differ"
    gen = torch.Generator().manual_seed(0)
    hist = torch.zeros(NBIN, dtype=torch.float64)
    per_layer, per_proj = {}, {}
    sums = dict(n=0, l=0.0, ad=0.0, ll=0.0, dd=0.0, ld=0.0, num=0.0, den=0.0, zero=0, l0=0, l1=0, fc=0.0, fn=0.0)
    sub_l, sub_d = [], []
    delta = {}
    for key in sorted(D, key=lambda k: (int(re.search(r"layers\.(\d+)\.", k).group(1)), k)):
        d, fc, fn = D[key], Fc[key].float(), Fn[key].float()
        assert d.shape == fc.shape == fn.shape, key
        lam = laplace_lambda(fc, fn)
        step = lam * d
        delta[key] = step
        li = int(re.search(r"layers\.(\d+)\.", key).group(1))
        pj = re.search(r"\.(\w_proj)\.weight$", key).group(1)
        ad = d.abs()
        n = lam.numel()
        sl, num, den = float(lam.double().sum()), float(step.double().pow(2).sum()), float(d.double().pow(2).sum())
        for dic, k in ((per_layer, li), (per_proj, pj)):
            r = dic.setdefault(k, {"n": 0, "lam_sum": 0.0, "num": 0.0, "den": 0.0})
            r["n"] += n
            r["lam_sum"] += sl
            r["num"] += num
            r["den"] += den
        ld, add = lam.double(), ad.double()
        sums["n"] += n
        sums["l"] += sl
        sums["ad"] += float(add.sum())
        sums["ll"] += float((ld * ld).sum())
        sums["dd"] += float((add * add).sum())
        sums["ld"] += float((ld * add).sum())
        sums["num"] += num
        sums["den"] += den
        sums["zero"] += int(((fc + fn) == 0).sum())
        sums["l0"] += int((lam == 0).sum())
        sums["l1"] += int((lam == 1).sum())
        sums["fc"] += float(fc.double().sum())
        sums["fn"] += float(fn.double().sum())
        hist += torch.histc(lam.float(), bins=NBIN, min=0.0, max=1.0).double()
        idx = torch.randperm(n, generator=gen)[:NSUB]
        sub_l.append(lam.reshape(-1)[idx])
        sub_d.append(ad.reshape(-1)[idx])
    del D, Fc, Fn
    N_ = sums["n"]
    mean_l, mean_d = sums["l"] / N_, sums["ad"] / N_
    cov = sums["ld"] / N_ - mean_l * mean_d
    var_l, var_d = sums["ll"] / N_ - mean_l ** 2, sums["dd"] / N_ - mean_d ** 2
    cdf = torch.cumsum(hist, 0) / hist.sum()
    qv = {f"q{int(q * 100):02d}": round(float((torch.searchsorted(cdf, torch.tensor(q, dtype=torch.float64)) + 1) / NBIN), 4)
          for q in QS}
    SL, SD = torch.cat(sub_l), torch.cat(sub_d)
    stats = {
        "method": "laplace (Fisher-weighted WiSE), Lambda = F_n / (F_c + F_n), 0.5 where F_c + F_n = 0",
        "fc": a.fc, "fn": a.fn, "adapters": a.adapters, "elements": N_,
        "fisher_sum_zero_frac": sums["zero"] / N_, "lambda_eq0_frac": sums["l0"] / N_, "lambda_eq1_frac": sums["l1"] / N_,
        "mean_Fc": sums["fc"] / N_, "mean_Fn": sums["fn"] / N_,
        "lambda_mean": mean_l, "lambda_quantiles_(hist_1e-4)": qv,
        "effective_step_||lam*d||/||d||": (sums["num"] / sums["den"]) ** 0.5,
        "pearson_lambda_absdelta": cov / (var_l * var_d) ** 0.5,
        "spearman_lambda_absdelta_subsample": pearson(rankdata(SL), rankdata(SD)),
        "spearman_subsample_n": int(SL.numel()),
        "per_proj": {k: {"lambda_mean": v["lam_sum"] / v["n"], "eff_step": (v["num"] / v["den"]) ** 0.5}
                     for k, v in sorted(per_proj.items())},
        "per_layer": {str(k): {"lambda_mean": round(v["lam_sum"] / v["n"], 5), "eff_step": round((v["num"] / v["den"]) ** 0.5, 5)}
                      for k, v in sorted(per_layer.items())},
    }
    index = json.loads((BASE / "model.safetensors.index.json").read_text())["weight_map"]
    miss = [k for k in delta if k not in index]
    assert not miss, miss[:3]
    shards = {index[k] for k in delta}
    out.mkdir(parents=True)
    for f in BASE.iterdir():
        if f.is_file() and f.name not in shards:
            try:
                os.link(f, out / f.name)
            except OSError:
                shutil.copy2(f, out / f.name)
    n = 0
    for sh in sorted(shards):
        state = load_file(str(BASE / sh))
        for k, d in delta.items():
            if index[k] == sh:
                state[k] = (state[k].float() + d).to(state[k].dtype).contiguous()
                n += 1
        save_file(state, str(out / sh), metadata={"format": "pt"})
    with open(stats_p, "x") as fh:
        json.dump(stats, fh, indent=1)
    norms = [float(d.norm()) for d in delta.values()]
    print(json.dumps({"out": str(out), "adapted": n, "shards_rewritten": len(shards),
                      "delta_norm_mean": round(sum(norms) / len(norms), 5), "lambda_mean": round(mean_l, 5),
                      "lambda_quantiles": qv, "effective_step": round(stats["effective_step_||lam*d||/||d||"], 5),
                      "pearson_lambda_absdelta": round(stats["pearson_lambda_absdelta"], 5),
                      "spearman_lambda_absdelta_subsample": round(stats["spearman_lambda_absdelta_subsample"], 5),
                      "fisher_sum_zero_frac": stats["fisher_sum_zero_frac"]}), flush=True)


if __name__ == "__main__":
    main()
