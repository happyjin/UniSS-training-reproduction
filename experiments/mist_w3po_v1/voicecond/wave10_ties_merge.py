"""Robust wave 10 (reports/mist_w3po_v1/robust_wave10_ties_prereg.zh-CN.md): TIES-merge (Yadav et al., NeurIPS 2023) the
two single-direction LoRAs (wave10_dir_train.py adapters_final.pt of D_zh2en and D_en2zh), then WiSE 0.5, into a full
Qwen3-Omni copy that si_thinker_stream's vLLM serves (--thinker).

For every adapted matrix (q/k/v/o of every text layer; key thinker.model.<name>.weight), with the paper defaults and no
tuning:
  tau_d   = (alpha / rank) * b_d @ a_d                       (fp32, d in {zh2en, en2zh})
  trim    : keep the top k = 20 % of |tau_d| of that matrix (exactly ceil(0.2 n) entries, torch.topk), zero the rest
  elect   : gamma = sign(sum_d trimmed tau_d)
  merge   : tau* = disjoint mean = sum_d [sign(trimmed tau_d) == gamma] trimmed tau_d / max(1, #agreeing d)
  lambda 1; WiSE: W = W0 + wise * tau* with wise = 0.5 (same alpha as ERM-WiSE).
Files without an adapted weight are hard-linked from the release (never modified); shards holding one are rewritten
(as wave5_m_merge.py). Refuses to overwrite. Prints one JSON line with "adapted" (the orchestrator greps it) and
summary statistics of tau*.
    <qwen3omni python> experiments/mist_w3po_v1/voicecond/wave10_ties_merge.py --zh2en <run>/adapters_final.pt \
        --en2zh <run>/adapters_final.pt --out <dir> [--k 0.2] [--lam 1] [--wise 0.5]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"


def taus(path: str) -> dict:
    """adapters_final.pt -> {thinker.model.<stem>.weight: (alpha/rank) b @ a} (fp32)"""
    meta = json.load(open(Path(path).with_suffix(".json")))
    scale = meta["alpha"] / meta["rank"]
    st = torch.load(path, map_location="cpu")
    return {f"thinker.model.{k[:-2]}.weight": scale * (st[k[:-2] + ".b"].float() @ st[k].float())
            for k in st if k.endswith(".a")}


def trim(t: torch.Tensor, k: float) -> torch.Tensor:
    flat = t.reshape(-1)
    n = max(1, math.ceil(k * flat.numel()))
    idx = flat.abs().topk(n, sorted=False).indices
    out = torch.zeros_like(flat)
    out[idx] = flat[idx]
    return out.view_as(t)


def ties(ts: list[torch.Tensor], k: float, lam: float) -> tuple[torch.Tensor, dict]:
    tr = [trim(t, k) for t in ts]
    gamma = torch.sign(sum(tr))
    agree = [(torch.sign(t) == gamma) & (t != 0) for t in tr]
    num = sum(torch.where(m, t, torch.zeros_like(t)) for m, t in zip(agree, tr))
    cnt = sum(m.float() for m in agree)
    merged = lam * num / cnt.clamp(min=1)
    both = (tr[0] != 0) & (tr[1] != 0)
    conflict = both & (torch.sign(tr[0]) != torch.sign(tr[1]))
    stats = {"n": tr[0].numel(), "both_kept": int(both.sum()), "conflict": int(conflict.sum()),
             "nonzero_merged": int((merged != 0).sum())}
    return merged, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zh2en", required=True)
    ap.add_argument("--en2zh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--wise", type=float, default=0.5)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    for p, d in ((a.zh2en, "zh2en"), (a.en2zh, "en2zh")):
        m = json.load(open(Path(p).with_suffix(".json")))
        assert m.get("direction") == d, (p, m)
    T1, T2 = taus(a.zh2en), taus(a.en2zh)
    assert set(T1) == set(T2), "adapted matrices differ"
    delta, agg = {}, {"n": 0, "both_kept": 0, "conflict": 0, "nonzero_merged": 0}
    for key in T1:
        tm, st = ties([T1[key], T2[key]], a.k, a.lam)
        delta[key] = a.wise * tm
        for x in agg:
            agg[x] += st[x]
    del T1, T2
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
    norms = [float(d.norm()) for d in delta.values()]
    json.dump({"method": "ties", "k": a.k, "lam": a.lam, "wise": a.wise, "zh2en": a.zh2en, "en2zh": a.en2zh, **agg},
              open(out.parent / f"{out.name}.ties_merge.json", "x"), indent=1)
    print(json.dumps({"out": str(out), "adapted": n, "shards_rewritten": len(shards), "k": a.k, "lam": a.lam,
                      "wise": a.wise, "delta_norm_mean": round(sum(norms) / len(norms), 5),
                      "both_kept_frac": round(agg["both_kept"] / agg["n"], 5),
                      "conflict_frac_of_both": round(agg["conflict"] / max(1, agg["both_kept"]), 5),
                      "nonzero_merged_frac": round(agg["nonzero_merged"] / agg["n"], 5)}), flush=True)


if __name__ == "__main__":
    main()
