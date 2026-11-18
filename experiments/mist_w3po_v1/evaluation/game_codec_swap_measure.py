"""Measure the 2x2 codec-swap cells (game_codec_swap.py) with game_muffle_analysis.spectral / squim -> <dir>/measure.jsonl.
    python -m experiments.mist_w3po_v1.evaluation.game_codec_swap_measure <codec_swap/<dir>>
"""
import json, sys, multiprocessing as mp
from pathlib import Path
from experiments.mist_w3po_v1.evaluation.game_muffle_analysis import read16, spectral, squim
def job(r):
    out = {"id": r["id"]}
    for k in ("OO", "OV", "VO", "VV"):
        if k in r:
            x = read16(r[k]); s = spectral(x) or {}; s.pop("ltas", None)
            out[k] = {**s, **(squim(x) or {})}
    return out
if __name__ == "__main__":
    d = Path(sys.argv[1]); o = d / "measure.jsonl"
    if o.exists(): raise SystemExit(f"拒绝覆盖已有输出: {o}")
    L = [json.loads(l) for l in open(d / "index.jsonl")]
    with mp.get_context("fork").Pool(40) as p, open(o, "x") as fh:
        for r in p.imap_unordered(job, L): fh.write(json.dumps(r) + "\n")
    print("done", len(L), o)
