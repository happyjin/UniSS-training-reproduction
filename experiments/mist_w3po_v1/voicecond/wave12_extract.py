"""Robust wave 12 (reports/mist_w3po_v1/robust_wave12_overfit_diag_prereg.zh-CN.md): take the adapters out of ERM's
resume checkpoints ({step, adapters, opt, ema, ema_n}; the optimizer state is dropped) and write them in the format of
adapters_final.pt / .json (rank 16, alpha 32 = full weight, alpha = 1), so wave5_m_merge.py merges them unchanged.
Also writes the (a)-only item subsets wave12/data/items_<dir>_a.jsonl (meta set == "a" of wave2_S1). Refuses to overwrite.
    <qwen3omni python> experiments/mist_w3po_v1/voicecond/wave12_extract.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
RUN = REPO / "eval_outputs/mist_w3po_v1/wave5/M/run_erm"
OUT = REPO / "eval_outputs/mist_w3po_v1/wave12"
DATA = REPO / "eval_outputs/mist_w3po_v1/wave2_S1/data"
STEPS = (500, 1000, 1500, 2000)


def main() -> None:
    final = torch.load(RUN / "adapters_final.pt", map_location="cpu")
    meta = json.load(open(RUN / "adapters_final.json"))
    for s in STEPS:
        d = OUT / f"ckpt_{s:05d}"
        if d.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {d}")
        st = torch.load(RUN / f"resume_{s:05d}.pt", map_location="cpu", weights_only=False)
        assert st["step"] == s, (st["step"], s)
        ad = st["adapters"]
        assert set(ad) == set(final) and all(ad[k].shape == final[k].shape for k in final)
        assert all(v.dtype == torch.float32 for v in ad.values())
        d.mkdir(parents=True)
        torch.save(ad, d / "adapters.pt")
        m = dict(meta, steps=s, source=f"run_erm/resume_{s:05d}.pt (adapters only)")
        with open(d / "adapters.json", "x") as fh:
            json.dump(m, fh)
        print(json.dumps({"ckpt": s, "tensors": len(ad), "rank": m["rank"], "alpha": m["alpha"]}), flush=True)
    dd = OUT / "data"
    dd.mkdir(parents=True, exist_ok=True)
    for di in ("en2zh", "zh2en"):
        a_ids = {json.loads(l)["id"] for l in open(DATA / f"meta_{di}.jsonl") if json.loads(l)["set"] == "a"}
        rows = [l for l in open(DATA / f"items_{di}.jsonl") if json.loads(l)["id"] in a_ids]
        assert len(rows) == len(a_ids) == 600, (di, len(rows), len(a_ids))
        with open(dd / f"items_{di}_a.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(rows)
        print(json.dumps({"items": di, "n": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
