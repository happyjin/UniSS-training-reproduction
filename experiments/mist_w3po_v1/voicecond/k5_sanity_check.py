"""Robust K5 end-to-end sanity gate (reports/mist_w3po_v1/robust_k5_softvoice_prereg.zh-CN.md section 4).
Per direction: S = K5-full machinery in sanity mode (b = hard-token level centres, --limit 48, 1 shard, --dump-codes),
C = plain token path (idea1_si_infer_vo + V1 table) on the same subset / shard. Gate: median over items of the paired
TitaNet similarity difference S - C within +-0.003 in both directions. Record: share of items whose semantic codes are
identical in every block; median paired difference S - original V1 run (other shards / batches, record only).
    python -m experiments.mist_w3po_v1.voicecond.k5_sanity_check --root <k5 root> --out <json>
Exit 0 = pass, 3 = fail. Refuses an existing --out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
V1M = REPO / "eval_outputs/mist_w3po_v1/wave3_combos/V/metrics"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out, R = Path(a.out), Path(a.root)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    res, ok = {"gate": "|median(S - C)| <= 0.003 per direction"}, True
    for d in ("en2zh", "zh2en"):
        sim = {k: {r["id"]: r["sim"] for r in jl(R / "metrics" / f"sanity_{k}_{d}_spksim.jsonl")} for k in ("soft", "tok")}
        v1 = {r["id"]: r["sim"] for r in jl(V1M / f"v1_{d}_spksim.jsonl")}
        cod = {k: {r["id"]: [b.get("code_ids") for b in r["blocks"]] for r in jl(R / f"sanity_{k}_{d}" / "results.jsonl")}
               for k in ("soft", "tok")}
        ids = sorted(set(sim["soft"]) & set(sim["tok"]))
        dsc = [sim["soft"][i] - sim["tok"][i] for i in ids if sim["soft"][i] is not None and sim["tok"][i] is not None]
        dsv = [sim["soft"][i] - v1[i] for i in ids if sim["soft"][i] is not None and v1.get(i) is not None]
        same = [cod["soft"][i] == cod["tok"][i] for i in ids]
        md = float(np.median(dsc)) if dsc else float("nan")
        r = {"n": len(ids), "n_paired": len(dsc), "median_diff_soft_minus_tok": md,
             "max_abs_diff_soft_minus_tok": float(np.max(np.abs(dsc))) if dsc else None,
             "codes_identical_share": float(np.mean(same)) if same else None,
             "median_diff_soft_minus_v1_orig": float(np.median(dsv)) if dsv else None,
             "pass": bool(dsc) and abs(md) <= 0.003}
        ok &= r["pass"]
        res[d] = r
    res["pass"] = bool(ok)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
