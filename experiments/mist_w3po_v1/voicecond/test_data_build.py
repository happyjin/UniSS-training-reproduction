"""Robust line, final test: one items / meta file per direction (robust_test_prereg_20261120 section 6.2).
RealSI clean / noisy (data/open_voicecond_v1/realsi_noisy_v1, ids prefixed rsc__ / rsn__, set realsi, pair = original id,
stream = the sentence id (sentence-level bootstrap), rec = recording) + OOD-3 / OBSI single-speaker
(data/open_voicecond_v1/ood_single_v1, ids and meta unchanged). Output eval_outputs/mist_w3po_v1/test_robust/data/
{items,meta}_<dir>.jsonl + build.json. Refuses to overwrite.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RS = REPO / "data/open_voicecond_v1/realsi_noisy_v1"
OS_ = REPO / "data/open_voicecond_v1/ood_single_v1"
OUT = REPO / "eval_outputs/mist_w3po_v1/test_robust/data"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {OUT}")
    OUT.mkdir(parents=True)
    info = {}
    for d in ("en2zh", "zh2en"):
        items, meta = [], []
        for cond, pre in (("clean", "rsc__"), ("noisy", "rsn__")):
            for r in jl(RS / f"items_{cond}_{d}.jsonl"):
                i = pre + r["id"]
                items.append(dict(r, id=i))
                meta.append({"id": i, "set": "realsi", "cond": cond, "pair": r["id"], "stream": i,
                             "rec": r["id"].rsplit("_", 1)[0], "ovl": False, "dur_ms": int(r["dur_ms"])})
        oi, om = jl(OS_ / f"items_{d}.jsonl"), jl(OS_ / f"meta_{d}.jsonl")
        assert [r["id"] for r in oi] == [m["id"] for m in om]
        items += oi
        meta += om
        ids = [r["id"] for r in items]
        assert len(ids) == len(set(ids)), "duplicate ids"
        for r, m in zip(items, meta):
            assert r["id"] == m["id"] and int(r["dur_ms"]) == int(m["dur_ms"]) and Path(r["audio"]).exists() and r["translation"]
        for name, rows in (("items", items), ("meta", meta)):
            with open(OUT / f"{name}_{d}.jsonl", "x", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        info[d] = {"n": len(items), "cells": {f"{k[0]}/{k[1]}": v for k, v in sorted(Counter((m["set"], m["cond"]) for m in meta).items())},
                   "hours": round(sum(int(r["dur_ms"]) for r in items) / 3.6e6, 3)}
    (OUT / "build.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
