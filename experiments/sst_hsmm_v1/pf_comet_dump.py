"""portfolio 2026-11-27 (change 55, COMET auxiliary): dump (src, ref, ASR transcript, text translation) per system x direction x
test cell for COMET scoring (pf_comet_score.py, venvs/comet). Items = those every listed system has a playback D2 for, minus the
p3_score drops / OBSI exclusions (same cells as collect.py). Systems: B_b, N and test roots under eval_outputs/sst_hsmm_v1.
    PYTHONPATH=. qwen3omni/bin/python -m experiments.sst_hsmm_v1.pf_comet_dump --out <jsonl> p3hk p3hk2 p3pfa1 ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mist_w3po_v1.voicecond.combo_score import jl
from experiments.mist_w3po_v1.voicecond.test_score import EXCL
from experiments.sst_hsmm_v1 import p3_score as PS

D0 = PS.REPO / "eval_outputs/sst_hsmm_v1"


def load(name, d, first):
    """B_b / N playback files are replicated in every test root; take them from the first listed root"""
    if name in ("B_b", "N"):
        PS.R = D0 / first
        return PS.load(name, d)
    PS.R = D0 / name
    return PS.load("NEW", d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("roots", nargs="+")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refuse to overwrite {out}")
    names = ["B_b", "N"] + a.roots
    rows = []
    for d in PS.DIRS:
        meta = {m["id"]: m for m in jl(PS.T / "data" / f"meta_{d}.jsonl")}
        it = {r["id"]: r for r in jl(PS.T / "data" / f"items_{d}.jsonl")}
        P = {s: load(s, d, a.roots[0]) for s in names}
        common = set.intersection(*[set(p) for p in P.values()]) - PS.DROP
        for cell in PS.CELLS:
            ids = sorted(i for i in common if f"{meta[i]['set']}/{meta[i]['cond']}" == cell
                         and not (meta[i]["set"] == "obsi" and meta[i]["pair"] in EXCL[d]))
            for s in names:
                for i in ids:
                    rows.append({"sys": s, "dir": d, "cell": cell, "id": i, "src": it[i]["src_text"], "ref": it[i]["translation"],
                                 "asr": P[s][i]["asr"] or "", "text": P[s][i]["text"] or ""})
            print(d, cell, len(ids), flush=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print("rows", len(rows))


if __name__ == "__main__":
    main()
