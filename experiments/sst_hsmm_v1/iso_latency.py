"""sst_hsmm_v1 prereg change 15: iso-latency control (report only). N (test_robust UE_mbr/mbr) with a start-up playout
buffer delta: every block arrival a_i -> max(a_i, a_0 + delta) (the player starts delta later, later blocks unchanged), dip
player, playout.metrics. Per cell (p3_score cells / exclusions) mean D2 and NP_in for delta in DELTAS; for each HSMM system
(p3f P5', p3h P6b, p3i P7, p3j P7' if present) the delta where N's mean D2 equals the system's (linear interpolation) and N's
NP_in there vs the system's NP_in. Refuses to overwrite.
    qwen3omni PYTHONPATH=. python -m experiments.sst_hsmm_v1.iso_latency --out eval_outputs/sst_hsmm_v1/iso_latency.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

from experiments.sst_hsmm_v1 import p3_score as S

REPO = Path(__file__).resolve().parents[2]
DELTAS = [0, 250, 500, 750, 1000, 1250, 1500, 2000, 2500]
LANGS = {"en2zh": "cmn", "zh2en": "eng"}


def _one(args):
    r, lang = args
    from experiments.voice_onset_v1 import playout as PO
    it = PO.Item(r, lang)
    a0 = [p["a"] for p in it.P]
    out = {}
    for dl in DELTAS:
        A = [max(a, a0[0] + dl) for a in a0] if a0 else []
        for p, a in zip(it.P, A):
            p["a"] = a
        it.A = A
        m = PO.metrics(it, PO.simulate(it, None, dip=True))
        out[dl] = (m["D2"], m["NP_in"]) if m and m.get("D2") is not None else None
    return r["id"], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=48)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"拒绝覆盖已有输出: {a.out}")
    E = REPO / "eval_outputs/sst_hsmm_v1"
    systems = {k: v for k, v in (("P5'", "p3f"), ("P6b", "p3h"), ("P7", "p3i"), ("P7'", "p3j")) if (E / v / "summary.json").exists()}
    res = {}
    for d in S.DIRS:
        meta = {m["id"]: m for m in S.jl(S.T / "data" / f"meta_{d}.jsonl")}
        rows = [json.loads(l) for l in open(S.T / "UE_mbr/mbr" / f"talker_{d}" / "results.jsonl", encoding="utf-8") if l.strip()]
        rows = [r for r in rows if not r.get("error")]
        with mp.get_context("fork").Pool(a.procs) as pool:
            per = dict(pool.imap(_one, [(r, LANGS[d]) for r in rows], chunksize=8))
        for cell in S.CELLS:
            ids = sorted(i for i, v in per.items() if v and i in meta and f"{meta[i]['set']}/{meta[i]['cond']}" == cell
                         and not (meta[i]["set"] == "obsi" and meta[i]["pair"] in S.EXCL[d]) and i not in S.DROP
                         and all(v[x] is not None for x in DELTAS))
            curve = [(dl, sum(per[i][dl][0] for i in ids) / len(ids) / 1000.0, sum(per[i][dl][1] for i in ids) / len(ids)) for dl in DELTAS]
            c = {"n": len(ids), "curve": curve, "match": {}}
            for name, root in systems.items():
                sm = json.load(open(E / root / "summary.json"))["cells"][f"{d}/{cell}"]["NEW_vs_N"]
                d2s, nps = sm["d2"]["arm_mean"], sm["NP_in"]["arm"]
                m = None
                for (x0, y0, n0), (x1, y1, n1) in zip(curve, curve[1:]):
                    if y0 <= d2s <= y1 and y1 > y0:
                        w = (d2s - y0) / (y1 - y0)
                        m = {"delta_ms": x0 + w * (x1 - x0), "N_NP_in": n0 + w * (n1 - n0)}
                        break
                c["match"][name] = {"sys_D2": d2s, "sys_NP_in": nps, **(m or {"delta_ms": None, "N_NP_in": None})}
            res[f"{d}/{cell}"] = c
            print(d, cell, len(ids), [(x, round(y, 3), round(n, 3)) for x, y, n in curve[:5]],
                  {k: (round(v["sys_NP_in"], 3), None if v["N_NP_in"] is None else round(v["N_NP_in"], 3)) for k, v in c["match"].items()}, flush=True)
    with open(a.out, "x") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
