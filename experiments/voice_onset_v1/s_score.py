"""voice_onset_v1 S0 / S1 / S2 scoring (reports/voice_onset_v1/S0_S1_S2_prereg.zh-CN.md sections 2-5 + change 1).
Base talker systems (p3_score conventions): B_b, N (= UE_mbr), NS1 (= UE_mbrq, queue talker). A played system
<base>+<policy> shares the base's ASR / PESQ / similarity rows (speech-only wav, unchanged by construction) and takes D1 / D2
from eval_outputs/voice_onset_v1/s0/<base>_<policy>/<d>.jsonl. Cells as P3 (health_003 dropped in all systems, OBSI en2zh
without e031_0004). Per pair: wave3_v_score.compare (stream bootstrap 2000, seed 0) + the playout metrics NP_in / PT_in /
PT_all / NC / EndOffset (paired mean diff, same stream bootstrap).
    qwen3omni PYTHONPATH=. python -m experiments.voice_onset_v1.s_score --stage s0|s1|all --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.voicecond.combo_score import LANG, clean, jl, laal_defs, load
from experiments.mist_w3po_v1.voicecond.test_score import EXCL
from experiments.mist_w3po_v1.voicecond.wave3_v_score import compare
from experiments.voice_onset_v1 import p3_score as P3

REPO = Path(__file__).resolve().parents[2]
T = P3.T
S0 = REPO / "eval_outputs/voice_onset_v1/s0"
DIRS, CELLS, NOISY, DROP = P3.DIRS, P3.CELLS, P3.NOISY, P3.DROP
BASE = {"B_b": "B_b", "N": "UE_mbr", "NS1": "NS1"}
P3.SYS["NS1"] = ("UE_mbrq", "mbrq", "UE_mbrq/metrics/ecapa2_mbrq_{d}_spksim.jsonl")
PM = ("NP_in", "PT_in", "PT_all", "NC", "n_pause", "EndOffset")
STAGES = {
    "s0": (["B_b", "B_b+r0", "N", "N+r0"], [("N+r0", "N"), ("B_b+r0", "B_b")]),
    "s1": (["B_b", "N", "N+r0", "NS1", "NS1+r0"], [("NS1", "N"), ("NS1+r0", "N+r0"), ("NS1", "B_b")]),
    "all": (["B_b", "B_b+r0", "N", "N+r0", "NS1", "NS1+r0", "NS1+s2"],
            [("N+r0", "N"), ("B_b+r0", "B_b"), ("NS1", "N"), ("NS1+r0", "N+r0"), ("NS1+s2", "NS1+r0"),
             ("NS1+s2", "N"), ("NS1+s2", "B_b"), ("NS1+r0", "B_b")]),
}


def boot(meta, items, x, y, nb=2000, seed=0):
    st = {}
    for i in items:
        st.setdefault(meta[i]["stream"], []).append(i)
    keys = sorted(st)
    dx = {k: (sum(x[i] for i in st[k]), sum(y[i] for i in st[k]), len(st[k])) for k in keys}
    arr = np.array([dx[k] for k in keys], dtype=np.float64)
    diff = arr[:, 0].sum() / arr[:, 2].sum() - arr[:, 1].sum() / arr[:, 2].sum()
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(nb):
        s = arr[rng.integers(0, len(keys), len(keys))]
        bs.append(s[:, 0].sum() / s[:, 2].sum() - s[:, 1].sum() / s[:, 2].sum())
    return {"arm": arr[:, 0].sum() / arr[:, 2].sum(), "ref": arr[:, 1].sum() / arr[:, 2].sum(), "diff": diff,
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    systems, pairs = STAGES[a.stage]
    LD = laal_defs()
    res = {"stage": a.stage, "cells": {}, "means": {}, "dropped": sorted(DROP)}
    for d in DIRS:
        lang = LANG[d]
        meta = {m["id"]: m for m in jl(T / "data" / f"meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(T / "data" / f"items_{d}.jsonl")}
        P, PMv = {}, {}
        for s in systems:
            base, _, pol = s.partition("+")
            Pb = load(P3.sysd(BASE[base], "ecapa2"), d, LD)
            pol = pol or "none"
            rows = {r["id"]: r for r in jl(S0 / f"{base}_{pol}" / f"{d}.jsonl")}
            assert set(rows) == set(Pb), (s, d, len(rows), len(Pb))
            P[s] = {i: dict(v, d1=(rows[i]["D1"] / 1000 if rows[i]["D1"] is not None else None),
                            d2=(rows[i]["D2"] / 1000 if rows[i]["D2"] is not None else None)) for i, v in Pb.items()}
            if pol == "none":   # the player reproduces the stored timeline
                for i, v in Pb.items():
                    if v["d2"] is not None:
                        assert abs(v["d2"] - P[s][i]["d2"]) < 1e-6, (s, i)
            PMv[s] = {k: {i: (rows[i].get(k) if rows[i].get(k) is not None else 0.0) for i in rows} for k in PM}
        common = set.intersection(*[set(p) for p in P.values()]) - DROP
        for cell in CELLS:
            items = sorted(i for i in common if f"{meta[i]['set']}/{meta[i]['cond']}" == cell
                           and not (meta[i]["set"] == "obsi" and meta[i]["pair"] in EXCL[d]))
            k = f"{d}/{cell}"
            cres = {"n_items": len(items)}
            for s in systems:
                for q in ("d1", "d2"):
                    v = [P[s][i][q] for i in items if P[s][i][q] is not None]
                    res["means"][f"{k}/{s}/{q}"] = round(float(np.mean(v)), 3)
                for q in PM:
                    res["means"][f"{k}/{s}/{q}"] = round(float(np.mean([PMv[s][q][i] for i in items])), 3)
            for x, y in pairs:
                c = compare(meta, ref, items, P[x], P[y], lang, "stream")
                for q in PM:
                    c[q] = boot(meta, items, PMv[x][q], PMv[y][q])
                cres[f"{x}_vs_{y}"] = c
            res["cells"][k] = cres
    v = {}
    if a.stage in ("s0", "all"):
        bad = []
        for k, c in res["cells"].items():
            m = c["N+r0_vs_N"]
            npr = m["NP_in"]["diff"] / max(m["NP_in"]["ref"], 1e-9)
            ncr = m["NC"]["diff"] / max(m["NC"]["ref"], 1e-9)
            ok = npr <= -0.30 and ncr <= -0.50 and m["d2"]["diff"] <= 0.20
            same = m["asr"]["diff"] == 0
            if not (ok and same):
                bad.append([k, round(npr, 3), round(ncr, 3), m["d2"]["diff"], same])
        v["S0"] = {"failing_cells": bad, "PASS": not bad}
    if a.stage in ("s1", "all"):
        bad = []
        for k, c in res["cells"].items():
            m = c["NS1_vs_N"]
            ok = m["asr"]["diff"] >= -0.3 and m["asr"]["ci95"][1] >= 0 and m["d2"]["diff"] < 0
            if not ok:
                bad.append([k, m["asr"]["diff"], m["asr"]["ci95"], m["d2"]["diff"]])
        v["S1"] = {"failing_cells": bad, "PASS": not bad}
    if a.stage == "all":
        lower = [k for k, c in res["cells"].items() if c["NS1+s2_vs_NS1+r0"]["NP_in"]["diff"] < 0]
        d2bad = [[k, c["NS1+s2_vs_NS1+r0"]["d2"]["diff"]] for k, c in res["cells"].items()
                 if c["NS1+s2_vs_NS1+r0"]["d2"]["diff"] > 0.05]
        v["S2"] = {"cells_NP_in_lower": lower, "d2_over_0.05": d2bad, "PASS": len(lower) >= 8 and not d2bad}
    res["verdict"] = v
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(clean(res), fh, ensure_ascii=False, indent=1, allow_nan=False)
    print(json.dumps(v, indent=1))


if __name__ == "__main__":
    main()
