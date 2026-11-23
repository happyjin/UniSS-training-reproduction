"""sst_hsmm_v1 P3 test scoring (reports/sst_hsmm_v1/prereg.zh-CN.md 3.6 + change 2). Systems B_b, N (= test_robust UE_mbr)
and NEW (eval_outputs/sst_hsmm_v1/p3: HSMM Thinker + Kalman/AbS voice + T1c), all three through the dip player
(s0_apply --policy dip -> p3/play/<sys>_<d>.jsonl): D1 / D2 / NP_in / NC / PT_in / EndOffset from the player; ASR-BLEU
(speech-only ASR), PESQ (squim), identity (ECAPA2) via wave3_v_score.compare (unit = stream, 2000 resamples, seed 0);
playout metrics by s_score.boot. Cells as voice_onset_v1 P3 (health_003 dropped, OBSI en2zh without e031_0004).
Rules NEW vs N per cell (10 cells): (1) ASR diff >= -0.3 and CI high >= 0 in all 10; (2) D2 diff < 0 in >= 8;
(3) NP_in rel <= -30% in >= 8; (4) NC rel <= -50% in >= 8. Identity CI low >= -0.01: reported only.
--root p3c --rules pareto (prereg changes 6 / 7, P3''): NEW = <root>; rules (2) D2 diff <= +0.05 in >= 8 cells,
(3) NP_in rel <= -25% in >= 8, (4) NC reported only; (1) unchanged.
    qwen3omni PYTHONPATH=. python -m experiments.sst_hsmm_v1.p3_score --out eval_outputs/sst_hsmm_v1/p3/summary.json
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from experiments.mist_w3po_v1.voicecond.combo_score import LANG, clean, jl, text_compare
from experiments.mist_w3po_v1.voicecond.test_score import EXCL
from experiments.mist_w3po_v1.voicecond.wave3_v_score import compare
from experiments.voice_onset_v1 import p3_score as P3
from experiments.voice_onset_v1.s_score import boot

REPO = Path(__file__).resolve().parents[2]
R = REPO / "eval_outputs/sst_hsmm_v1/p3"
T, DIRS, CELLS, DROP = P3.T, P3.DIRS, P3.CELLS, P3.DROP
PM = ("NP_in", "PT_in", "PT_all", "NC", "n_pause", "EndOffset")
PAIRS = [("NEW", "N"), ("NEW", "B_b"), ("N", "B_b")]


def sysd(name):
    if name == "NEW":
        return {"talker": lambda d: R / "new" / f"talker_{d}",
                "metrics": lambda d, k: R / "metrics" / (f"ecapa2_new_{d}_spksim.jsonl" if k == "spksim" else f"new_{d}_{k}.jsonl"),
                "thinker": lambda d: R / f"thinker_{d}" / f"blocks_{d}_w1.jsonl"}
    return P3.sysd({"N": "UE_mbr", "B_b": "B_b"}[name], "ecapa2")


def load(name, d):
    S = sysd(name)
    run = S["talker"](d)
    asr = {r["id"]: r.get("asr_text", "") for r in jl(run / "asr_speech.jsonl")}
    pesq = {r["id"]: r.get("pesq") for r in jl(S["metrics"](d, "squim"))}
    sim = {r["id"]: r.get("sim") for r in jl(S["metrics"](d, "spksim"))}
    tr = {r["id"]: r["translation"] for r in jl(S["thinker"](d))}
    per = {}
    for r in jl(R / "play" / f"{name}_{d}.jsonl"):
        if r.get("D2") is None:
            continue
        i = r["id"]
        per[i] = {"asr": asr.get(i, ""), "pesq": pesq.get(i), "sim": sim.get(i), "text": tr.get(i), "d1": r["D1"] / 1000,
                  "d2": r["D2"] / 1000, **{k: (r.get(k) if r.get(k) is not None else 0.0) for k in PM}}
    return per


def pair(meta, ref, items, X, Y, lang):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = compare(meta, ref, items, X, Y, lang, "stream")
    c["text_bleu"] = text_compare(meta, ref, items, X, Y, lang, "stream")
    for q in PM:
        b = boot(meta, items, {i: X[i][q] for i in items}, {i: Y[i][q] for i in items})
        c[q] = {"arm": float(b["arm"]), "ref": float(b["ref"]), "diff": float(b["diff"]), "ci95": [float(v) for v in b["ci95"]]}
        c[q]["rel"] = c[q]["diff"] / max(c[q]["ref"], 1e-9)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="p3", help="NEW run dir under eval_outputs/sst_hsmm_v1 (p3c for P3'')")
    ap.add_argument("--rules", choices=("orig", "pareto"), default="orig")
    a = ap.parse_args()
    global R
    R = REPO / "eval_outputs/sst_hsmm_v1" / a.root
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    res = {"cells": {}, "dropped": sorted(DROP)}
    for d in DIRS:
        lang = LANG[d]
        meta = {m["id"]: m for m in jl(T / "data" / f"meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(T / "data" / f"items_{d}.jsonl")}
        P = {s: load(s, d) for s in ("B_b", "N", "NEW")}
        common = set.intersection(*[set(p) for p in P.values()]) - DROP
        for cell in CELLS:
            items = sorted(i for i in common if f"{meta[i]['set']}/{meta[i]['cond']}" == cell
                           and not (meta[i]["set"] == "obsi" and meta[i]["pair"] in EXCL[d]))
            cres = {"n_items": len(items), "n_sys": {s: sum(1 for i in P[s] if f"{meta[i]['set']}/{meta[i]['cond']}" == cell)
                                                     for s in P}}
            for x, y in PAIRS:
                cres[f"{x}_vs_{y}"] = pair(meta, ref, items, P[x], P[y], lang)
            res["cells"][f"{d}/{cell}"] = cres
    C = {k: c["NEW_vs_N"] for k, c in res["cells"].items()}
    r1 = [[k, m["asr"]["diff"], m["asr"]["ci95"]] for k, m in C.items() if not (m["asr"]["diff"] >= -0.3 and m["asr"]["ci95"][1] >= 0)]
    par = a.rules == "pareto"
    r2 = [k for k, m in C.items() if (m["d2"]["diff"] <= 0.05 if par else m["d2"]["diff"] < 0)]
    r3 = [k for k, m in C.items() if m["NP_in"]["rel"] <= (-0.25 if par else -0.30)]
    r4 = [k for k, m in C.items() if m["NC"]["rel"] <= -0.50]
    idb = [[k, m["sim"]["ci95"][0]] for k, m in C.items() if m["sim"]["ci95"][0] < -0.01]
    v = {"R1_asr_failing": r1, "R2_d2_lower_cells": len(r2), "R3_np_in_cells": len(r3), "R4_nc_cells": len(r4),
         "identity_ci_low_below_-0.01": idb,
         "pass": {"R1": not r1, "R2": len(r2) >= 8, "R3": len(r3) >= 8, **({} if par else {"R4": len(r4) >= 8})},
         "rules": a.rules}
    v["NEW_established"] = all(v["pass"].values())
    res["verdict"] = v
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(clean(res), fh, ensure_ascii=False, indent=1, allow_nan=False)
    print(json.dumps(clean(v), indent=1))


if __name__ == "__main__":
    main()
