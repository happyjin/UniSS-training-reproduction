"""Robust wave 16 NCWI fit (reports/mist_w3po_v1/robust_wave16_ncwi_prereg.zh-CN.md, section 2; method: main-line plan 4.2).

Per direction: halves by sha1(pair) % 2 (pair = the sentence id clean / noisy twins share); for each fitting half h the
b-hat quartiles of h's (a) items are the bin edges; per bin, f(alpha) = mean sentence text BLEU of h's (a) items in the bin
at the three endpoints alpha = 0 (the 3 base runs, per-sentence mean), 0.5 (wave 14 unist_wise05), 1 (wave 14 unist);
exact quadratic f = a0 + g alpha - c alpha^2 through the three points; alpha* = clip(g / 2c, 0, 1) if c > 0, else the best
endpoint (ties: 0.5, then 0, then 1). The items of the OTHER half (a and b) get the alpha* of their bin under h's edges.
Writes <out>/fit.json and <out>/assign_<dir>.jsonl ({id, key, pair, half, bin, alpha, bhat}). Refuses to overwrite.
    <qwen3omni python> experiments/mist_w3po_v1/voicecond/wave16_fit.py --out eval_outputs/mist_w3po_v1/wave16/fit
    <qwen3omni python> experiments/mist_w3po_v1/voicecond/wave16_fit.py --selftest      (toy data, closed form)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"
W16 = REPO / "eval_outputs/mist_w3po_v1/wave16"
W14 = REPO / "eval_outputs/mist_w3po_v1/wave14"
BASE_RUNS = (S1, REPO / "eval_outputs/mist_w3po_v1/wave5/base_rerun1", REPO / "eval_outputs/mist_w3po_v1/wave5/base_rerun2")
ENDS = {0.5: W14 / "eval_unist_wise05", 1.0: W14 / "eval_unist"}
LANG = {"en2zh": "cmn", "zh2en": "eng"}


def jl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def half_of(pair: str) -> int:
    return int(hashlib.sha1(pair.encode("utf-8")).hexdigest(), 16) % 2


def edges_of(b) -> list:
    return [float(v) for v in np.quantile(np.asarray(b, dtype=np.float64), [0.25, 0.5, 0.75])]


def bin_of(b: float, edges) -> int:
    return int(np.searchsorted(np.asarray(edges), b, side="right"))


def quad(f0: float, f05: float, f1: float) -> dict:
    """exact f(alpha) = a0 + g alpha - c alpha^2 through (0, f0), (0.5, f05), (1, f1) and alpha*"""
    a0 = f0
    c = 2.0 * (2.0 * f05 - f0 - f1)
    g = f1 - f0 + c
    if c > 0:
        al, rule = float(np.clip(g / (2.0 * c), 0.0, 1.0)), "vertex" if 0 < g / (2.0 * c) < 1 else "clipped"
    else:
        al = max(((f05, 0.5), (f0, 0.0), (f1, 1.0)), key=lambda t: t[0])[1]   # max keeps the first of ties
        rule = "best_endpoint"
    return {"a0": a0, "g": g, "c": c, "alpha": round(al, 4), "rule": rule}


def selftest() -> None:
    rng = np.random.default_rng(0)
    n = 0
    for _ in range(2000):
        a0, g, c = rng.normal(20, 5), rng.normal(0, 8), rng.normal(0, 8)
        f = lambda x: a0 + g * x - c * x * x
        q = quad(f(0.0), f(0.5), f(1.0))
        assert abs(q["a0"] - a0) < 1e-9 and abs(q["g"] - g) < 1e-9 and abs(q["c"] - c) < 1e-9, (a0, g, c, q)
        grid = np.linspace(0, 1, 100001)
        if c > 0:   # the clipped vertex is the argmax of f on [0, 1]
            assert abs(q["alpha"] - grid[np.argmax(f(grid))]) < 1e-4, (a0, g, c, q)
        else:       # concave-free: the best of the three endpoints
            assert f(q["alpha"]) == max(f(0.0), f(0.5), f(1.0))
        n += 1
    assert quad(10, 12, 10) == {"a0": 10, "g": 8.0, "c": 8.0, "alpha": 0.5, "rule": "vertex"}
    assert quad(10, 11, 13)["rule"] == "best_endpoint" and quad(10, 11, 13)["alpha"] == 1.0      # c = -4
    assert quad(10, 10, 10)["alpha"] == 0.5                                                     # c = 0, tie -> 0.5
    assert quad(10, 9, 10)["alpha"] == 0.0                                                      # c < 0, tie 0 / 1 -> 0
    q = quad(10, 10.5, 9);  assert q["c"] == 4.0 and q["g"] == 3.0 and q["alpha"] == 0.375 and q["rule"] == "vertex"
    q = quad(10, 9.5, 8);   assert q["c"] == 2.0 and q["g"] == 0.0 and q["alpha"] == 0.0 and q["rule"] == "clipped"
    q = quad(8, 12, 13);    assert q["c"] == 6.0 and q["g"] == 11.0 and q["alpha"] == 0.9167      # g / 2c = 11 / 12
    q = quad(8, 12.5, 13);  assert q["c"] == 8.0 and q["g"] == 13.0 and q["alpha"] == 0.8125
    q = quad(8, 9, 13);     assert q["rule"] == "best_endpoint" and q["alpha"] == 1.0            # c = -6
    q = quad(12, 13, 8);    assert q["c"] == 12.0 and q["g"] == 8.0 and q["alpha"] == 0.3333
    q = quad(8, 12, 16.5);  assert q["c"] == -1.0 and q["alpha"] == 1.0
    q = quad(8, 13, 20);    assert q["c"] == -4.0 and q["alpha"] == 1.0
    q = quad(5, 12, 20);    assert q["c"] == -2.0
    q = quad(0, 3, 4.5);    assert q["c"] == 3.0 and q["g"] == 7.5 and q["alpha"] == 1.0 and q["rule"] == "clipped"
    q = quad(4.5, 3, 0);    assert q["c"] == 3.0 and q["g"] == -1.5 and q["alpha"] == 0.0 and q["rule"] == "clipped"
    # bins: quartile edges, values on an edge go to the upper bin, every value lands in 0..3
    x = np.arange(1, 101, dtype=float)
    e = edges_of(x)
    assert e == [25.75, 50.5, 75.25], e
    b = [bin_of(v, e) for v in x]
    assert [b.count(k) for k in range(4)] == [25, 25, 25, 25], [b.count(k) for k in range(4)]
    assert bin_of(25.75, e) == 1 and bin_of(-1e9, e) == 0 and bin_of(1e9, e) == 3
    # halves: deterministic, twins share one, roughly balanced
    pairs = [f"p{i}" for i in range(4000)]
    h = [half_of(p) for p in pairs]
    assert h == [half_of(p) for p in pairs] and 0.45 < np.mean(h) < 0.55
    assert half_of("common_voice_en_19942594") == half_of("common_voice_en_19942594")
    print(f"wave16_fit selftest OK ({n} random quadratics + fixed cases, bins, halves)")


def sent_bleu(lang: str):
    from sacrebleu.metrics import BLEU
    from evaluation.text_metrics import normalize_for_bleu
    m = BLEU(tokenize="zh" if lang == "cmn" else "13a", effective_order=True)
    return lambda h, r: float(m.sentence_score(normalize_for_bleu(h, lang), [normalize_for_bleu(r, lang)]).score)


def hyps(root: Path, d: str, ids) -> dict:
    blk = {r["id"]: "".join(b["text"] for b in r["blocks"]) for r in jl(root / f"thinker_{d}" / f"blocks_{d}_w1.jsonl")}
    miss = [i for i in ids if i not in blk]
    assert not miss, f"{root} {d}: {len(miss)} items missing"
    return blk


def main() -> None:
    import sys
    sys.path.insert(0, str(REPO))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    selftest()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    res = {"endpoints": {"0": [str(r.relative_to(REPO)) for r in BASE_RUNS],
                         **{str(k): str(v.relative_to(REPO)) for k, v in ENDS.items()}}}
    assigns = {}
    for d in ("en2zh", "zh2en"):
        meta = {m["id"]: m for m in jl(S1 / "data" / f"meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(S1 / "data" / f"items_{d}.jsonl")}
        bh = {r["id"]: r for r in jl(W16 / "noise" / f"bhat_{d}.jsonl")}
        assert set(bh) == set(meta), d
        sb = sent_bleu(LANG[d])
        a_ids = sorted(i for i, m in meta.items() if m["set"] == "a")
        H = {0.0: [hyps(r, d, a_ids) for r in BASE_RUNS], **{k: [hyps(v, d, a_ids)] for k, v in ENDS.items()}}
        sbleu = {al: {i: float(np.mean([sb(h[i], ref[i]) for h in hs])) for i in a_ids} for al, hs in H.items()}
        half = {i: half_of(m["pair"]) for i, m in meta.items()}
        res[d] = {}
        for h in (0, 1):
            fit_ids = [i for i in a_ids if half[i] == h]
            e = edges_of([bh[i]["bhat"] for i in fit_ids])
            bins = []
            for k in range(4):
                ids = [i for i in fit_ids if bin_of(bh[i]["bhat"], e) == k]
                f = {al: float(np.mean([sbleu[al][i] for i in ids])) for al in (0.0, 0.5, 1.0)}
                q = quad(f[0.0], f[0.5], f[1.0])
                bins.append({"bin": k, "n_fit_a": len(ids), "n_fit_a_noisy": sum(meta[i]["cond"] == "noisy" for i in ids),
                             "f0": round(f[0.0], 4), "f05": round(f[0.5], 4), "f1": round(f[1.0], 4),
                             **{x: (round(v, 4) if isinstance(v, float) else v) for x, v in q.items()}})
            res[d][f"fit_half{h}"] = {"n_fit_a": len(fit_ids), "edges_bhat": e,
                                      "edges_db": [round(10 * np.log10(v + 1e-12), 3) for v in e], "bins": bins}
        rows = []
        for i, m in meta.items():
            fh = 1 - half[i]                       # cross-fit: the alpha table of the other half
            fr = res[d][f"fit_half{fh}"]
            k = bin_of(bh[i]["bhat"], fr["edges_bhat"])
            rows.append({"id": i, "key": bh[i]["key"], "pair": m["pair"], "half": half[i], "fit_half": fh, "bin": k,
                         "alpha": fr["bins"][k]["alpha"], "bhat": bh[i]["bhat"]})
        assigns[d] = rows
        al = sorted({r["alpha"] for r in rows})
        assert len(al) <= 8, al
        res[d]["distinct_alphas"] = al
        res[d]["n_items_per_alpha"] = {str(x): sum(r["alpha"] == x for r in rows) for x in al}
        res[d]["n_items_per_alpha_set_cond"] = {
            str(x): {f"{s}_{c}": sum(r["alpha"] == x and meta[r["id"]]["set"] == s and meta[r["id"]]["cond"] == c
                                     for r in rows) for s in ("a", "b") for c in ("clean", "noisy")} for x in al}
    out.mkdir(parents=True)
    for d, rows in assigns.items():
        with open(out / f"assign_{d}.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with open(out / "fit.json", "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({d: {"alphas": res[d]["distinct_alphas"],
                          **{f"half{h}": [b["alpha"] for b in res[d][f"fit_half{h}"]["bins"]] for h in (0, 1)}}
                      for d in ("en2zh", "zh2en")}), flush=True)


if __name__ == "__main__":
    main()
