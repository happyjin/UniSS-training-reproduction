"""v11 P0-1 PA factorial ablation: numbers and paired tests (reports/mist_w3po_v1/v11_p01_factorial_prereg.zh-CN.md §4-§5).
CPU, read-only; writes one new JSON (refuses to overwrite).

Rows (text = B's, joint_v8h6k_20261113T073658Z_s06000; run = <text><suffix>, seed 2 = <run>_tts2):
  R0 __t1c  R1 __rlt1c  R2 __rlc1s2  R3 __rlc1b1s2  R3p __rlc1b1s<N>  R4 __rlplemk1  R5 __rlplem  (+ dev calibration rows)
Inputs:
  ASR-BLEU   decode/<run>/score_asr.json (tcpBLEU all / ov / 1spk; per-meeting sacreBLEU statistics); the two seeds'
             statistics are averaged per meeting (make_tables_v10.pooled) and the paired test is make_tables_v10.paired
             (exact two-sided sign-flip over all 2^M flips, 95% interval from 10,000 meeting-bootstrap resamples).
  latency    <A>/lat/{perword,blockstart}/<set>_<row>_s<k>.json(.units.jsonl) from v11_p01_postproc.sh: unit means
             (point = mean of the two seeds) and post_review_v10_analysis.paired_ratio on per-meeting sums.
  rate       talker/results.jsonl (units per second of speech; zh2en: English words, en2zh: CJK chars / Latin words).
  UTMOS/SIM  <A>/quality/<run>/quality.json (seed 1); paired UTMOS over common streams, meeting bootstrap.
  pauses     <A>/pauses/block_pauses_<set>.jsonl (>= 0.5 s gaps per minute of speech; mid-sentence ones).
Holm over the primary family (R5 - R3p: ASR-BLEU and LAAL_block on the test sets ami / ali / en).

    python -m experiments.mist_w3po_v1.evaluation.v11_factorial_analysis --sets devz deve \
        --rows R0:__t1c R1:__rlt1c ... --out <new json> [--shift-z N --shift-e N]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.evaluation import post_review_v10_analysis as PRA
from experiments.mist_w3po_v1.evaluation.talker_block_pauses import n_words
from experiments.mist_w3po_v1.evaluation.zh_units import n_units

EO = Path("eval_outputs/mist_w3po_v1")
A = EO / "joint/v11f_analysis"
DEC, SPD = EO / "decode", EO / "speech"
RT1 = "joint_v8h6k_20261113T073658Z_s06000"
TEXT = {"devz": f"jdev_bz_{RT1}", "deve": f"jdev_be_{RT1}", "ami": f"jtest_bami_{RT1}", "ali": f"jtest_bali_{RT1}",
        "en": f"jtest_ben_{RT1}"}
ZH_OUT = {"deve", "en"}           # Chinese output: count CJK chars / Latin words
NBOOT = 10000
CONTRASTS = (("R1", "R0"), ("R2", "R1"), ("R3", "R2"), ("R4", "R2"), ("R5", "R4"), ("R5", "R3"), ("R5", "R3p"),
             ("R3p", "R3"), ("R5", "R0"))
LK = {"speech_laal_online": "slaal", "speech_end_online": "endoff", "speech_start_online": "startoff"}


def bleu_vec(S):
    S = np.atleast_2d(np.asarray(S, dtype=np.float64))
    hyp, ref, cor, tot = S[:, 0], S[:, 1], S[:, 2:6], S[:, 6:10]
    k = np.cumsum(cor == 0, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(cor > 0, 100.0 * cor / np.maximum(tot, 1), 100.0 / (np.power(2.0, k) * np.maximum(tot, 1)))
        p = np.where(tot > 0, p, 0.0)
        logp = np.where(p > 0, np.log(np.maximum(p, 1e-300)), -np.inf)
        bp = np.where(hyp < ref, np.exp(1 - ref / np.maximum(hyp, 1e-9)), 1.0)
        out = np.where(hyp > 0, bp * np.exp(logp.mean(1)), 0.0)
    return np.nan_to_num(out, nan=0.0, neginf=0.0)


def pooled(scores, reg):
    ms = set.intersection(*[set(s["per_meeting"]) for s in scores])
    return {m: np.mean([np.asarray(s["per_meeting"][m].get(f"{reg}_stats") or [0.0] * 10, float) for s in scores], 0)
            for m in ms}


def paired_bleu(a: dict, b: dict, rng) -> dict:
    """b - a on meeting-level BLEU statistics (make_tables_v10.paired)."""
    ms = sorted(set(a) & set(b))
    A_ = np.stack([a[m] for m in ms])
    B_ = np.stack([b[m] for m in ms])
    obs = float(bleu_vec(B_.sum(0))[0] - bleu_vec(A_.sum(0))[0])
    M_ = len(ms)
    idx = np.arange(2 ** M_, dtype=np.int64)
    F = ((idx[:, None] >> np.arange(M_)[None, :]) & 1).astype(np.float64)
    D = B_ - A_
    d = bleu_vec(B_.sum(0)[None, :] - F @ D) - bleu_vec(A_.sum(0)[None, :] + F @ D)
    p = float((np.abs(d) >= abs(obs) - 1e-9).mean())
    J = rng.integers(0, M_, (NBOOT, M_))
    ds = bleu_vec(B_[J].sum(1)) - bleu_vec(A_[J].sum(1))
    return {"diff": round(obs, 3), "p": p, "ci95": [round(float(np.percentile(ds, 2.5)), 3),
                                                    round(float(np.percentile(ds, 97.5)), 3)], "meetings": M_}


def holm(tests: dict) -> None:
    order = sorted(tests, key=lambda k: tests[k]["p"])
    m, run = len(order), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * tests[k]["p"]))
        tests[k]["p_holm"] = run


def j(p: Path):
    try:
        return json.load(open(p))
    except Exception:
        return None


def runs(s: str, sfx: str) -> list[str]:
    n = TEXT[s] + sfx
    return [n, n + "_tts2"]


def lat_units(s: str, row: str, k: int, mode: str):
    out = defaultdict(lambda: {"vals": [], "pm": defaultdict(lambda: [0.0, 0])})
    f = A / "lat" / mode / f"{s}_{row}_s{k}.json.units.jsonl"
    if not f.exists():
        return None
    for l in open(f):
        m, uid, reg, key, v = json.loads(l)
        if key not in LK or reg not in ("ov", "1spk"):
            continue
        out[key]["vals"].append(v)
        out[key]["pm"][m][0] += v
        out[key]["pm"][m][1] += 1
    return out


def rate_stats(s: str, run: str):
    cnt = n_units if s in ZH_OUT else n_words
    sec = u = 0.0
    pm = defaultdict(lambda: [0.0, 0.0])
    f = SPD / run / "talker/results.jsonl"
    if not f.exists():
        return None
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        if r.get("error"):
            continue
        m = PRA.meeting_of(r["id"].split("__S")[0])
        for b in r.get("blocks") or []:
            d, x = b["dur_ms"] / 1000, cnt(b["text"])
            sec += d
            u += x
            pm[m][0] += x
            pm[m][1] += d
    return {"units_per_s": u / sec, "speech_min": sec / 60}, {m: tuple(v) for m, v in pm.items()}


def analyse_set(s: str, rows: dict, rng) -> dict:
    R = {"rows": {}, "paired": {}}
    asr, latd, rated, utm = {}, {}, {}, {}
    pz = {}
    pf = A / "pauses" / f"block_pauses_{s}.jsonl"
    if pf.exists():
        for l in open(pf):
            r = json.loads(l)
            pz[r["talker"].rstrip("/").split("/")[-2]] = r
    for row, sfx in rows.items():
        rn = runs(s, sfx)
        x = {"runs": rn}
        sc = [j(DEC / n / "score_asr.json") for n in rn]
        if all(sc):
            for reg in ("all", "ov", "1spk"):
                x[f"asr_{reg}"] = round(float(np.mean([v[f"tcpBLEU_{reg}"] for v in sc])), 3)
            asr[row] = sc
        for mode in ("perword", "blockstart"):
            us = [lat_units(s, row, k, mode) for k in (1, 2)]
            if not all(us):
                continue
            for key, short in LK.items():
                if mode == "blockstart" and key != "speech_laal_online":
                    continue
                name = short if key != "speech_laal_online" else ("laal_spread" if mode == "perword" else "laal_block")
                x[name] = round(float(np.mean([np.mean(u[key]["vals"]) for u in us])), 4)
                latd[(row, name)] = PRA.pool_seeds([{m: tuple(v) for m, v in u[key]["pm"].items()} for u in us])
        rs = [rate_stats(s, n) for n in rn]
        if all(rs):
            x["rate"] = round(float(np.mean([r[0]["units_per_s"] for r in rs])), 4)
            x["speech_min"] = round(float(np.mean([r[0]["speech_min"] for r in rs])), 1)
            rated[row] = PRA.pool_seeds([r[1] for r in rs])
        ps = [pz.get(n) for n in rn]
        if all(ps):
            x["pauses_per_min"] = round(float(np.mean([p["pauses_all"]["per_min"] for p in ps])), 2)
            x["mid_pauses_per_min"] = round(float(np.mean([p["pauses_mid_sent"]["per_min"] for p in ps])), 2)
        q = A / "quality" / rn[0] / "quality.json"
        if q.exists():
            qq = json.load(open(q))
            x["utmos"], x["sim"] = round(qq["utmos_mean"], 3), round(qq["sim_mean"], 4)
            utm[row] = {json.loads(l)["id"]: json.loads(l)["utmos"] for l in open(str(q) + ".items.jsonl")}
        R["rows"][row] = x
    for c, r in CONTRASTS:
        if c not in rows or r not in rows:
            continue
        key = f"{c}-{r}"
        P = {}
        if c in asr and r in asr:
            for reg in ("all", "ov"):
                P[f"asr_{reg}"] = paired_bleu(pooled(asr[r], reg), pooled(asr[c], reg), rng)
        for name in ("laal_spread", "laal_block", "startoff", "endoff"):
            if (c, name) in latd and (r, name) in latd:
                P[name] = PRA.paired_ratio(latd[(r, name)], latd[(c, name)], rng)
        if c in rated and r in rated:
            P["rate"] = PRA.paired_ratio(rated[r], rated[c], rng)
        if c in utm and r in utm:
            Aa, Bb = defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0.0, 0])
            common = sorted(set(utm[c]) & set(utm[r]))
            for k in common:
                m = PRA.meeting_of(k.split("__S")[0])
                Bb[m][0] += utm[c][k]
                Bb[m][1] += 1
                Aa[m][0] += utm[r][k]
                Aa[m][1] += 1
            P["utmos"] = PRA.paired_ratio({m: tuple(v) for m, v in Aa.items()}, {m: tuple(v) for m, v in Bb.items()}, rng)
        R["paired"][key] = P
    return R


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--rows", nargs="+", required=True, help="ROW:suffix; R3p's suffix may hold {N}")
    ap.add_argument("--shift-z", type=int, default=0, help="R3p shift for zh2en sets (devz ami ali)")
    ap.add_argument("--shift-e", type=int, default=0, help="R3p shift for en2zh sets (deve en)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    rng = np.random.default_rng(20261114)
    res = {"doc": __doc__, "sets": {}}
    tests = {}
    for s in a.sets:
        N = a.shift_e if s in ZH_OUT else a.shift_z
        rows = {}
        for rs in a.rows:
            row, sfx = rs.split(":", 1)
            if "{N}" in sfx:
                if not N:
                    continue
                sfx = sfx.replace("{N}", str(N))
            rows[row] = sfx
        res["sets"][s] = analyse_set(s, rows, rng)
        res["sets"][s]["r3p_shift"] = N or None
        if s in ("ami", "ali", "en"):
            P = res["sets"][s]["paired"].get("R5-R3p", {})
            for m in ("asr_all", "laal_block"):
                if m in P:
                    t = dict(P[m])
                    t["p"] = t.get("p", t.get("p_signflip"))
                    tests[f"{s}:{m}"] = t
    if tests:
        holm(tests)
        res["primary_holm"] = tests
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
