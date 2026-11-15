"""v11 P0-4: post_review_v10_analysis.py with the rule-selected stage-2 step 4000 (+ wait bias -1) added as system "m4"
(jtest_m{s}_joint_v2p_20261113T120415Z_s04000_wbzm1em1__rlplem; its inputs under joint/post_review_v11_s4000, written
by scripts/s4000_postproc.sh; every other system read from joint/post_review_v10 as before). Pairs m4-b, m4-e2, m4-m
added; everything else unchanged. Original docstring:

Post-review analysis of the v10 test runs (paper/gstar_si_v10/review_v10.md, items M5 and M2). CPU, read-only on
existing outputs; writes one new JSON (refuses to overwrite).

Systems (test sets ami / ali / en, TTS seeds s1 / s2 = <run> / <run>_tts2):
  b   strict baseline        jtest_b{s}_joint_v8h6k_20261113T073658Z_s06000__t1c
  e2  B + rl + PL-EM         jtest_b{s}_joint_v8h6k_20261113T073658Z_s06000__rlplem
  m0  method, delta_w = 0    jtest_m{s}_joint_v2p_20261113T120415Z_s03500__rlplem
  m   method, delta_w = -1   jtest_m{s}_joint_v2p_20261113T120415Z_s03500_wbzm1em1__rlplem

M5 (latency). Inputs: <post_review_v10>/lat/{perword,blockstart}/<set>_<sys>_s<k>.json(.units.jsonl), written by
lat_units_dump.py (speech_latency_lit on the chunk protocol, same arguments as joint_lib.sh; perword = the paper's speech
LAAL and reproduces the stored lat_chunk_<run>.json field by field; blockstart = every word charged at its block's
online playback start, reproduces the v10 block-start table inputs). Region "all" (= ov + 1spk units).
  point      mean over the two seeds of the unit mean (the paper's number)
  paired     per meeting, the sum and the count of unit values, averaged over the two seeds (as gate_2seed pools the
             BLEU statistics); statistic = ratio of summed sums to summed counts; difference cand - ref over the
             meetings both have; 95% interval from 10,000 meeting-bootstrap resamples (rng 20261114, as
             make_tables_v10.py) and the exact two-sided sign-flip permutation p over all 2^M flips.
M2 (speech). Per run, from talker/results.jsonl (blocks without error): total synthesised speech (sum of block
dur_ms), output units (zh2en: whitespace tokens with a letter or digit = talker_block_pauses.n_words; en2zh: CJK
characters / Latin words = zh_units.n_units), units per second of speech, speech seconds per window (windows with an
output stream), load (speech / source speech union, from the latency json). Rate differences: same paired meeting
bootstrap (ratio of summed units to summed seconds).
UTMOS / SIM: talker/talker_quality.py outputs under <post_review_v10>/quality/<run>/ (seed 1, every stream with
>= 0.5 s speech); paired UTMOS difference as pmc_listen_compare.paired (same-stream pairs, ratio of sums, meeting
bootstrap) but with meeting = window id before "_on_" (pmc_listen_compare's prefix-before-first-"_" would merge the
AliMeeting meetings of one room) and 10,000 resamples.
Block / pause statistics: talker_block_pauses outputs under <post_review_v10>/pauses/ (copied through).

    python -m experiments.mist_w3po_v1.evaluation.post_review_v10_analysis --out <new json>
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.evaluation.talker_block_pauses import n_words
from experiments.mist_w3po_v1.evaluation.zh_units import n_units

EO = Path("eval_outputs/mist_w3po_v1")
PR = EO / "joint/post_review_v10"
PR4 = EO / "joint/post_review_v11_s4000"


def prd(sy: str) -> Path:
    return PR4 if sy == "m4" else PR
SPD = EO / "speech"
RB = "joint_v8h6k_20261113T073658Z_s06000"
RM = "joint_v2p_20261113T120415Z_s03500"
SETS = ("ami", "ali", "en")
SYSS = ("b", "e2", "m0", "m", "m4")
PAIRS = (("m", "b"), ("e2", "b"), ("m", "e2"), ("m0", "b"), ("m", "m0"), ("m4", "b"), ("m4", "e2"), ("m4", "m"))
LKEYS = {"speech_laal_online": "slaal", "speech_end_online": "endoff", "speech_start_online": "startoff",
         "text_laal_online": "tlaal"}
NBOOT = 10000


def run_name(s: str, sy: str, k: int) -> str:
    n = {"b": f"jtest_b{s}_{RB}__t1c", "e2": f"jtest_b{s}_{RB}__rlplem", "m0": f"jtest_m{s}_{RM}__rlplem",
         "m": f"jtest_m{s}_{RM}_wbzm1em1__rlplem",
         "m4": f"jtest_m{s}_joint_v2p_20261113T120415Z_s04000_wbzm1em1__rlplem"}[sy]
    return n + ("_tts2" if k == 2 else "")


def meeting_of(stream_id: str) -> str:
    return stream_id.split("_on_")[0]


# ------------------------------------------------------------------------------------------------ paired statistics
def paired_ratio(A: dict, B: dict, rng) -> dict:
    """B - A of sum/count statistics; A, B: meeting -> (sum, count)."""
    ms = sorted(set(A) & set(B))
    a = np.array([A[m] for m in ms], float)
    b = np.array([B[m] for m in ms], float)
    obs = b[:, 0].sum() / b[:, 1].sum() - a[:, 0].sum() / a[:, 1].sum()
    M_ = len(ms)
    J = rng.integers(0, M_, (NBOOT, M_))
    bs = b[J].sum(1)
    as_ = a[J].sum(1)
    ds = bs[:, 0] / bs[:, 1] - as_[:, 0] / as_[:, 1]
    idx = np.arange(2 ** M_, dtype=np.int64)
    F = ((idx[:, None] >> np.arange(M_)[None, :]) & 1).astype(np.float64)
    D = b - a
    Bf = b.sum(0)[None, :] - F @ D
    Af = a.sum(0)[None, :] + F @ D
    d = Bf[:, 0] / Bf[:, 1] - Af[:, 0] / Af[:, 1]
    p = float((np.abs(d) >= abs(obs) - 1e-9).mean())
    return {"diff_pooled": round(float(obs), 4), "ci95": [round(float(np.percentile(ds, 2.5)), 4),
                                                          round(float(np.percentile(ds, 97.5)), 4)],
            "p_signflip": p, "p_boot_le0": round(float((ds <= 0).mean()), 4), "meetings": M_}


def pool_seeds(per_seed: list) -> dict:
    """[meeting -> (sum, count)] per seed -> meeting -> mean over seeds (meetings common to all seeds)."""
    ms = set.intersection(*[set(x) for x in per_seed])
    return {m: tuple(np.mean([x[m] for x in per_seed], 0)) for m in ms}


# ------------------------------------------------------------------------------------------------ M5 latency
def lat_units(mode: str, s: str, sy: str, k: int) -> dict:
    """key -> {"vals": [...], "per_meeting": meeting -> (sum, count)} for region all."""
    out = defaultdict(lambda: {"vals": [], "pm": defaultdict(lambda: [0.0, 0])})
    for l in open(prd(sy) / "lat" / mode / f"{s}_{sy}_s{k}.json.units.jsonl"):
        m, uid, reg, key, v = json.loads(l)
        if key not in LKEYS or reg not in ("ov", "1spk"):
            continue
        out[key]["vals"].append(v)
        out[key]["pm"][m][0] += v
        out[key]["pm"][m][1] += 1
    return out


def latency(rng) -> dict:
    R = {}
    for s in SETS:
        R[s] = {"systems": {}, "paired": {}}
        D = {}
        for mode in ("perword", "blockstart"):
            for sy in SYSS:
                us = [lat_units(mode, s, sy, k) for k in (1, 2)]
                for key, short in LKEYS.items():
                    if mode == "blockstart" and key != "speech_laal_online":
                        continue          # only the LAAL formula changes between the conventions
                    name = short if key != "speech_laal_online" else f"slaal_{mode}"
                    seeds = [float(np.mean(u[key]["vals"])) for u in us]
                    stored = [json.load(open(prd(sy) / "lat" / mode / f"{s}_{sy}_s{k}.json"))["unit"][key]["all"]["mean"]
                              for k in (1, 2)]
                    assert all(abs(a - b) < 6e-4 for a, b in zip(seeds, stored)), (s, sy, key, seeds, stored)
                    R[s]["systems"].setdefault(sy, {})[name] = {"mean": round(float(np.mean(seeds)), 4),
                                                                "seeds": [round(x, 4) for x in seeds],
                                                                "n_units": [len(u[key]["vals"]) for u in us]}
                    D[(sy, name)] = pool_seeds([{m: tuple(v) for m, v in u[key]["pm"].items()} for u in us])
        names = sorted({n for (_, n) in D})
        for c, r in PAIRS:
            for name in names:
                res = paired_ratio(D[(r, name)], D[(c, name)], rng)
                res["diff"] = round(R[s]["systems"][c][name]["mean"] - R[s]["systems"][r][name]["mean"], 4)
                R[s]["paired"].setdefault(f"{c}-{r}", {})[name] = res
    # endoff is mode independent: check once
    for s in SETS:
        for sy in SYSS:
            for k in (1, 2):
                a = json.load(open(prd(sy) / "lat/perword" / f"{s}_{sy}_s{k}.json"))["unit"]["speech_end_online"]
                b = json.load(open(prd(sy) / "lat/blockstart" / f"{s}_{sy}_s{k}.json"))["unit"]["speech_end_online"]
                assert a == b, (s, sy, k)
    return R


# ------------------------------------------------------------------------------------------------ M2 speech
def speech_stats(s: str, sy: str, k: int) -> tuple[dict, dict]:
    n = run_name(s, sy, k)
    cnt = n_units if s == "en" else n_words
    sec = units = 0.0
    per_win = defaultdict(float)
    pm = defaultdict(lambda: [0.0, 0.0])        # meeting -> [units, seconds]
    blocks = 0
    for l in open(SPD / n / "talker/results.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r.get("error"):
            continue
        wid = r["id"].split("__S")[0]
        per_win[wid] += 0.0
        for b in r.get("blocks") or []:
            d = b["dur_ms"] / 1000
            u = cnt(b["text"])
            sec += d
            units += u
            per_win[wid] += d
            pm[meeting_of(wid)][0] += u
            pm[meeting_of(wid)][1] += d
            blocks += 1
    lat = json.load(open(prd(sy) / "lat/perword" / f"{s}_{sy}_s{k}.json"))
    pw = np.array(list(per_win.values()))
    res = {"run": n, "speech_s": round(sec, 1), "units": int(units), "units_per_s": round(units / sec, 4),
           "blocks": blocks, "windows": len(pw), "speech_s_per_window": {"mean": round(float(pw.mean()), 3),
                                                                     "p50": round(float(np.median(pw)), 3)},
           "load_speech_over_src_union": lat["load"]["speech_over_src_union"],
           "src_speech_union_s": lat["load"]["src_speech_union_s"], "block_dur_s": lat["load"]["block_dur"]}
    return res, {m: (v[0], v[1]) for m, v in pm.items()}


def speech(rng) -> dict:
    R = {}
    for s in SETS:
        R[s] = {"unit": "CJK char / Latin word" if s == "en" else "English word", "systems": {}, "paired_rate": {}}
        P = {}
        for sy in SYSS:
            st = [speech_stats(s, sy, k) for k in (1, 2)]
            seeds = [x[0] for x in st]
            R[s]["systems"][sy] = {
                "seeds": seeds,
                "units_per_s": round(float(np.mean([x["units_per_s"] for x in seeds])), 4),
                "speech_s_per_window": round(float(np.mean([x["speech_s_per_window"]["mean"] for x in seeds])), 3),
                "speech_min": round(float(np.mean([x["speech_s"] for x in seeds])) / 60, 2),
                "load": round(float(np.mean([x["load_speech_over_src_union"] for x in seeds])), 4)}
            P[sy] = pool_seeds([x[1] for x in st])
        for c, r in PAIRS:
            res = paired_ratio(P[r], P[c], rng)
            res["diff"] = round(R[s]["systems"][c]["units_per_s"] - R[s]["systems"][r]["units_per_s"], 4)
            R[s]["paired_rate"][f"{c}-{r}"] = res
    return R


def quality(rng) -> dict:
    R = {}
    for s in SETS:
        R[s] = {"systems": {}, "paired_utmos": {}}
        items = {}
        for sy in ("b", "e2", "m", "m4"):
            d = prd(sy) / "quality" / run_name(s, sy, 1)
            q = json.load(open(d / "quality.json")) if (d / "quality.json").exists() else None
            if q is None:
                continue
            R[s]["systems"][sy] = {"utmos_mean": q["utmos_mean"], "sim_mean": q["sim_mean"], "n_utmos": q["n_utmos"],
                                   "seed": 1}
            items[sy] = {json.loads(l)["id"]: json.loads(l)["utmos"] for l in open(d / "quality.json.items.jsonl")}
        for c, r in (("m", "b"), ("e2", "b"), ("m", "e2"), ("m4", "b"), ("m4", "m")):
            if c not in items or r not in items:
                continue
            A, B = defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0.0, 0])
            common = sorted(set(items[c]) & set(items[r]))
            for k in common:
                m = meeting_of(k.split("__S")[0])
                B[m][0] += items[c][k]
                B[m][1] += 1
                A[m][0] += items[r][k]
                A[m][1] += 1
            res = paired_ratio({m: tuple(v) for m, v in A.items()}, {m: tuple(v) for m, v in B.items()}, rng)
            res["pairs"] = len(common)
            R[s]["paired_utmos"][f"{c}-{r}"] = res
    return R


def pauses() -> dict:
    R = {}
    for s in SETS:
        ps = [p for p in (PR / "pauses" / f"block_pauses_{s}.jsonl", PR4 / "pauses" / f"block_pauses_{s}.jsonl")
              if p.exists()]
        if not ps:
            continue
        R[s] = {}
        for l in (x for p in ps for x in open(p)):
            r = json.loads(l)
            n = r["talker"].rstrip("/").split("/")[-2]
            for sy in SYSS:
                for k in (1, 2):
                    if n == run_name(s, sy, k):
                        R[s].setdefault(sy, {})[f"s{k}"] = {x: r[x] for x in ("blocks", "forced_share", "words_per_block",
                                                                              "share_1w", "block_dur_s", "speech_min",
                                                                              "pauses_all", "pauses_mid_sent")}
    return R


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    rng = np.random.default_rng(20261114)
    res = {"doc": __doc__, "systems": {sy: {s: [run_name(s, sy, 1), run_name(s, sy, 2)] for s in SETS} for sy in SYSS},
           "latency": latency(rng), "speech": speech(rng), "quality": quality(rng), "block_pauses": pauses()}
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
