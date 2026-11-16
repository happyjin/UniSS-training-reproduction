"""v11 P1-4: tables from v11_metrics (scores/), v11_comet (comet/) and v11_asr2 (asr2/) -> summary.json + printed
markdown. Second-ASR hypotheses are built here (asr_hyp rule, refuses overwrite) and scored with v11_metrics.score_ext.
Meeting-level statistics as the paper (Table sig10): paired bootstrap over meetings (10,000 draws) and an exact
sign-flip test on per-meeting differences; BLEU from summed per-meeting n-gram statistics, both TTS seeds pooled.

    python -m experiments.mist_w3po_v1.evaluation.v11_summarize
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.evaluation import v11_metrics as V  # noqa: E402

EO = REPO / "eval_outputs/mist_w3po_v1/v11_metrics"
SC, CO, A2 = EO / "scores", EO / "comet", EO / "asr2"
RNG = np.random.default_rng(20261115)
NB = 10000
TEST = ("ami", "ali", "en")
SETS = ("devz", "deve") + TEST
LANG_EN = {"devz": True, "deve": False, "ami": True, "ali": True, "en": False, "off_en2zh": False, "off_zh2en": True,
           "off_ali": True}


def bleu_of(stats, eng):
    from sacrebleu.metrics import BLEU
    return BLEU(tokenize="13a" if eng else "zh")._compute_score_from_stats(list(stats)).score


def load(key):
    f = SC / f"{key.replace(':', '__')}.json"
    return json.load(open(f)) if f.exists() else None


def pm_stats(res, variant="std", reg="all"):
    return {m: np.array(v[f"{reg}_stats"]) for m, v in res[variant]["per_meeting"].items() if v.get(f"{reg}_stats")}


def signflip(d):
    d = np.asarray(d, float)
    n = len(d)
    if n > 22:
        s = RNG.choice([-1, 1], size=(200000, n))
    else:
        s = np.array(list(itertools.product([-1, 1], repeat=n)))
    obs = abs(d.sum())
    return float((np.abs((s * d).sum(1)) >= obs - 1e-12).mean())


def paired_bleu(A: list[dict], B: list[dict], eng: bool):
    """A, B: lists (seeds) of meeting -> stats; difference B - A of pooled corpus BLEU, bootstrap CI, sign-flip p."""
    ms = sorted(set.intersection(*[set(x) for x in A + B]))
    a = np.array([sum(x[m] for x in A) for m in ms])
    b = np.array([sum(x[m] for x in B) for m in ms])
    diff = bleu_of(b.sum(0), eng) - bleu_of(a.sum(0), eng)
    bs = []
    for _ in range(NB):
        i = RNG.integers(0, len(ms), len(ms))
        bs.append(bleu_of(b[i].sum(0), eng) - bleu_of(a[i].sum(0), eng))
    per = [bleu_of(b[k], eng) - bleu_of(a[k], eng) for k in range(len(ms))]
    return {"diff": round(diff, 2), "ci": [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)],
            "p": round(signflip(per), 4), "meetings": len(ms)}


def comet_pm(key, fld="comet"):
    f = CO / f"{key.replace(':', '__')}.json"
    if not f.exists():
        return None
    r = json.load(open(f))
    s, n = defaultdict(float), defaultdict(int)
    for u, c in zip(r["units"], r[fld]):
        s[u["meeting"]] += c
        n[u["meeting"]] += 1
    return {m: (s[m], n[m]) for m in s}, r["mean"], r.get("mean_dedup")


def paired_comet(A: list[dict], B: list[dict]):
    ms = sorted(set.intersection(*[set(x) for x in A + B]))
    a = np.array([[sum(x[m][0] for x in A), sum(x[m][1] for x in A)] for m in ms])
    b = np.array([[sum(x[m][0] for x in B), sum(x[m][1] for x in B)] for m in ms])
    mean = lambda z: z[:, 0].sum() / z[:, 1].sum()  # noqa: E731
    diff = mean(b) - mean(a)
    bs = []
    for _ in range(NB):
        i = RNG.integers(0, len(ms), len(ms))
        bs.append(mean(b[i]) - mean(a[i]))
    per = b[:, 0] / b[:, 1] - a[:, 0] / a[:, 1]
    return {"diff": round(100 * diff, 2), "ci": [round(100 * float(np.percentile(bs, 2.5)), 2),
                                                 round(100 * float(np.percentile(bs, 97.5)), 2)],
            "p": round(signflip(per), 4), "meetings": len(ms)}


def holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = round(run, 4)
    return out


def asr2_jobs():
    """second-ASR hypothesis files of the test speech runs -> v11_metrics jobs (std scoring only)."""
    import subprocess
    jobs = []
    for s in TEST:
        b = f"{V.TXT[s]}_{V.RT_B}"
        m = f"{V.TXT[s].replace('_b', '_m')}_{V.RT_W}"
        for sysk, base, sp in (("b", b, "__t1c"), ("e2", b, "__rlplem"), ("m", m, "__rlplem")):
            for seed in ("", "_tts2"):
                run = base + sp + seed
                a = A2 / run / "asr2_speech.jsonl"
                h = A2 / run / "hyp_asr2.jsonl"
                if not a.exists():
                    continue
                if not h.exists():
                    subprocess.run([sys.executable, "-m", "experiments.mist_w3po_v1.evaluation.asr_hyp",
                                    str(REPO / V.DEC / run / "hyp.jsonl"), str(a), str(h)], check=True, cwd=REPO)
                jobs.append({"key": f"{s}:{sysk}:asr2{seed or '_tts1'}", "set": s, "dr": V.DR[s], "hyp": str(h),
                             "windows": V.WIN[s], "ref": V.REF[s], "online": True, "relabel": False, "check": ""})
    return jobs


def score_asr2(jobs):
    out = {}
    for j in jobs:
        f = SC / f"{j['key'].replace(':', '__')}.json"
        if not f.exists():
            H = {h["id"]: h for h in map(json.loads, open(j["hyp"], encoding="utf-8"))}
            W = [w for w in map(json.loads, open(REPO / j["windows"], encoding="utf-8")) if w["id"] in H]
            RW = list(map(json.loads, open(REPO / j["ref"], encoding="utf-8")))
            U = V.universe(j["dr"], {w["meeting"] for w in W})
            r = V.score_ext(j["dr"], W, H, RW, U, online=True)
            with open(f, "x", encoding="utf-8") as fh:
                json.dump({"job": j, "std": r}, fh, ensure_ascii=False)
        out[j["key"]] = json.load(open(f))
    return out


def main() -> None:
    S: dict = {"dedup": {}, "blind": {}, "short": {}, "comet": {}, "asr2": {}, "tests": {}}
    # ---- (a) dedup and (b') blind, every row
    for f in sorted(SC.glob("*.json")):
        r = json.load(open(f))
        k = r["job"]["key"]
        if "asr2" in k or "dedup" not in r:
            continue
        S["dedup"][k] = {v: {g: r[v][f"tcpBLEU_{g}"] for g in ("all", "ov", "1spk")} for v in ("std", "dedup", "blind")}
        S["dedup"][k]["dropped_pieces"] = r["dedup"]["dedup_dropped_pieces"]
        S["short"][k] = r["std"]["short_content"]
    # ---- paired tests on dedup / blind text and speech
    T = {}
    for s in SETS:
        eng = LANG_EN[s]
        b, m = load(f"{s}:b:text"), load(f"{s}:m:text")
        for var in ("std", "dedup", "blind"):
            for reg in ("all", "ov"):
                T[f"{s}:text:m-b:{var}:{reg}"] = paired_bleu([pm_stats(b, var, reg)], [pm_stats(m, var, reg)], eng)
        sp = {k: [load(f"{s}:{k}:asr_tts1"), load(f"{s}:{k}:asr_tts2")] for k in ("b", "e2", "m")}
        for var in ("std", "dedup", "blind"):
            for x, y in (("b", "e2"), ("e2", "m"), ("b", "m")):
                T[f"{s}:speech:{y}-{x}:{var}:all"] = paired_bleu([pm_stats(r, var) for r in sp[x]],
                                                                  [pm_stats(r, var) for r in sp[y]], eng)
    for s, eng in (("off_en2zh", False), ("off_zh2en", True), ("off_ali", True)):
        o, rl, c = load(f"{s}:O2"), load(f"{s}:O2-nosd-relabel"), load(f"{s}:C-ft2")
        for var in ("std", "dedup", "blind"):
            for reg in ("all", "ov"):
                T[f"{s}:O2-rel:{var}:{reg}"] = paired_bleu([pm_stats(rl, var, reg)], [pm_stats(o, var, reg)], eng)
                T[f"{s}:O2-cas:{var}:{reg}"] = paired_bleu([pm_stats(c, var, reg)], [pm_stats(o, var, reg)], eng)
    S["tests"]["bleu"] = T
    # ---- (b) short content recall: per-meeting pooled tests for O2 vs relabel / cascade, M vs B
    def short_pm(r, band):
        return {m: v["short"].get(band, [0, 0, 0.0]) for m, v in r["std"]["per_meeting"].items()}
    ST = {}
    for s, pairs in (("off_en2zh", (("O2-nosd-relabel", "O2"), ("C-ft2", "O2"))),
                     ("off_zh2en", (("O2-nosd-relabel", "O2"), ("C-ft2", "O2"))),
                     ("off_ali", (("O2-nosd-relabel", "O2"), ("C-ft2", "O2"))),
                     ("ami", (("b:text", "m:text"),)), ("ali", (("b:text", "m:text"),)), ("en", (("b:text", "m:text"),))):
        for x, y in pairs:
            A, B = load(f"{s}:{x}"), load(f"{s}:{y}")
            for band in ("le1.5_ov", "tiny_ov"):
                pa, pb = short_pm(A, band), short_pm(B, band)
                ms = sorted(m for m in pa if pa[m][1] and pb.get(m, [0, 0])[1])
                a = np.array([pa[m][:2] for m in ms], float)
                bb = np.array([pb[m][:2] for m in ms], float)
                rate = lambda z: z[:, 0].sum() / z[:, 1].sum()  # noqa: E731
                bs = []
                for _ in range(NB):
                    i = RNG.integers(0, len(ms), len(ms))
                    bs.append(rate(bb[i]) - rate(a[i]))
                ST[f"{s}:{y}-{x}:{band}"] = {"a": round(100 * rate(a), 1), "b": round(100 * rate(bb), 1),
                                             "diff": round(100 * (rate(bb) - rate(a)), 1),
                                             "ci": [round(100 * float(np.percentile(bs, 2.5)), 1),
                                                    round(100 * float(np.percentile(bs, 97.5)), 1)],
                                             "p": round(signflip(bb[:, 0] / bb[:, 1] - a[:, 0] / a[:, 1]), 4)}
    S["tests"]["short"] = ST
    # ---- (c) COMET
    C, CT = {}, {}
    for f in sorted(CO.glob("*.json")):
        r = json.load(open(f))
        C[r["key"]] = {"mean": round(100 * r["mean"], 2), "mean_dedup": round(100 * r["mean_dedup"], 2)}
    S["comet"] = C
    for s in SETS:
        g = lambda k: comet_pm(f"{s}:{k}")  # noqa: E731
        b, m = g("b:text"), g("m:text")
        if b and m:
            CT[f"{s}:text:m-b"] = paired_comet([b[0]], [m[0]])
        sp = {k: [g(f"{k}:asr_tts1"), g(f"{k}:asr_tts2")] for k in ("b", "e2", "m")}
        if all(all(sp[k]) for k in sp):
            for x, y in (("b", "e2"), ("e2", "m"), ("b", "m")):
                CT[f"{s}:speech:{y}-{x}"] = paired_comet([z[0] for z in sp[x]], [z[0] for z in sp[y]])
    for s in ("off_en2zh", "off_zh2en", "off_ali"):
        o, rl, c = comet_pm(f"{s}:O2"), comet_pm(f"{s}:O2-nosd-relabel"), comet_pm(f"{s}:C-ft2")
        if o and rl:
            CT[f"{s}:O2-rel"] = paired_comet([rl[0]], [o[0]])
        if o and c:
            CT[f"{s}:O2-cas"] = paired_comet([c[0]], [o[0]])
    fam = {k: v["p"] for k, v in CT.items() if k.split(":")[0] in TEST}
    for k, h in holm(fam).items():
        CT[k]["p_holm9"] = h
    S["tests"]["comet"] = CT
    # ---- (d) second ASR
    jobs = asr2_jobs()
    R2 = score_asr2(jobs)
    AT = {}
    for s in TEST:
        eng = LANG_EN[s]
        sp = {k: [R2.get(f"{s}:{k}:asr2_tts1"), R2.get(f"{s}:{k}:asr2_tts2")] for k in ("b", "e2", "m")}
        if not all(all(v) for v in sp.values()):
            continue
        for k, v in sp.items():
            st = [pm_stats(x) for x in v]
            S["asr2"][f"{s}:{k}"] = round(bleu_of(sum(sum(x.values()) for x in st), eng), 2)
        for x, y in (("b", "e2"), ("e2", "m"), ("b", "m")):
            AT[f"{s}:{y}-{x}"] = paired_bleu([pm_stats(r) for r in sp[x]], [pm_stats(r) for r in sp[y]], eng)
    for k, h in holm({k: v["p"] for k, v in AT.items()}).items():
        AT[k]["p_holm"] = h
    S["tests"]["asr2"] = AT
    # first-ASR pooled BLEU for the same rows (cross-check against the paper's ASR-BLEU)
    S["asr1"] = {}
    for s in TEST:
        for k in ("b", "e2", "m"):
            st = [pm_stats(load(f"{s}:{k}:asr_tts1")), pm_stats(load(f"{s}:{k}:asr_tts2"))]
            S["asr1"][f"{s}:{k}"] = round(bleu_of(sum(sum(x.values()) for x in st), LANG_EN[s]), 2)
    out = EO / "summary_v11_p14.json"
    with open(out, "x", encoding="utf-8") as fh:   # refuses to overwrite
        json.dump(S, fh, indent=1, ensure_ascii=False)
    print(json.dumps({k: S[k] for k in ("comet", "asr1", "asr2")}, ensure_ascii=False)[:3000])
    print(out)


if __name__ == "__main__":
    main()
