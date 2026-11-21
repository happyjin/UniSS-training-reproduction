"""Why does AbS lower ASR-BLEU? CPU-only analysis of existing S1-dev outputs (no new runs).
Talker fidelity = edit error of the ASR transcript against the Thinker text the Talker was asked to speak
(chars for cmn, words for eng), which removes translation quality from the comparison."""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
sys.path.insert(0, str(REPO))
from evaluation.text_metrics import normalize_for_bleu  # noqa: E402
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import Bleu  # noqa: E402

E = REPO / "eval_outputs/mist_w3po_v1"
LANG = {"en2zh": "cmn", "zh2en": "eng"}


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def runs(d):
    return {("B", 1): E / f"wave2_S1/talker_{d}_s20261021", ("V1", 1): E / f"wave3_combos/V/v1/talker_{d}",
            ("oracle", 1): E / f"oracle_voice/talker_{d}", ("AbS", 1): E / f"b_abs/talker_abs_{d}",
            ("AbS-C", 1): E / f"abs_c/talker_absc_{d}_s20261021",
            ("B", 2): E / f"b_abs_seed2/talker_B_{d}", ("V1", 2): E / f"b_abs_seed2/talker_v1_{d}",
            ("AbS", 2): E / f"b_abs_seed2/talker_abs_{d}", ("AbS-C", 2): E / f"abs_c/talker_absc_{d}_s20261022"}


def toks(x, lang):
    x = normalize_for_bleu(x or "", lang)
    return [c for c in x if not c.isspace()] if lang == "cmn" else x.split()


def edit(a, b):
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def dims(i):
    return [(i >> (2 * k)) & 3 for k in range(6)]


def nflip(a, b):
    return sum(x != y for p, q in zip(a, b) for x, y in zip(dims(p), dims(q)))


out = {}
for d in ("en2zh", "zh2en"):
    lang = LANG[d]
    bl = Bleu(lang)
    R = {}
    for (arm, s), p in runs(d).items():
        recs = {r["id"]: r for r in jl(p / "results.jsonl")}
        asr = {r["id"]: r.get("asr_text", "") for r in jl(p / "asr_speech.jsonl")}
        R[(arm, s)] = (recs, asr)
    ids = sorted(set.intersection(*[set(v[0]) for v in R.values()]))
    th = R[("V1", 1)][0]
    # Thinker text must be identical across arms (same text blocks)
    same = np.mean([all(R[k][0][i]["translated_text"] == th[i]["translated_text"] for k in R) for i in ids])
    srch = {}
    for f in sorted((E / "abs_c/search").glob(f"search_{d}_*.jsonl")):
        for r in jl(f):
            srch[r["id"]] = r
    tokA = {r["id"]: r["ids"] for r in jl(E / f"b_abs/tokens/tokens_abs_{d}.jsonl")}
    tokC = {r["id"]: r["ids"] for r in jl(E / f"abs_c/tokens/tokens_absc_{d}.jsonl")}
    per = {}
    for i in ids:
        t = toks(th[i]["translated_text"], lang)
        row = {"cell": i.split("__")[0], "n": max(len(t), 1)}
        for k, (recs, asr) in R.items():
            row[k] = edit(toks(asr.get(i, ""), lang), t)
        row["st"] = {k: bl.stats([R[k][1].get(i, "")], [th[i]["translation_ref"]])[0] for k in R}
        sr = srch.get(i)
        row["flipA"] = nflip(tokA[i], sr["ids_v1"]) if sr else None
        row["flipC"] = nflip(tokC[i], sr["ids_v1"]) if sr else None
        row["dstoiA"] = (sr["stoi_abs"] - sr["stoi_v1"]) if sr else None
        row["dstoiC"] = (sr["stoi_best"] - sr["stoi_v1"]) if sr else None
        per[i] = row

    def ter(rows, k):
        return 100 * sum(r[k] for r in rows) / sum(r["n"] for r in rows)

    def bleu(rows, k):
        return bl.score(np.sum([r["st"][k] for r in rows], 0))

    def m2(f, rows, arm):  # mean over the seeds available
        ks = [k for k in R if k[0] == arm]
        return np.mean([f(rows, k) for k in ks])

    cells = {}
    for c in sorted({r["cell"] for r in per.values()}):
        rows = [r for r in per.values() if r["cell"] == c]
        cells[c] = {"n": len(rows), **{f"TER_{a}": round(m2(ter, rows, a), 2) for a in ("B", "V1", "oracle", "AbS", "AbS-C")},
                    **{f"BLEU_{a}": round(m2(bleu, rows, a), 2) for a in ("B", "V1", "oracle", "AbS", "AbS-C")},
                    "TER_V1_s1-s2": round(ter(rows, ("V1", 1)) - ter(rows, ("V1", 2)), 2),
                    "TER_B_s1-s2": round(ter(rows, ("B", 1)) - ter(rows, ("B", 2)), 2)}
    # dose-response: terciles of FSQ dims flipped (AbS vs V1), noisy + clean rows of judged cells
    def bins(key, arm, rows_all):
        rows_all = [r for r in rows_all if r[key] is not None]
        q = np.quantile([r[key] for r in rows_all], [1 / 3, 2 / 3])
        res = []
        for lo, hi, name in ((-1e9, q[0], "low"), (q[0], q[1], "mid"), (q[1], 1e9, "high")):
            rows = [r for r in rows_all if lo < r[key] <= hi] if name != "low" else [r for r in rows_all if r[key] <= q[0]]
            dT = m2(ter, rows, arm) - m2(ter, rows, "V1")
            dB = m2(bleu, rows, arm) - m2(bleu, rows, "V1")
            nz = abs(ter(rows, ("V1", 1)) - ter(rows, ("V1", 2)))
            res.append({"bin": name, "range": [round(float(min(r[key] for r in rows)), 3), round(float(max(r[key] for r in rows)), 3)],
                        "n": len(rows), "dTER": round(dT, 2), "dBLEU": round(dB, 2), "V1_seed_gap_TER": round(nz, 2)})
        return res
    judged = [r for r in per.values() if r["cell"] in ("a_noisy", "a_clean", "b_noisy", "b_clean")]
    noisy = [r for r in judged if "noisy" in r["cell"]]
    out[d] = {"n_items": len(ids), "thinker_text_identical": float(same), "cells": cells,
              "flipA_mean": round(float(np.mean([r["flipA"] for r in judged])), 1),
              "flipC_mean": round(float(np.mean([r["flipC"] for r in judged])), 1),
              "AbS_by_flips(all judged)": bins("flipA", "AbS", judged),
              "AbS_by_flips(noisy)": bins("flipA", "AbS", noisy),
              "AbS_by_dSTOI(noisy)": bins("dstoiA", "AbS", noisy),
              "AbSC_by_flips(all judged)": bins("flipC", "AbS-C", judged)}
print(json.dumps(out, ensure_ascii=False, indent=1))
