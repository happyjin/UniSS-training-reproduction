"""Robust wave 9 scoring (reports/mist_w3po_v1/robust_wave9_anchor_prereg.zh-CN.md incl. 变更 1; a copy of
wave5_m_score.py with the wave-9 arms and cells added): A_wise05 (voice-anchored Thinker, WiSE 0.5) against the S1 base
(+ 2 Thinker-only base reruns), against wave 5's erm_wise05 and, once it exists, against C_wise05 (crosstalk control).
New cells: b_xt = dev (b) turns whose meta "ovl" is true (another speaker's reference turn overlaps the cut), split into
b_xt_in1s (the overlap reaches into the first 1.0 s of the cut, so u = C(first 1 s) is contaminated) and b_xt_after1s.
Rule 1 (unified rule, unchanged from wave 5): A_wise05 vs base: no degradation in every direction x {a, b} x condition,
text BLEU (vs the 3-base mean) and ASR-BLEU (vs S1): point >= -0.3 and CI upper >= 0; and some noisy cell (direction x
{a, b, b_no_crosstalk}) with a text BLEU gain whose CI lower > 0 and point > that cell's base noise floor.
Rule 2 (变更 1): X = erm_wise05: A_wise05 - X text BLEU CI lower > 0 and point > the cell's base noise floor in some
direction x b_xt x {clean, noisy}; and in every direction x {a, b_no_crosstalk} x condition point >= -0.3 or CI upper >= 0.
The same rule against C_wise05 is reported as rule2C when that arm exists (the prereg's original rule 2).
Auxiliary: D1 / D2 LAAL means; LA2 draft flip rate on b_xt (wave9 flips(): a read t_k (not the last) whose draft and the
previous read's draft both extend past the committed prefix n_k and disagree on unit n_k; rate = flips / such pairs).
The original wave-5 docstring follows.

Robust wave 5, main line M scoring (reports/mist_w3po_v1/robust_wave5_prereg.zh-CN.md, incl. the unified rule and
change 2): the merged ERM / Fishr Thinkers against the S1 base (same LA2 chain, same T1c Talker, seed 20261021, 4 shards)
and Fishr against ERM.

Per direction x set (a: CoVoST 2 pairs; b: OB-SI dev single-speaker cuts; b_no_crosstalk) x condition (clean / noisy):
text BLEU and ASR-BLEU (S1 convention: normalize_for_bleu, sacrebleu zh / 13a; wave2_s1_score.Bleu), D1 / D2 LAAL means
(auxiliary, no gate); system differences with a paired bootstrap 95 % CI (2000, seed 0) over units = sentences (a) or
single-speaker streams (b).
Change 3: two Thinker-only reruns of the base (wave5/base_rerun{1,2}) join S1 as 3 base runs; text BLEU is judged
against their mean, and a gain must also exceed the largest pairwise difference of the 3 base runs in that cell.
Primary rule per arm vs base: (1) no BLEU degradation in every direction x {a, b} x condition, for text BLEU and ASR-BLEU:
point >= -0.3 and CI upper >= 0; (2) at least one noisy cell (direction x {a, b, b_no_crosstalk}) with a text BLEU gain
whose CI lower > 0. Secondary (reported): the original G1-G3 and Fishr - ERM.
    <qwen3omni python> experiments/mist_w3po_v1/voicecond/wave5_m_score.py --out eval_outputs/mist_w3po_v1/wave5/M/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond import wave2_s1_score as S  # noqa: E402

ROOTS = {"base": REPO / "eval_outputs/mist_w3po_v1/wave2_S1",
         "erm_wise05": REPO / "eval_outputs/mist_w3po_v1/wave5/M/eval_erm_wise05",
         "A_wise05": REPO / "eval_outputs/mist_w3po_v1/wave9/eval_A_wise05",
         "C_wise05": REPO / "eval_outputs/mist_w3po_v1/wave9/eval_C_wise05"}
RERUNS = {"base_r1": REPO / "eval_outputs/mist_w3po_v1/wave5/base_rerun1",
          "base_r2": REPO / "eval_outputs/mist_w3po_v1/wave5/base_rerun2"}   # change 3: Thinker-only reruns of the base
BASES = ("base", "base_r1", "base_r2")
SEED = "20261021"
NB = 2000


def cell(meta, pers, lang, set_, cond, keep):
    bl = S.Bleu(lang)
    ids = sorted(i for i, m in meta.items() if m["set"] == set_ and m["cond"] == cond and keep(m))
    units = sorted({meta[i]["stream"] for i in ids})
    up = {u: k for k, u in enumerate(units)}
    pu = np.array([up[meta[i]["stream"]] for i in ids])
    st, mean = {}, {}
    for sysn, per in pers.items():
        for kind in ("text", "asr"):
            if kind == "asr" and f"asr{SEED}" not in per[ids[0]]:
                continue
            hyp = [per[i]["hyp"] if kind == "text" else per[i][f"asr{SEED}"] for i in ids]
            x = bl.stats(hyp, [per[i]["ref"] for i in ids])
            U = np.zeros((len(units), x.shape[1]), dtype=np.int64)
            np.add.at(U, pu, x)
            st[(sysn, kind)] = U
        for k in ("d1", "d2"):
            if f"{k}_{SEED}" not in per[ids[0]]:
                continue
            s_, n_ = np.zeros(len(units)), np.zeros(len(units))
            for j, i in enumerate(ids):
                v = per[i][f"{k}_{SEED}"]
                if v is not None:
                    s_[pu[j]] += v
                    n_[pu[j]] += 1
            mean[(sysn, k)] = (s_, n_)
    rng = np.random.default_rng(0)
    W = [np.ones(len(units))] + [np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units)).astype(float)
                                 for _ in range(NB)]

    def val(sysn, k, w):
        if k in ("text", "asr"):
            return bl.score((st[(sysn, k)] * w[:, None]).sum(0))
        s_, n_ = mean[(sysn, k)]
        return float((s_ * w).sum() / max((n_ * w).sum(), 1e-9))
    out = {"n_sentences": len(ids), "n_units": len(units)}
    for k in ("text", "asr", "d1", "d2"):
        have = [sysn for sysn in pers if (sysn, k) in st or (sysn, k) in mean]
        vals = {sysn: [val(sysn, k, w) for w in W] for sysn in have}
        r = {sysn: round(v[0], 3) for sysn, v in vals.items()}
        if k == "text" and all(b in vals for b in BASES):
            vals["base_mean"] = list(np.mean([vals[b] for b in BASES], axis=0))
            r["base_mean"] = round(vals["base_mean"][0], 3)
            r["base_floor"] = round(max(abs(vals[x][0] - vals[y][0]) for i, x in enumerate(BASES) for y in BASES[i + 1:]), 3)
        for x, y in (("erm_wise05", "base"), ("erm_wise05", "base_mean"), ("A_wise05", "base"), ("A_wise05", "base_mean"),
                     ("A_wise05", "erm_wise05"), ("C_wise05", "base"), ("C_wise05", "base_mean"), ("A_wise05", "C_wise05"),
                     ("C_wise05", "erm_wise05")):
            if x in vals and y in vals:
                d = np.array(vals[x]) - np.array(vals[y])
                r[f"{x}-{y}"] = [round(float(d[0]), 3), round(float(np.percentile(d[1:], 2.5)), 3),
                                 round(float(np.percentile(d[1:], 97.5)), 3)]
        out[k] = r
    return out


def ovl_in_first_s(d: str) -> set:
    """dev (b) pairs where another speaker's reference turn overlaps the first 1.0 s of the cut (wave2_s1_prep's rule
    restricted to [start, start + 1 s]); computed on the clean bench condition (same times in both)"""
    from collections import defaultdict
    from experiments.mist_w3po_v1.voicecond.wave2_s1_prep import BENCH
    base = BENCH / "clean" / d
    turns = defaultdict(list)
    for f in sorted((base / "utts").glob("*.jsonl")):
        for u in S.jl(f):
            turns[f.stem].append((u["begin_s"], u["end_s"], u["speaker_id"]))
    out = set()
    for w in S.jl(base / "refs_windows.jsonl"):
        u = w["utts"][0]
        b = w["start_s"]
        e = b + min(1.0, w["dur_ms"] / 1000)
        if any(sp != u["speaker_id"] and tb < e and te > b for tb, te, sp in turns[w["meeting"]]):
            out.add(w["meeting"][2:] + "_" + w["id"].rsplit("_", 1)[1])
    return out


def units(text: str, zh: bool) -> list:
    return [c for c in text if not c.isspace()] if zh else text.split()


def flips(rec: dict, zh: bool):
    """(flip count, comparable pairs) of one item's LA2 reads (drafts include the committed prefix)"""
    dr, bl = rec["drafts"], rec["blocks"]
    nf = npair = 0
    t_end = bl[-1]["write_ms"] if bl and bl[-1].get("final") else None     # the last read = the end of the audio
    for k in range(1, len(dr)):
        t = dr[k]["t"]
        if t_end is not None and t >= t_end:
            continue                                       # the last read (forced commit) is not a LocalAgreement step
        n = len(units("".join(b["text"] for b in bl if b["write_ms"] < t), zh))
        prev, cur = units(dr[k - 1]["draft"], zh), units(dr[k]["draft"], zh)
        if len(prev) > n and len(cur) > n:
            npair += 1
            nf += prev[n] != cur[n]
    return nf, npair


def flip_table(meta, d, ovl) -> dict:
    zh = d.endswith("zh")
    out = {}
    roots = {**ROOTS, **RERUNS}
    for c in ("clean", "noisy"):
        ids = sorted(i for i, m in meta.items() if m["set"] == "b" and m["cond"] == c and m["pair"] in ovl)
        units_ = sorted({meta[i]["stream"] for i in ids})
        up = {u: k for k, u in enumerate(units_)}
        cnt = {}
        for sysn, root in roots.items():
            f = root / f"thinker_{d}" / f"blocks_{d}_w1.jsonl"
            if not f.exists():
                continue
            recs = {r["id"]: r for r in S.jl(f)}
            if not all(i in recs and "drafts" in recs[i] for i in ids):
                continue
            F_ = np.zeros((len(units_), 2))
            for i in ids:
                a_, b_ = flips(recs[i], zh)
                F_[up[meta[i]["stream"]]] += (a_, b_)
            cnt[sysn] = F_
        rng = np.random.default_rng(0)
        W = [np.ones(len(units_))] + [np.bincount(rng.integers(0, len(units_), len(units_)), minlength=len(units_)).astype(float)
                                      for _ in range(NB)]
        rate = {sysn: [float((F_[:, 0] * w).sum() / max((F_[:, 1] * w).sum(), 1e-9)) for w in W] for sysn, F_ in cnt.items()}
        r = {sysn: round(v[0], 4) for sysn, v in rate.items()}
        r["pairs"] = {sysn: int(F_[:, 1].sum()) for sysn, F_ in cnt.items()}
        for x, y in (("A_wise05", "base"), ("A_wise05", "erm_wise05"), ("A_wise05", "C_wise05"), ("erm_wise05", "base")):
            if x in rate and y in rate:
                dd = np.array(rate[x]) - np.array(rate[y])
                r[f"{x}-{y}"] = [round(float(dd[0]), 4), round(float(np.percentile(dd[1:], 2.5)), 4),
                                 round(float(np.percentile(dd[1:], 97.5)), 4)]
        out[c] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    LD = S.laal_defs()
    res = {"systems": {k: str(v.relative_to(REPO)) for k, v in ROOTS.items()}, "seed": SEED}
    gates = {}
    for d in ("en2zh", "zh2en"):
        pers, meta = {}, None
        for sysn, root in ROOTS.items():
            if not (root / f"talker_{d}_s{SEED}" / "asr_speech.jsonl").exists():
                continue
            meta, pers[sysn] = S.load_dir(root, d, [SEED], LD)
        for rn, rr in RERUNS.items():
            bf = rr / f"thinker_{d}" / f"blocks_{d}_w1.jsonl"
            if bf.exists():
                blk = {r["id"]: "".join(b["text"] for b in r["blocks"]) for r in S.jl(bf)}
                assert set(blk) >= set(meta), f"{rn} {d}: missing items"
                pers[rn] = {i: {"hyp": blk[i], "ref": pers["base"][i]["ref"]} for i in meta}
        lang = S.LANG[d]
        ovl = {m["pair"] for m in meta.values() if m["set"] == "b" and m["ovl"]}
        in1 = ovl_in_first_s(d)
        res[d] = {"n_b_xt_in1s_pairs": len(in1 & ovl)}
        for sub, set_, keep in (("a", "a", lambda m: True), ("b", "b", lambda m: True),
                                ("b_no_crosstalk", "b", lambda m, o=ovl: m["pair"] not in o),
                                ("b_xt", "b", lambda m, o=ovl: m["pair"] in o),
                                ("b_xt_in1s", "b", lambda m, o=ovl, q=in1: m["pair"] in o and m["pair"] in q),
                                ("b_xt_after1s", "b", lambda m, o=ovl, q=in1: m["pair"] in o and m["pair"] not in q)):
            res[d][sub] = {c: cell(meta, pers, lang, set_, c, keep) for c in ("clean", "noisy")}
        res[d]["flip_b_xt"] = flip_table(meta, d, ovl)
    for arm in ("A_wise05", "C_wise05", "erm_wise05"):
        if not all(arm in res[d]["a"]["clean"]["text"] for d in ("en2zh", "zh2en")):
            continue
        k, km = f"{arm}-base", f"{arm}-base_mean"
        nodeg = {f"{d}/{sub}/{c}/{m}": res[d][sub][c][m][km if m == "text" else k] for d in ("en2zh", "zh2en")
                 for sub in ("a", "b") for c in ("clean", "noisy") for m in ("text", "asr")}
        nd_ok = all(v[0] >= -0.3 and v[2] >= 0 for v in nodeg.values())
        imp = {f"{d}/{sub}/noisy/text": res[d][sub]["noisy"]["text"][km] + [res[d][sub]["noisy"]["text"]["base_floor"]]
               for d in ("en2zh", "zh2en") for sub in ("a", "b", "b_no_crosstalk")}
        imp_ok = any(v[1] > 0 and v[0] > v[3] for v in imp.values())
        g = {"no_degradation": nd_ok, "no_degradation_failures": {x: v for x, v in nodeg.items() if not (v[0] >= -0.3 and v[2] >= 0)},
             "improvement": imp_ok, "improvement_cells_[d,lo,hi,floor]": {x: v for x, v in imp.items() if v[1] > 0},
             "PASS": nd_ok and imp_ok}
        if arm == "A_wise05":
            for tag, ref in (("rule2", "erm_wise05"), ("rule2C", "C_wise05")):
                kk = f"A_wise05-{ref}"
                if kk not in res["zh2en"]["a"]["clean"]["text"]:
                    continue
                inc = {f"{d}/b_xt/{c}/text": res[d]["b_xt"][c]["text"][kk] + [res[d]["b_xt"][c]["text"]["base_floor"]]
                       for d in ("en2zh", "zh2en") for c in ("clean", "noisy")}
                inc_ok = any(v[1] > 0 and v[0] > v[3] for v in inc.values())
                keep_ = {f"{d}/{sub}/{c}/text": res[d][sub][c]["text"][kk] for d in ("en2zh", "zh2en")
                         for sub in ("a", "b_no_crosstalk") for c in ("clean", "noisy")}
                bad = {x: v for x, v in keep_.items() if not (v[0] >= -0.3 or v[2] >= 0)}
                g[f"{tag}_vs"] = ref
                g[f"{tag}_b_xt_cells_[d,lo,hi,floor]"] = inc
                g[f"{tag}_increment"] = inc_ok
                g[f"{tag}_a_bnoxt_failures"] = bad
                g[f"{tag}_PASS"] = inc_ok and not bad
            g["BOTH_PASS(rule1,rule2)"] = g["PASS"] and g.get("rule2_PASS", False)
        gates[arm] = g
    res["gates"] = gates
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(gates, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
