"""Why is QABC (A0) weaker against LocalAgreement in Zh->En than in En->Zh on RealSI text? Per policy and read window:
LAAL, BLEU and its parts (n-gram precisions, length ratio); agreement with the same Thinker's own near-offline output
(LA2 at a 6 s window: the frozen Thinker for LA2, the adapted one for A0 and LA2-adapted) -- how much early commitment
costs, independent of the reference; filler words per 100 output words; share of output words written at the final read.
Read-only on existing outputs; prints a table and writes the json given.

    PYTHONPATH=. python3 reports/uniss_si_v4/zh2en_diag.py out.json
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "paper/bayes_simul_s2st_v4/scripts")
from endpoints import CJK, EN, make_run, realsi_refs, text_run, _bleu  # noqa: E402
from realsi_text import POLICIES, R  # noqa: E402

FILL = {"zh2en": re.compile(r"\b(um+|uh+|you know|i mean|that is|like this|well|so|then|kind of|sort of|actually)\b", re.I),
        "en2zh": re.compile(r"(嗯|呃|那个|就是|然后|这个|其实|对吧)")}


def files(spec, d):
    out = {}
    for folder, suf in spec:
        pat = re.compile(rf"blocks_{d}_w([\d.]+){re.escape(suf.format(r=R[d]))}\.jsonl$")
        for f in sorted(Path(folder).glob(f"blocks_{d}_w*.jsonl")):
            m = pat.match(f.name)
            if m and float(m.group(1)) not in out:
                out[float(m.group(1))] = f
    return dict(sorted(out.items()))


def corpus(run, BLEU, tok):
    s = list(run["stats"].sum(0))
    b = BLEU(tokenize=tok)._compute_score_from_stats(s)
    return {"bleu": round(b.score, 2), "p": [round(x, 1) for x in b.precisions], "len_ratio": round(s[0] / s[1], 3)}


def main():
    BLEU, _ = _bleu()
    res = {}
    for d in ("en2zh", "zh2en"):
        lang, tok = ("cmn", "zh") if d == "en2zh" else ("eng", "13a")
        unit = CJK if lang == "cmn" else EN
        refs = realsi_refs(d)
        pols = {k: files(POLICIES[k], d) for k in ("LA2", "LA2-adapted", "A0")}
        off = {"LA2": pols["LA2"][6.0], "LA2-adapted": pols["LA2-adapted"][6.0], "A0": pols["LA2-adapted"][6.0]}
        for name, ws in pols.items():
            orecs = {r["id"]: r["translation"] for r in map(json.loads, open(off[name], encoding="utf-8"))}
            for w, f in ws.items():
                recs = [r for r in map(json.loads, open(f, encoding="utf-8")) if r["id"] in refs]
                run = text_run(f, refs, lang)
                row = corpus(run, BLEU, tok) | {"laal": round(float(sum(x for x in run["lag"] if x == x)
                                                                     / sum(1 for x in run["lag"] if x == x)), 3)}
                ids = [r["id"] for r in recs if r["id"] in orecs]
                own = make_run(ids, [r["translation"] for r in recs if r["id"] in orecs], [orecs[i] for i in ids],
                               [float("nan")] * len(ids), lang)
                row["vs_own_offline"] = corpus(own, BLEU, tok)["bleu"]
                n_units = sum(len(unit.findall(r["translation"])) for r in recs)
                row["fill_per100"] = round(100 * sum(len(FILL[d].findall(r["translation"])) for r in recs) / max(n_units, 1), 2)
                row["final_share"] = round(sum(len(unit.findall(b["text"])) for r in recs for b in r["blocks"]
                                               if b.get("final")) / max(n_units, 1), 3)
                res.setdefault(d, {}).setdefault(name, {})[f"{w:g}"] = row
        rf = sum(len(FILL[d].findall(t)) for t, _ in refs.values()) / sum(len(unit.findall(t)) for t, _ in refs.values())
        res[d]["ref_fill_per100"] = round(100 * rf, 2)
    with open(sys.argv[1], "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    for d in res:
        print(f"== {d}  (reference fillers per 100 words: {res[d]['ref_fill_per100']})")
        print("policy        w     LAAL  BLEU  p1/p2/p3/p4            len   vsOwnOff fill/100 final")
        for name in ("LA2", "LA2-adapted", "A0"):
            for w, x in res[d][name].items():
                if x["laal"] <= 3.2:
                    print(f"{name:12s} {w:>4s} {x['laal']:6.2f} {x['bleu']:5.2f} {'/'.join(map(str, x['p'])):22s} "
                          f"{x['len_ratio']:.3f} {x['vs_own_offline']:7.2f} {x['fill_per100']:7.2f} {x['final_share']:.3f}")


if __name__ == "__main__":
    main()
