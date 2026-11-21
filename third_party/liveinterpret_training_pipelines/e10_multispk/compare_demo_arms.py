"""Per-session comparison of the versions (arms) on an esports listening page (add_demo_arm.py output):
  bleu        session text BLEU of the arm's pieces against the page's silver references (src[].ref; machine refs)
  one_write   share of windows whose text was all written at one step (raw decodes; None for B)
  silent      share of the 300 s where the translation track is silent; gaps5 = number of gaps > 5 s
  dup         adjacent identical pieces / pieces
  sw_min      speaker-label switches per minute between consecutive pieces; n_spk = distinct labels used

    $PYE e10_multispk/compare_demo_arms.py --page e10_multispk/output/demo_game_v3/page/index.html \
        --raw V3=e10_multispk/output/demo_game_v3/raw_zh2en+e10_multispk/output/demo_game_v3/raw_en2zh \
        --raw E10=e10_multispk/output/demo_game_v1/raw_zh2en+e10_multispk/output/demo_game_v1/raw_zh2en_r2+... \
        --out e10_multispk/output/demo_game_v3/compare.json
"""
import argparse
import ast
import glob
import json
from collections import defaultdict

import sacrebleu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", required=True)
    ap.add_argument("--raw", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    h = open(a.page, encoding="utf-8").read()
    i = h.index("const DATA = ") + len("const DATA = ")
    D = json.loads(h[i: h.index("\n", i)].rstrip().rstrip(";"))
    raws = {}
    for spec in a.raw:
        k, dirs = spec.split("=", 1)
        R = {}
        for d in dirs.split("+"):
            for f in glob.glob(f"{d}/part_*.jsonl"):
                for line in open(f, encoding="utf-8"):
                    r = json.loads(line)
                    R[r["id"]] = r
        raws[k] = R
    res = defaultdict(dict)
    for s in D["samples"]:
        src = s["src"] if isinstance(s["src"], list) else ast.literal_eval(s["src"])
        zh = s["dir"] == "en2zh"
        ref = ("" if zh else " ").join(x["ref"] for x in src if x.get("ref"))
        for k in D["arm_order"]:
            tg = sorted(s["arms"][k]["tgt"], key=lambda x: x["t"])
            hyp = ("" if zh else " ").join(x["text"] for x in tg)
            b = sacrebleu.corpus_bleu([hyp], [[ref]], tokenize="zh" if zh else "13a").score
            cov = []
            for x in tg:
                if cov and x["t"] <= cov[-1][1]:
                    cov[-1][1] = max(cov[-1][1], x["e"])
                else:
                    cov.append([x["t"], x["e"]])
            T, prev, gaps, speech = 300.0, 0.0, [], 0.0
            for x0, x1 in cov:
                if x0 < T:
                    gaps.append(x0 - prev)
                    speech += min(x1, T) - x0
                prev = x1
            gaps.append(max(0.0, T - prev))
            dup = sum(1 for p, q in zip(tg, tg[1:]) if p["text"] == q["text"])
            sw = sum(1 for p, q in zip(tg, tg[1:]) if p["spk"] != q["spk"])
            row = {"bleu": round(b, 2), "silent": round(1 - speech / T, 3), "gaps5": sum(g > 5 for g in gaps),
                   "max_gap": round(max(gaps), 1), "dup": round(dup / max(1, len(tg)), 3),
                   "sw_min": round(sw / 5.0, 1), "n_spk": len({x["spk"] for x in tg}), "pieces": len(tg)}
            R = raws.get(k)
            if R:
                ws = [r for r in R.values() if r["id"].split("_")[0] == s["id"]]
                one = sum(1 for r in ws if len({g["write_ms"] for g in r.get("segs", []) if g["text"].strip()}) == 1)
                row["one_write"] = round(one / max(1, len(ws)), 2)
                row["cap256"] = sum(1 for r in ws if r.get("stop") == "max_text_tokens")
                row["guard"] = sum(int(r.get("n_guard") or 0) for r in ws)
            res[s["id"]][k] = row
    keys = ["bleu", "one_write", "silent", "gaps5", "max_gap", "dup", "cap256", "sw_min", "n_spk"]
    print("session arm   " + " ".join(f"{c:>9s}" for c in keys))
    for sid, arms in res.items():
        for k, row in arms.items():
            print(f"{sid:7s} {k:5s} " + " ".join(f"{str(row.get(c, '-')):>9s}" for c in keys))
    json.dump(res, open(a.out, "x"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
