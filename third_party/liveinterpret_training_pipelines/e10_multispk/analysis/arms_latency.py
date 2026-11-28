"""Latency / waiting comparison of decoded arms against baseline B on the esports page (no references needed):
  laal     per-window text LAAL (s): word delay = write_ms of its step, |X| = window duration, one shared gamma
           per window (|Y| = max output words over all compared arms incl. B); mean over windows
  first    first write in the window (s); drought = longest stretch without a write inside the window (s)
  gaps3    translation-track silences > 3 s in the 300 s session (count / seconds), from the page timeline
  words    output words; bleu = page silver-reference BLEU (compare_demo_arms.py definition)

    $PYE e10_multispk/analysis/arms_latency.py --page <page/index.html> \
        --raw V3=<dir>+<dir> --raw T=... --out <json>
"""
import argparse
import ast
import glob
import json
import re
import statistics as st
from collections import defaultdict

import sacrebleu

U = "/opt/dlami/nvme/jasonleeeli/projects/UniSS"


def toks(t, zh):
    return re.findall(r"[一-鿿]|[A-Za-z0-9']+", t) if zh else re.findall(r"[A-Za-z0-9']+", t)


def laal(ws, dur, ny, zh):
    d = []
    for wm, t in sorted(ws, key=lambda x: x[0]):
        d += [wm] * len(toks(t, zh))
    if not d:
        return None
    tau = next((k + 1 for k, x in enumerate(d) if x >= dur), len(d))
    return sum(d[k] - k * dur / ny for k in range(tau)) / tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", required=True)
    ap.add_argument("--raw", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    h = open(a.page, encoding="utf-8").read()
    i = h.index("const DATA = ") + len("const DATA = ")
    D = json.loads(h[i: h.index("\n", i)].rstrip().rstrip(";"))
    WIN = {}
    for d in ("zh2en", "en2zh"):
        for w in map(json.loads, open(f"{U}/eval_outputs/mist_w3po_v1/data/windows_game_{d}_c2g06_v1/windows_test_online_{d}.jsonl")):
            WIN[w["id"]] = w
    A = defaultdict(lambda: defaultdict(list))
    arms = []
    for spec in a.raw:
        k, dirs = spec.split("=", 1)
        arms.append(k)
        for dd in dirs.split("+"):
            for f in glob.glob(f"{dd}/part_*.jsonl"):
                for r in map(json.loads, open(f)):
                    A[r["id"]][k] += [(g["write_ms"], g["text"].strip()) for g in r.get("segs") or [] if g["text"].strip()]
    for d in ("zh2en", "en2zh"):
        for r in map(json.loads, open(f"{U}/eval_outputs/mist_w3po_v1/speech/game_b_{d}__t1c/talker_in/blocks.jsonl")):
            wid = r["id"].split("__")[0]
            A[wid]["B"] += [(b["write_ms"], b["text"].strip()) for b in r["blocks"] if b["text"].strip()]
    arms.append("B")
    res = {}
    for s in D["samples"]:
        sid, zh = s["id"], s["dir"] == "en2zh"
        R = {k: defaultdict(list) for k in arms}
        for wid, w in WIN.items():
            if w["meeting"] != sid or any(not A[wid][k] for k in arms):
                continue
            dur = w["dur_ms"]
            ny = max(sum(len(toks(t, zh)) for _, t in A[wid][k]) for k in arms)
            for k in arms:
                xs = A[wid][k]
                R[k]["laal"].append(laal(xs, dur, ny, zh) / 1000)
                R[k]["first"].append(min(x[0] for x in xs) / 1000)
                ts = sorted({0} | {x[0] for x in xs} | {dur})
                R[k]["drought"].append(max(q - p for p, q in zip(ts, ts[1:])) / 1000)
        src = s["src"] if isinstance(s["src"], list) else ast.literal_eval(s["src"])
        ref = ("" if zh else " ").join(x["ref"] for x in src if x.get("ref"))
        row = {}
        for k in arms:
            tg = sorted(s["arms"][k]["tgt"], key=lambda x: x["t"]) if k in s["arms"] else []
            cov, prev, gaps = [], 0.0, []
            for x in tg:
                if cov and x["t"] <= cov[-1][1]:
                    cov[-1][1] = max(cov[-1][1], x["e"])
                else:
                    cov.append([x["t"], x["e"]])
            for x0, x1 in cov:
                if x0 > prev + 3 and x0 < 300:
                    gaps.append(min(x0, 300) - prev)
                prev = max(prev, x1)
            if prev + 3 < 300:
                gaps.append(300 - prev)
            hyp = ("" if zh else " ").join(x["text"] for x in tg)
            row[k] = {"n_win": len(R[k]["laal"]), "laal": round(st.mean(R[k]["laal"]), 2),
                      "first": round(st.mean(R[k]["first"]), 2), "drought": round(st.mean(R[k]["drought"]), 2),
                      "gaps3_n": len(gaps), "gaps3_s": round(sum(gaps)),
                      "words": sum(len(toks(x["text"], zh)) for x in tg),
                      "bleu": round(sacrebleu.corpus_bleu([hyp], [[ref]], tokenize="zh" if zh else "13a").score, 2)}
        res[sid] = row
    keys = ["n_win", "laal", "first", "drought", "gaps3_n", "gaps3_s", "words", "bleu"]
    print("session arm  " + " ".join(f"{c:>8s}" for c in keys))
    for sid, row in res.items():
        for k, r in row.items():
            print(f"{sid:7s} {k:4s} " + " ".join(f"{str(r[c]):>8s}" for c in keys))
    json.dump(res, open(a.out, "x"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
