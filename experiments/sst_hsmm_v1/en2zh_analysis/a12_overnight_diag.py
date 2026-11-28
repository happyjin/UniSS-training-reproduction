"""overnight 2026-11-28 diagnosis (report only, test outputs already scored): why every round fails en2zh.
per system vs p3hk, both directions, official normalisation (normalize_for_bleu):
  1. text BLEU of the system, of the offline ceiling (wave17 whole-audio, change 20), and agreement BLEU with the ceiling
  2. early (non-final blocks) vs tail: share of units committed early, clipped 1/2-gram precision of early text vs ref and vs ceiling
  3. counterfactual: swap sentences back to p3hk by where the final output first differs from p3hk
     (inside the system's early-committed text vs only in the tail) -> BLEU recovered (text and ASR), cell mean
    PYTHONPATH=. <qwen3omni python> experiments/sst_hsmm_v1/en2zh_analysis/a12_overnight_diag.py
"""
import collections, glob, json, sys
import numpy as np
sys.path.insert(0, ".")
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import Bleu
from evaluation.text_metrics import normalize_for_bleu

E = "eval_outputs/sst_hsmm_v1"
SYS = ["p3hk", "p3hk2", "p3pfk1", "p3pfg2", "p3pfg3", "p3pfr1", "p3pfq1", "p3pfs2", "p3pfe1", "p3p2fxff"]
NAME = {"p3hk": "p3hk", "p3hk2": "p3hk2", "p3pfk1": "HCD", "p3pfg2": "HCG v2", "p3pfg3": "HCG+B1", "p3pfr1": "CCA",
        "p3pfq1": "EGA", "p3pfs2": "SAA", "p3pfe1": "E1", "p3p2fxff": "S3"}
D = collections.defaultdict(dict)
for l in open(f"{E}/pv8/comet/dump.jsonl", encoding="utf-8"):
    r = json.loads(l); D[(r["sys"], r["dir"])][r["id"]] = r


import functools


@functools.lru_cache(maxsize=None)
def norm(t, lang): return normalize_for_bleu(t, lang)


def units(t, lang):
    s = norm(t, lang)
    return s.split() if lang == "eng" else [c for c in s if not c.isspace()]


def grams(c, n): return collections.Counter(tuple(c[k:k + n]) for k in range(len(c) - n + 1))


def clip(h, r, n):
    H, R = grams(h, n), grams(r, n)
    return sum(min(v, R[g]) for g, v in H.items()), sum(H.values())


_ST = {}


def sent_stats(bl, hyp, ref):   # per-sentence sufficient statistics, cached (normalisation is the slow part)
    k = (bl.lang, hyp, ref)
    if k not in _ST:
        _ST[k] = np.array(bl.m._extract_corpus_statistics([norm(hyp, bl.lang)], [[norm(ref, bl.lang)]]), dtype=np.int64)[0]
    return _ST[k]


def cellmean(bl, rows, key, refkey="ref", refs=None):
    cells = collections.defaultdict(list)
    for r in rows: cells[r["cell"]].append(r)
    v = []
    for c, rs in cells.items():
        v.append(bl.score(np.stack([sent_stats(bl, r[key], refs[r["id"]] if refs else r[refkey]) for r in rs])))
    return float(np.mean(v))


for d in ("en2zh", "zh2en"):
    lang = "cmn" if d == "en2zh" else "eng"
    bl = Bleu(lang)
    ceil = {}
    for f in glob.glob(f"{E}/ceiling/offline_{d}_*.jsonl"):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); ceil[r["id"]] = r["translation"]
    base = D[("p3hk", d)]
    ids = sorted(i for i in base if i in ceil)
    print(f"\n===== {d}: items {len(ids)} (of {len(base)}) with ceiling")
    crow = [dict(base[i], text=ceil[i]) for i in ids]
    print(f"ceiling text BLEU {cellmean(bl, crow, 'text'):.2f}")
    print("sys     | textBLEU agreeCeil | early: share  p1ref p2ref p1ceil p2ceil | tail p1ref | swap-back (text/ASR): early-diff n  rec | tail-only n rec | same n")
    for s in SYS:
        A = D[(s, d)]
        B = {}
        for f in glob.glob(f"{E}/{s}/thinker_{d}_*/blocks_*.jsonl"):
            for l in open(f, encoding="utf-8"):
                r = json.loads(l); B[r["id"]] = r
        rows = [A[i] for i in ids]
        tb = cellmean(bl, rows, "text"); ag = cellmean(bl, rows, "text", refs=ceil)
        a = collections.defaultdict(float)
        cls = {}
        for i in ids:
            if i not in B: continue
            r = B[i]
            early = sum((units(b["text"], lang) for b in r["blocks"] if not b["final"]), [])
            tail = sum((units(b["text"], lang) for b in r["blocks"] if b["final"]), [])
            ref, ce = units(A[i]["ref"], lang), units(ceil[i], lang)
            for n in (1, 2):
                m, t = clip(early, ref, n); a[f"er{n}"] += m; a[f"et{n}"] += t
                m, t = clip(early, ce, n); a[f"ec{n}"] += m
            m1, t1 = clip(early, ref, 1); m2, t2 = clip(early + tail, ref, 1); a["tr"] += m2 - m1; a["tt"] += t2 - t1
            a["early"] += len(early); a["all"] += len(early) + len(tail)
            fin, fb = units(A[i]["text"], lang), units(base[i]["text"], lang)
            k = 0
            while k < len(fin) and k < len(fb) and fin[k] == fb[k]: k += 1
            cls[i] = "same" if fin == fb else ("early" if k < len(early) else "tail")
        out = []
        for c in ("early", "tail"):
            sub = {i for i in ids if cls.get(i) == c}
            sw = [base[i] if i in sub else A[i] for i in ids]
            rt = cellmean(bl, sw, "text") - tb
            ra = cellmean(bl, sw, "asr") - cellmean(bl, rows, "asr")
            out.append(f"{len(sub):5d} {rt:+.2f}/{ra:+.2f}")
        f = lambda m, t: 100 * a[m] / max(1, a[t])
        print(f"{NAME[s]:7s} | {tb:6.2f} {ag:6.2f}   |  {a['early']/max(1,a['all']):.3f}  {f('er1','et1'):5.1f} {f('er2','et2'):5.1f} {f('ec1','et1'):5.1f} {f('ec2','et2'):5.1f} | {f('tr','tt'):5.1f} | "
              f"{out[0]} | {out[1]} | {sum(1 for v in cls.values() if v == 'same')}", flush=True)
