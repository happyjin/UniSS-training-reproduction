"""2026-11-28 (report only, existing test outputs, CPU): where does a streaming system lose text BLEU against the offline ceiling
(wave17 whole-audio greedy, ceiling/offline_<d>_*.jsonl)? Per item (scored test items of the pv8 COMET dump, official normalisation):
  F = final output, E = text of the non-final (early-committed) blocks, C = ceiling, all as normalised units (zh chars / en words).
  cat same     F == C
  cat tail     E is a prefix of C (early commits agree with offline), F != C  -> divergence only in the final read
  cat early    E is not a prefix of C                                      -> divergence inside early-committed text
Counterfactual text BLEU (5-cell mean): replace F by C for the items of one category -> recovered BLEU of that category.
For cat early: position of the first divergence inside its block (units before the block end), block write time / audio length,
divergence type at the first divergence (as a14: local_sub = same-length swap that re-synchronises within 6 units, local_indel,
structural), and whether the final output re-joins the ceiling after it (difflib matched share of the rest).
    PYTHONPATH=. <qwen3omni python> experiments/sst_hsmm_v1/en2zh_analysis/a18_stream_gap.py [sys ...]   (default p3hk)
"""
import collections, difflib, functools, glob, json, sys
sys.path.insert(0, ".")
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import Bleu
from evaluation.text_metrics import normalize_for_bleu

E0 = "eval_outputs/sst_hsmm_v1"
SYS = sys.argv[1:] or ["p3hk"]
DUMP = collections.defaultdict(dict)
for l in open(f"{E0}/pv8/comet/dump.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["sys"] == "p3hk": DUMP[r["dir"]][r["id"]] = r


@functools.lru_cache(maxsize=None)
def U(t, lang): return tuple(normalize_for_bleu(t, lang).split())


def lcp(a, b):
    j = 0
    while j < len(a) and j < len(b) and a[j] == b[j]: j += 1
    return j


def dtype(a, b, j):
    for s in range(1, 13):
        for u in range(0, 7):
            v = s - u
            if 0 <= v <= 6 and len(a[j + u:j + u + 2]) == 2 and a[j + u:j + u + 2] == b[j + v:j + v + 2]:
                return "local_sub" if u == v else "local_indel"
    return "structural"


for d in ("en2zh", "zh2en"):
    lang = "cmn" if d == "en2zh" else "eng"; bl = Bleu(lang)
    ceil = {r["id"]: r["translation"] for f in glob.glob(f"{E0}/ceiling/offline_{d}_*.jsonl") for r in map(json.loads, open(f, encoding="utf-8"))}
    items = {i: r for i, r in DUMP[d].items() if i in ceil}
    cells = sorted({r["cell"] for r in items.values()})
    dur = {}
    for f in glob.glob(f"{E0}/p3hk/data/items_{d}_*.jsonl"):
        for r in map(json.loads, open(f, encoding="utf-8")):
            dur[r["id"]] = r.get("dur_ms") or r.get("duration_ms") or r.get("src_dur_ms")

    def mbleu(H):
        sc = []
        for c in cells:
            ids = [i for i in items if items[i]["cell"] == c]
            sc.append(bl.score(bl.stats([H[i] for i in ids], [items[i]["ref"] for i in ids])))
        return sum(sc) / len(sc)

    for s in SYS:
        rows = {r["id"]: r for f in glob.glob(f"{E0}/{s}/thinker_{d}/blocks_*.jsonl") for r in map(json.loads, open(f, encoding="utf-8"))}
        ids = [i for i in items if i in rows]
        F = {i: "".join(b["text"] for b in rows[i]["blocks"]) for i in ids}
        cat, info = {}, {}
        for i in ids:
            f_, c_ = U(F[i], lang), U(ceil[i], lang)
            blocks = [b for b in rows[i]["blocks"] if not b["final"]]
            e_ = U("".join(b["text"] for b in blocks), lang)
            if f_ == c_: cat[i] = "same"
            elif e_ == c_[:len(e_)]: cat[i] = "tail"
            else:
                cat[i] = "early"; p = lcp(e_, c_)
                ends, acc = [], 0
                for b in blocks:
                    acc += len(U(b["text"], lang)); ends.append((acc, b["write_ms"]))
                k = next(x for x in ends if x[0] > p)
                rest = difflib.SequenceMatcher(None, f_[p:], c_[p:], autojunk=False)
                m = sum(x.size for x in rest.get_matching_blocks())
                info[i] = {"to_end": k[0] - p - 1, "t_rel": k[1] / dur[i] if dur.get(i) else None, "type": dtype(e_, c_, p),
                           "rejoin": m / max(1, len(c_) - p), "early_units": len(e_), "pos": p}
        base = mbleu({i: F[i] for i in ids} | {i: items[i]["text"] for i in items if i not in F})
        ceilb = mbleu({i: ceil[i] for i in items})
        n = collections.Counter(cat.values())
        print(f"\n== {d} {s}: items {len(ids)} | text BLEU {base:.2f}, ceiling {ceilb:.2f}, gap {ceilb - base:.2f}")
        for c in ("same", "tail", "early"):
            H = {i: (ceil[i] if cat.get(i) == c else F[i]) for i in ids}
            print(f"  {c:5s} n {n[c]:4d} ({n[c] / len(ids):.3f})  swap->ceiling recovers {mbleu(H) - base:+.2f}")
        if info:
            V = list(info.values())
            q = collections.Counter(min(v["to_end"], 5) for v in V)
            print("  early: units before block end at 1st divergence (0 = last unit of the block; 5 = >=5):",
                  {k: round(q[k] / len(V), 3) for k in sorted(q)})
            print("  early: divergence type", {k: round(v / len(V), 3) for k, v in collections.Counter(v["type"] for v in V).items()})
            tr = [v["t_rel"] for v in V if v["t_rel"] is not None]
            if tr:
                tr.sort(); print(f"  early: block write time / audio length  median {tr[len(tr) // 2]:.2f}  q25 {tr[len(tr) // 4]:.2f}  q75 {tr[3 * len(tr) // 4]:.2f}")
            rj = sorted(v["rejoin"] for v in V); print(f"  early: final rejoins ceiling after divergence (matched share of rest) median {rj[len(rj) // 2]:.2f}")
            # recovered BLEU by divergence class
            for key, fn in (("frontier (last unit of block)", lambda v: v["to_end"] == 0), ("1-2 units before end", lambda v: 1 <= v["to_end"] <= 2),
                            (">=3 units before end", lambda v: v["to_end"] >= 3), ("local_sub", lambda v: v["type"] == "local_sub"),
                            ("local_indel", lambda v: v["type"] == "local_indel"), ("structural", lambda v: v["type"] == "structural")):
                S_ = {i for i, v in info.items() if fn(v)}
                H = {i: (ceil[i] if i in S_ else F[i]) for i in ids}
                print(f"    {key:30s} n {len(S_):4d}  swap->ceiling recovers {mbleu(H) - base:+.2f}")
        # check: items with no early commit at all but F != C (pure final-read discrepancy)
        ne = [i for i in ids if not U("".join(b["text"] for b in rows[i]["blocks"] if not b["final"]), lang)]
        nd = [i for i in ne if U(F[i], lang) != U(ceil[i], lang)]
        print(f"  items without early commits {len(ne)}, of which final != ceiling {len(nd)}")
