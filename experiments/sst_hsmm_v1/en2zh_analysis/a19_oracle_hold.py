"""2026-11-28 (report only, existing test outputs, CPU): best case of any 'hold where the translation will be reordered' commit
policy on top of a streaming system. For every item whose early-committed text E diverges from the offline ceiling C (a18 cat early),
first divergence at unit p: oracle hold = units of E from p on are not written before t*, the first read (stored drafts) whose draft
agrees with C on units [0, p] (i.e. the draft already has the offline structure at the divergence point); if no read does, t* = the
final read. Optimistic on both axes: the item is then scored as the ceiling (BLEU upper bound) and only E[p:] is delayed (no knock-on
delay of later units; latency lower bound).
Latency proxy per item = mean write time of the output units (final-block units at the final write); reported as the mean over items
of (proxy - p3hk proxy), with p3hk2 (theta lowered) as calibration of the proxy against the official D1/D2 differences.
    PYTHONPATH=. <qwen3omni python> experiments/sst_hsmm_v1/en2zh_analysis/a19_oracle_hold.py
"""
import collections, functools, glob, json, sys
sys.path.insert(0, ".")
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import Bleu
from evaluation.text_metrics import normalize_for_bleu

E0 = "eval_outputs/sst_hsmm_v1"
DUMP = collections.defaultdict(dict)
for l in open(f"{E0}/pv8/comet/dump.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["sys"] == "p3hk": DUMP[r["dir"]][r["id"]] = r


def lcp(a, b):
    j = 0
    while j < len(a) and j < len(b) and a[j] == b[j]: j += 1
    return j


for d in ("en2zh", "zh2en"):
    lang = "cmn" if d == "en2zh" else "eng"; bl = Bleu(lang)
    U = functools.lru_cache(maxsize=None)(lambda t: tuple(normalize_for_bleu(t, lang).split()))
    ceil = {r["id"]: r["translation"] for f in glob.glob(f"{E0}/ceiling/offline_{d}_*.jsonl") for r in map(json.loads, open(f, encoding="utf-8"))}
    items = {i: r for i, r in DUMP[d].items() if i in ceil}
    cells = sorted({r["cell"] for r in items.values()})

    def load(s):
        return {r["id"]: r for f in glob.glob(f"{E0}/{s}/thinker_{d}/blocks_*.jsonl") for r in map(json.loads, open(f, encoding="utf-8"))}

    def proxy(blocks, shift=None):
        """mean write time (s) of output units; shift = (p, t_star): early units with index >= p written no earlier than t_star"""
        t, k = [], 0
        for b in blocks:
            for _ in U(b["text"]):
                w = b["write_ms"]
                if shift and not b["final"] and k >= shift[0]: w = max(w, shift[1])
                t.append(w); k += 1
        return sum(t) / len(t) / 1000 if t else None

    def mbleu(H):
        sc = []
        for c in cells:
            ids = [i for i in items if items[i]["cell"] == c]
            sc.append(bl.score(bl.stats([H[i] for i in ids], [items[i]["ref"] for i in ids])))
        return sum(sc) / len(sc)

    R = load("p3hk"); R2 = load("p3hk2")
    ids = [i for i in items if i in R and i in R2]
    F = {i: "".join(b["text"] for b in R[i]["blocks"]) for i in ids}
    base_b = mbleu(F); base_p = {i: proxy(R[i]["blocks"]) for i in ids}
    p2 = [proxy(R2[i]["blocks"]) - base_p[i] for i in ids if base_p[i] is not None and proxy(R2[i]["blocks"]) is not None]
    b2 = mbleu({i: "".join(b["text"] for b in R2[i]["blocks"]) for i in ids})
    print(f"\n== {d}: items {len(ids)} | p3hk text BLEU {base_b:.2f}")
    print(f"  calibration p3hk2: dBLEU {b2 - base_b:+.2f}  dproxy {sum(p2) / len(p2):+.3f} s  -> {(b2 - base_b) / (sum(p2) / len(p2)):.1f} BLEU per proxy-second")
    hold, nores = {}, 0
    for i in ids:
        early = [b for b in R[i]["blocks"] if not b["final"]]
        e_, c_ = U("".join(b["text"] for b in early)), U(ceil[i])
        if not early or e_ == c_[:len(e_)]: continue
        p = lcp(e_, c_)
        fin = [b["write_ms"] for b in R[i]["blocks"] if b["final"]]
        tstar = None
        for dr in R[i].get("drafts", []):
            u = U(dr["draft"])
            if len(u) > p and u[:p + 1] == c_[:p + 1]:
                tstar = dr["t"]; break
        if tstar is None: tstar = fin[0] if fin else early[-1]["write_ms"]; nores += 1
        hold[i] = (p, tstar)
    dp = [proxy(R[i]["blocks"], hold[i]) - base_p[i] for i in hold]
    tot = sum(dp) / len(ids)
    gain = mbleu({i: (ceil[i] if i in hold else F[i]) for i in ids}) - base_b
    print(f"  oracle hold: items held {len(hold)} (no read agrees before final: {nores}) | dBLEU <= {gain:+.2f} | dproxy >= {tot:+.3f} s over all items "
          f"(held items mean {sum(dp) / max(1, len(dp)):+.3f} s) -> {gain / max(1e-9, tot):.1f} BLEU per proxy-second")
    # by hold length
    q = sorted(dp); print("  hold cost per held item (s): median %.2f q75 %.2f q90 %.2f" % (q[len(q) // 2], q[3 * len(q) // 4], q[9 * len(q) // 10]))
    # budget curve: hold only the cheapest items first (oracle ranking by cost per item)
    order = sorted(hold, key=lambda i: proxy(R[i]["blocks"], hold[i]) - base_p[i])
    for frac in (0.25, 0.5, 0.75):
        S_ = set(order[:int(frac * len(order))])
        g = mbleu({i: (ceil[i] if i in S_ else F[i]) for i in ids}) - base_b
        c = sum(proxy(R[i]["blocks"], hold[i]) - base_p[i] for i in S_) / len(ids)
        print(f"    cheapest {int(frac * 100)}% of holds: dBLEU <= {g:+.2f}  dproxy >= {c:+.3f} s  -> {g / max(1e-9, c):.1f} BLEU/s")
