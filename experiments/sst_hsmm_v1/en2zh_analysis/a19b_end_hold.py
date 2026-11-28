# end-aware hold bound: skip every non-final commit written within W ms before the final read (perfect end-of-source knowledge);
# items whose first divergence (a18) lies in a skipped block are scored as the ceiling (upper bound); cost = all skipped units delayed to the final read
import collections, functools, glob, json, sys
sys.path.insert(0, ".")
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import Bleu
from evaluation.text_metrics import normalize_for_bleu
E0 = "eval_outputs/sst_hsmm_v1"
DUMP = collections.defaultdict(dict)
for l in open(f"{E0}/pv8/comet/dump.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["sys"] == "p3hk": DUMP[r["dir"]][r["id"]] = r
for d in ("en2zh", "zh2en"):
    lang = "cmn" if d == "en2zh" else "eng"; bl = Bleu(lang)
    U = functools.lru_cache(maxsize=None)(lambda t: tuple(normalize_for_bleu(t, lang).split()))
    ceil = {r["id"]: r["translation"] for f in glob.glob(f"{E0}/ceiling/offline_{d}_*.jsonl") for r in map(json.loads, open(f, encoding="utf-8"))}
    items = {i: r for i, r in DUMP[d].items() if i in ceil}; cells = sorted({r["cell"] for r in items.values()})
    R = {r["id"]: r for f in glob.glob(f"{E0}/p3hk/thinker_{d}/blocks_*.jsonl") for r in map(json.loads, open(f, encoding="utf-8"))}
    ids = [i for i in items if i in R]; F = {i: "".join(b["text"] for b in R[i]["blocks"]) for i in ids}
    def mbleu(H):
        sc = []
        for c in cells:
            I = [i for i in ids if items[i]["cell"] == c]; sc.append(bl.score(bl.stats([H[i] for i in I], [items[i]["ref"] for i in I])))
        return sum(sc) / len(sc)
    base = mbleu(F)
    for W in (300, 500, 800, 1200):
        dsum, hit = 0.0, set()
        for i in ids:
            B = R[i]["blocks"]; fin = [b["write_ms"] for b in B if b["final"]]
            if not fin: continue
            tf = fin[0]; early = [b for b in B if not b["final"]]
            e_, c_ = U("".join(b["text"] for b in early)), U(ceil[i])
            div = None
            if e_ != c_[:len(e_)]:
                j = 0
                while j < len(e_) and j < len(c_) and e_[j] == c_[j]: j += 1
                div = j
            k, t_old, t_new = 0, [], []
            for b in B:
                for _ in U(b["text"]):
                    w = b["write_ms"]; t_old.append(w)
                    skip = (not b["final"]) and (tf - w) <= W
                    t_new.append(tf if skip else w)
                    if skip and div is not None and k >= div: hit.add(i)
                    k += 1
            if t_old: dsum += (sum(t_new) - sum(t_old)) / len(t_old) / 1000
        g = mbleu({i: (ceil[i] if i in hit else F[i]) for i in ids}) - base; c = dsum / len(ids)
        print(f"{d} W={W:4d} ms: divergent items caught {len(hit):4d} | dBLEU <= {g:+.2f}  dproxy {c:+.3f} s -> {g / max(1e-9, c):.1f} BLEU/s")
