# report-only: test text BLEU (official Bleu, mean of 5 cells) of en2zh thinker outputs, items/refs/cells from the pv8 COMET dump
import json, glob, re, sys, collections
sys.path.insert(0, ".")
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import Bleu
bl = Bleu("cmn"); D0 = "eval_outputs/sst_hsmm_v1"
rows = [json.loads(l) for l in open(f"{D0}/pv8/comet/dump.jsonl", encoding="utf-8")]
ref = {}; cell = {}
for r in rows:
    if r["dir"] == "en2zh" and r["sys"] == "p3hk": ref[r["id"]] = r["ref"]; cell[r["id"]] = r["cell"]
dumptext = {r["id"]: r["text"] for r in rows if r["dir"] == "en2zh" and r["sys"] == "p3hk"}
def th(s): return {r["id"]: r["translation"] for f in glob.glob(f"{D0}/{s}/thinker_en2zh/*.jsonl") for r in map(json.loads, open(f, encoding="utf-8"))}
T = {s: th(s) for s in ("p3hk", "p3pfo1", "p3pfv1")}
print("p3hk thinker == dump text:", sum(T["p3hk"].get(i) == dumptext[i] for i in ref), "/", len(ref))
loop = {i for i, t in T["p3pfv1"].items() if re.search(r"(.{1,6})\1{5,}", t or "")}
cells = sorted(set(cell.values()))
for name, excl in (("all", set()), ("loop-free (p3pfv1 loops removed from every system)", loop)):
    print("==", name)
    for s in T:
        sc = []
        for c in cells:
            ids = [i for i in ref if cell[i] == c and i not in excl and i in T[s]]
            sc.append(bl.score(bl.stats([T[s][i] for i in ids], [ref[i] for i in ids])))
        print(f"  {s:7s} mean {sum(sc)/len(sc):6.2f} | " + " ".join(f"{c} {x:5.2f}" for c, x in zip(cells, sc)))
print("loops in scored items:", len(loop & set(ref)), "of", len(ref))
for s in T:
    t = "".join(T[s][i] for i in ref if i in T[s] and i not in loop)
    print(s, {k: t.count(k) for k in ("觉得", "认为", "这个", "那个", "就是", "了", "呢", "吧")})
t = "".join(ref.values()); print("ref", {k: t.count(k) for k in ("觉得", "认为", "这个", "那个", "就是", "了", "呢", "吧")})
