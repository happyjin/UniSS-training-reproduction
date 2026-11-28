# 2026-11-28 en2zh BLEU-vs-COMET analysis (report only, existing test outputs, CPU): qwen3omni python experiments/sst_hsmm_v1/en2zh_analysis/a17b_latin_homophone_attrib.py
import json, re, sys, collections
sys.path.insert(0, ".")
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import Bleu
from evaluation.text_metrics import normalize_chinese
bl = Bleu("cmn"); D0 = "eval_outputs/sst_hsmm_v1"
rows = [json.loads(l) for l in open(f"{D0}/pv8/comet/dump.jsonl", encoding="utf-8")]
X = collections.defaultdict(dict)
for r in rows:
    if r["dir"] == "en2zh": X[r["sys"]][r["id"]] = r
cells = sorted({r["cell"] for r in X["p3hk"].values()})
def mb(sys_, f, field):
    sc = []
    for c in cells:
        ids = [i for i, r in X[sys_].items() if r["cell"] == c]
        sc.append(bl.score(bl.stats([f(X[sys_][i][field]) for i in ids], [f(X[sys_][i]["ref"]) for i in ids])))
    return sum(sc) / len(sc)
LAT = re.compile(r"[A-Za-z]"); DIG = re.compile(r"[0-9０-９]")
F = {"official": lambda s: s,
     "no Latin": lambda s: LAT.sub("", s),
     "no Latin/digits": lambda s: DIG.sub("", LAT.sub("", s)),
     "+ 他她它 merged": lambda s: re.sub("[她它]", "他", DIG.sub("", LAT.sub("", s))),
     "+ 的得地 merged": lambda s: re.sub("[得地]", "的", re.sub("[她它]", "他", DIG.sub("", LAT.sub("", s)))),
     "+ 在再/做作 merged": lambda s: re.sub("再", "在", re.sub("作", "做", re.sub("[得地]", "的", re.sub("[她它]", "他", DIG.sub("", LAT.sub("", s))))))}
for s in ("B_b", "p3hk", "p3pfg2"):
    print(s)
    for k, f in F.items():
        t, a = mb(s, f, "text"), mb(s, f, "asr")
        print(f"  {k:22s} text {t:6.2f} asr {a:6.2f} loss {t-a:5.2f}")
# Latin stats
for s in ("p3hk", "p3pfg2"):
    nt = sum(len(LAT.findall(r["text"])) for r in X[s].values()); na = sum(len(LAT.findall(r["asr"])) for r in X[s].values())
    nr = sum(len(LAT.findall(r["ref"])) for r in X[s].values()); nc = sum(len(normalize_chinese(r["ref"]).split()) for r in X[s].values())
    print(s, "Latin letters text", nt, "asr", na, "ref", nr, "ref tokens", nc, "items with Latin in text", sum(bool(LAT.search(r["text"])) for r in X[s].values()))
ex = [r for r in X["p3hk"].values() if LAT.search(r["text"])][:8]
for r in ex: print("REF:", r["ref"][:80], "\nTXT:", r["text"][:80], "\nASR:", r["asr"][:80], "\n")
