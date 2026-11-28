# 2026-11-28 en2zh BLEU-vs-COMET analysis (report only, existing test outputs, CPU): qwen3omni python experiments/sst_hsmm_v1/en2zh_analysis/a17c_latin_split_comet.py
# COMET / BLEU diffs vs p3hk on items with and without Latin (en2zh), per system
import json, re, sys, collections, random
sys.path.insert(0, ".")
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import Bleu
bl = Bleu("cmn"); D0 = "eval_outputs/sst_hsmm_v1"; LAT = re.compile(r"[A-Za-z]")
C = collections.defaultdict(dict); X = collections.defaultdict(dict)
for l in open(f"{D0}/pv8/comet/comet.json.rows.jsonl"):
    r = json.loads(l)
    if r["dir"] == "en2zh": C[r["sys"]][r["id"]] = (r["comet_asr"], r["comet_text"])
for l in open(f"{D0}/pv8/comet/dump.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["dir"] == "en2zh": X[r["sys"]][r["id"]] = r
for s in ("p3pfg2", "p3pfe1", "p3pfh3", "p3p2fxff", "p3hk2"):
    ids = [i for i in X["p3hk"] if i in X[s] and i in C[s] and i in C["p3hk"]]
    groups = {"no Latin anywhere": [i for i in ids if not (LAT.search(X["p3hk"][i]["text"]) or LAT.search(X[s][i]["text"]) or LAT.search(X[s][i]["ref"]))],
              "Latin somewhere": [i for i in ids if (LAT.search(X["p3hk"][i]["text"]) or LAT.search(X[s][i]["text"]) or LAT.search(X[s][i]["ref"]))]}
    for g, G in groups.items():
        dca = sum(C[s][i][0] - C["p3hk"][i][0] for i in G) / len(G); dct = sum(C[s][i][1] - C["p3hk"][i][1] for i in G) / len(G)
        ba = bl.score(bl.stats([X[s][i]["asr"] for i in G], [X[s][i]["ref"] for i in G])) - bl.score(bl.stats([X["p3hk"][i]["asr"] for i in G], [X[s][i]["ref"] for i in G]))
        bt = bl.score(bl.stats([X[s][i]["text"] for i in G], [X[s][i]["ref"] for i in G])) - bl.score(bl.stats([X["p3hk"][i]["text"] for i in G], [X[s][i]["ref"] for i in G]))
        print(f"{s:8s} {g:18s} n {len(G):4d} | dCOMET asr {dca:+.4f} text {dct:+.4f} | dBLEU(pooled) asr {ba:+.2f} text {bt:+.2f}")
