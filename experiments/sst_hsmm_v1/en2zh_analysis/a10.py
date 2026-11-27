import json, glob, re, sys
sys.path.insert(0, ".")
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import Bleu
import sacrebleu
D0 = "eval_outputs/sst_hsmm_v1"
SY = {"w17": f"{D0}/devoff/offline_{{d}}_*.jsonl", "p2ft(E1/H3)": f"{D0}/pv3/offtag/p2ft_{{d}}_*.jsonl", "h1tr(H1)": f"{D0}/pv3/offtag/h1tr_{{d}}_*.jsonl", "h4m(H4)": f"{D0}/pv3/h4off/h4m_{{d}}_*.jsonl"}
for d in ("en2zh", "zh2en"):
    B = Bleu("cmn" if d == "en2zh" else "eng")
    items = {json.loads(l)["id"]: json.loads(l) for l in open(f"eval_outputs/mist_w3po_v1/wave2_S1/data/items_{d}.jsonl")}
    T = {}
    for k, p in SY.items():
        t = {}
        for f in glob.glob(p.format(d=d)):
            for l in open(f):
                r = json.loads(l); t[r["id"]] = r["translation"]
        T[k] = t
    full = [k for k in T if len(T[k]) >= len(T["w17"]) * 0.99]
    ids = sorted(set.intersection(*[set(T[k]) for k in full]) & set(items))
    for sub_name, sub in (("all", ids), ("ref has Latin", [i for i in ids if re.search("[A-Za-z]", items[i]["translation"])]), ("ref no Latin", [i for i in ids if not re.search("[A-Za-z]", items[i]["translation"])])):
        if d == "zh2en" and sub_name != "all": continue
        R = [items[i]["translation"] for i in sub]
        row = []
        for k in full:
            H = [T[k][i] for i in sub]
            off = B.score(B.stats(H, R)); raw = sacrebleu.corpus_bleu(H, [R], tokenize="zh" if d == "en2zh" else "13a").score
            lat = sum(len(re.findall("[A-Za-z]", h)) for h in H) / max(1, sum(len(re.sub(r"\s", "", h)) for h in H)) * 100
            row.append(f"{k}: off {off:5.2f} raw {raw:5.2f}" + (f" latin {lat:4.1f}" if d == "en2zh" else ""))
        print(d, f"{sub_name:13s} n={len(sub):4d} |", " | ".join(row))
