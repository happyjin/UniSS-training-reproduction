import json, glob, re, sys
sys.path.insert(0, ".")
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import Bleu
import numpy as np, sacrebleu
D0 = "eval_outputs/sst_hsmm_v1"
for d in ("en2zh", "zh2en"):
    B = Bleu("cmn" if d == "en2zh" else "eng")
    for dev in ("cvdev", "edev"):
        ref = {json.loads(l)["id"]: json.loads(l)["translation"] for l in open(f"{D0}/{dev}/data/items_{d}.jsonl")}
        out = {}
        for m in ("pfb0", "pfd1", "pfe1", "pfh3", "pfh1", "pfe1d1", "pfh4"):
            spec = "heal_0.70" if d == "en2zh" else "base_0.70"
            tr = {}
            for f in glob.glob(f"{D0}/{m}/dev/{dev}/run_{d}_*/blocks_{d}_{spec}.jsonl"):
                for l in open(f):
                    r = json.loads(l); tr[r["id"]] = r["translation"]
            if not tr: continue
            ids = sorted(set(tr) & set(ref)); H = [tr[i] for i in ids]; R = [ref[i] for i in ids]
            raw = sacrebleu.corpus_bleu(H, [R], tokenize="zh" if d == "en2zh" else "13a").score
            off = B.score(B.stats(H, R))
            out[m] = (raw, off)
        b = out["pfb0"]
        print(d, dev, " | ".join(f"{m}: raw {v[0]-b[0]:+.2f} official {v[1]-b[1]:+.2f}" for m, v in out.items() if m != "pfb0"), f"| pfb0 raw {b[0]:.2f} off {b[1]:.2f}")
