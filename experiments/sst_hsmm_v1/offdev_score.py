"""change 47+: score whole-audio dev decodes per subset (a = CoVoST/CV read speech, b = UniST-Emilia; clean/noisy).
    python offdev_score.py --out score.json name=glob [name=glob ...]   (glob over jsonl with {"id","translation"}; {d} = direction)
Reports corpus BLEU (zh tokenizer for en2zh, 13a for zh2en) and hyp/ref length ratio on the ids common to all systems."""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import sacrebleu

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "eval_outputs/mist_w3po_v1/wave2_S1/data"

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("systems", nargs="+")
a = ap.parse_args()
if Path(a.out).exists():
    raise SystemExit(f"refuse to overwrite {a.out}")
res = {}
for d in ("en2zh", "zh2en"):
    tok = "zh" if d == "en2zh" else "13a"
    ref = {r["id"]: r["translation"] for r in map(json.loads, open(DATA / f"items_{d}.jsonl", encoding="utf-8"))}
    sys_ = {}
    for spec in a.systems:
        name, g = spec.split("=", 1)
        sys_[name] = {r["id"]: r["translation"] for f in sorted(glob.glob(g.format(d=d))) for r in map(json.loads, open(f, encoding="utf-8"))}
    ids = [i for i in ref if all(i in s for s in sys_.values())]
    cells = defaultdict(list)
    for i in ids:
        cells[i.split("__")[0]].append(i)
    cells["all"] = ids
    for c, L in sorted(cells.items()):
        R = [[ref[i] for i in L]]
        for name, s in sys_.items():
            H = [s[i] for i in L]
            b = sacrebleu.corpus_bleu(H, R, tokenize=tok)
            res.setdefault(d, {}).setdefault(c, {})[name] = {"bleu": round(b.score, 2), "len_ratio": round(b.sys_len / b.ref_len, 4), "n": len(L)}
        print(d, c, json.dumps(res[d][c]), flush=True)
json.dump(res, open(a.out, "x"), indent=1)
