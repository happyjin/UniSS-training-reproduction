"""change 57 G0: official-BLEU (wave2_s1_score.Bleu) score of whole-audio wave2_S1 dev decodes, model vs wave17 (devoff), on the
ids both have; PASS iff en2zh model >= wave17 - 0.5 (zh2en reported only). Also Latin chars per 100 normalised chars (en2zh).
    python pv4_g0_score.py --out <json> --model "<glob with {d}>"   (refuses to overwrite)"""
import argparse, glob, json, re, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import Bleu  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--model", required=True)
a = ap.parse_args()
if Path(a.out).exists():
    raise SystemExit(f"refuse to overwrite {a.out}")
DATA = REPO / "eval_outputs/mist_w3po_v1/wave2_S1/data"
W17 = str(REPO / "eval_outputs/sst_hsmm_v1/devoff/offline_{d}_*.jsonl")
res = {}
for d in ("en2zh", "zh2en"):
    bl = Bleu("cmn" if d == "en2zh" else "eng")
    ref = {r["id"]: r["translation"] for r in map(json.loads, open(DATA / f"items_{d}.jsonl", encoding="utf-8"))}
    S = {}
    for name, g in (("w17", W17), ("model", a.model)):
        S[name] = {r["id"]: r["translation"] for f in sorted(glob.glob(g.format(d=d))) for r in map(json.loads, open(f, encoding="utf-8"))}
    ids = [i for i in ref if i in S["w17"] and i in S["model"]]
    res[d] = {"n": len(ids)}
    for name in S:
        H = [S[name][i] for i in ids]
        res[d][name] = round(bl.score(bl.stats(H, [ref[i] for i in ids])), 2)
        if d == "en2zh":
            res[d][name + "_latin_per_100"] = round(100 * sum(len(re.findall("[A-Za-z]", h)) for h in H) / max(1, sum(len(re.sub(r"\s", "", h)) for h in H)), 2)
res["pass"] = bool(res["en2zh"]["n"] > 1000 and res["en2zh"]["model"] >= res["en2zh"]["w17"] - 0.5)
json.dump(res, open(a.out, "x"), indent=1)
print(json.dumps(res))
