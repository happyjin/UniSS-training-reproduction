"""Manifest + audio for the step-3 training-data review page."""
import json, random
from collections import Counter
from pathlib import Path
import numpy as np, soundfile as sf
ROOT = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS"); PAGE = Path(__file__).resolve().parent
items = {}
for d in ("en2zh", "zh2en"):
    for l in open(ROOT / f"data/unist_simul_thinker_v1/items_{d}.jsonl"):
        r = json.loads(l); items[r["id"]] = r
def ogg(src, dst):
    x, sr = sf.read(src, dtype="float32", always_2d=True); x = x.mean(1)
    if sr != 16000:
        import librosa; x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    pk = float(np.abs(x).max()) or 1.0; x = 0.9 * x / pk
    dst.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(str(dst), "x", 16000, 1, format="OGG", subtype="VORBIS") as fh: fh.write(x)
    return round(len(x) / 16000, 2)
man = {}
# A: the rewrite pilot
pa = [json.loads(l) for l in open(ROOT / "data/si_interp_rewrite_v1/pilot/rewrites.audit.jsonl")]
man["rewrite_summary"] = json.load(open(ROOT / "data/si_interp_rewrite_v1/pilot/rewrites.audit.json"))
rng = random.Random(1); ex = []
for d in ("en2zh", "zh2en"):
    P = [a for a in pa if a["dir"] == d]
    ch = [a for a in P if a["rewrite"].strip() != a["oracle"].strip()]
    for a in rng.sample(ch, 6):
        ex.append({k: a[k] for k in ("dir", "ref", "oracle", "rewrite", "accepted", "len_ratio", "chrf_oracle_ref", "chrf_rewrite_ref")})
man["rewrite_examples"] = ex
# B: the start oracle
S = [json.loads(l) for l in open(ROOT / "data/si_start_oracle_v1/start_oracle.strict2.jsonl")]
man["start_summary"] = json.load(open(ROOT / "data/si_start_oracle_v1/start_oracle.audit.json"))
reasons = Counter((r["dir"], r.get("reject")) for r in S if r.get("reject"))
kept = Counter(r["dir"] for r in S if r["best_L"] > r["lcp"])
man["funnel"] = {d: {"start_reads": sum(1 for r in S if r["dir"] == d),
                     "first_pass_lengthened": man["start_summary"][d]["lengthened"],
                     "kept": kept[d], "rejected": {k[1]: v for k, v in reasons.items() if k[0] == d}} for d in ("en2zh", "zh2en")}
man["rows"] = json.load(open(ROOT / "data/si_thinker_oracle_prefix_v3_startoracle/README.md").read().split("\n\n", 1)[1]) if False else None
rng = random.Random(7); samples = []
for d in ("en2zh", "zh2en"):
    acc = [r for r in S if r["dir"] == d and r["best_L"] > r["lcp"]]
    rej = [r for r in S if r["dir"] == d and r.get("reject")]
    for tag, pool, n in (("采用", acc, 24), ("拒绝", rej, 12)):
        for r in rng.sample(pool, n):
            a = f"audio/{d}_{len(samples):03d}.ogg"; dur = ogg(items[r["id"]]["audio"], PAGE / a)
            samples.append({"dir": d, "tag": tag, "audio": a, "dur": dur, "t": r["t"] / 1000, "draft": r["draft"],
                            "old": (r["draft"].replace(" ", "")[: r["lcp"]] if d == "en2zh" else " ".join(r["draft"].split()[: r["lcp"]])),
                            "new": r["target"] if tag == "采用" else r.get("target_rejected", ""), "reason": r.get("reject"),
                            "full": r["full"], "oracle": r["oracle"], "ref": r["ref"], "c0": r["chrf_oracle"], "c1": r["chrf_full"]})
man["samples"] = samples
(PAGE / "manifest.json").write_text(json.dumps(man, ensure_ascii=False))
print(len(samples), man["funnel"])
