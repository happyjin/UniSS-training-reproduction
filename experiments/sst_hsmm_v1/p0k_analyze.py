"""sst_hsmm_v1 prereg change 44 (P0-K) analysis: base / snap / heal on edev en2zh (hsc15 shards <root>/run_*/blocks_en2zh_<v>.jsonl).
Per variant: corpus BLEU (sacrebleu zh), mean proxy D2 (s; p0rl_analyze.playback), gap, pauses, share of non-final commits that end
inside a token (rec["mid"]), heal / snap bookkeeping. pass = BLEU - BLEU(base) >= +0.3 and D2 - D2(base) <= +0.03.
Refuses to overwrite.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.sst_hsmm_v1.p0rl_analyze import playback  # noqa: E402
from experiments.voice_onset_v1.playout import si_units  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"refuse to overwrite {a.out}")
    import sacrebleu
    d = "en2zh"
    items = {r["id"]: r for r in map(json.loads, open(Path(a.data) / f"items_{d}.jsonl", encoding="utf-8"))}
    res = {}
    for v in ("base", "snap", "heal"):
        rows = {}
        for f in sorted(glob.glob(str(Path(a.root) / "run_*" / f"blocks_{d}_{v}.jsonl"))):
            for r in map(json.loads, open(f, encoding="utf-8")):
                rows[r["id"]] = r
        if not rows:
            continue
        ids = [i for i in items if i in rows]
        st = [playback(rows[i]["blocks"], "cmn", items[i]["dur_ms"], len(si_units(items[i]["translation"], "cmn"))) for i in ids]
        mids = [x.get("mid") for i in ids for x in rows[i]["drafts"] if "mid" in x]
        res[v] = {"n": len(ids), "bleu": float(sacrebleu.corpus_bleu([rows[i]["translation"] for i in ids], [[items[i]["translation"] for i in ids]], tokenize="zh").score),
                  "d2": float(np.nanmean([s[0] for s in st])), "gap": float(np.mean([s[1] for s in st])), "pauses": float(np.mean([s[2] for s in st])),
                  "mid_share": float(np.mean(mids)) if mids else None, "sent_with_mid": float(np.mean([any(x.get("mid") for x in rows[i]["drafts"]) for i in ids])),
                  "blocks": float(np.mean([len(rows[i]["blocks"]) for i in ids]))}
    for v in ("snap", "heal"):
        if v in res and "base" in res:
            res[v]["dbleu"] = res[v]["bleu"] - res["base"]["bleu"]
            res[v]["dd2"] = res[v]["d2"] - res["base"]["d2"]
            res[v]["pass"] = bool(res[v]["dbleu"] >= 0.3 and res[v]["dd2"] <= 0.03)
    ok = [v for v in ("snap", "heal") if res.get(v, {}).get("pass")]
    if ok:
        best = max(ok, key=lambda v: (round(res[v]["bleu"], 1), -res[v]["d2"]))
        if len(ok) == 2 and abs(res["snap"]["bleu"] - res["heal"]["bleu"]) < 0.1:
            best = min(ok, key=lambda v: res[v]["d2"])
        res["choice"] = best
    else:
        res["choice"] = None
    with open(a.out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
