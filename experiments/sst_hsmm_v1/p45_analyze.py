"""sst_hsmm_v1 prereg change 45 analysis: hsc16 variants on edev (<root>/run_<d>_*/blocks_<d>_<spec>.jsonl, spec = mode_theta[_kappa]).
Per direction and variant: corpus BLEU (sacrebleu zh / 13a), mean proxy D2 (s; p0rl_analyze.playback), gap, pauses.
Selection (pick_h10 rule, per direction): reference = base_0.75 of the same engine; among variants with BLEU >= ref - 0.30 the lowest D2.
Verdicts: D1 = cec_0.75 vs heal_0.75 (en2zh): BLEU >= +0.3, or D2 <= -0.05 with BLEU >= -0.1. D2 = best pandora (lowest D2 among
pandora variants with BLEU >= comparator - 0.30) vs heal_0.75 (en2zh) / base_0.75 (zh2en): D2 <= -0.05 and BLEU >= -0.30.
Refuses to overwrite.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
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
    res = {}
    for d in ("en2zh", "zh2en"):
        zh = d.endswith("zh")
        lang = "cmn" if zh else "eng"
        items = {r["id"]: r for r in map(json.loads, open(Path(a.data) / f"items_{d}.jsonl", encoding="utf-8"))}
        rows = defaultdict(dict)
        for f in sorted(glob.glob(str(Path(a.root) / f"run_{d}_*" / f"blocks_{d}_*.jsonl"))):
            spec = re.match(rf"blocks_{d}_(.+)\.jsonl", Path(f).name).group(1)
            for r in map(json.loads, open(f, encoding="utf-8")):
                rows[spec][r["id"]] = r
        if not rows:
            continue
        ids = [i for i in items if all(i in rows[s] for s in rows)]
        R = {}
        for s, X in rows.items():
            st = [playback(X[i]["blocks"], lang, items[i]["dur_ms"], len(si_units(items[i]["translation"], lang))) for i in ids]
            R[s] = {"bleu": float(sacrebleu.corpus_bleu([X[i]["translation"] for i in ids], [[items[i]["translation"] for i in ids]],
                                                        tokenize="zh" if zh else "13a").score),
                    "d2": float(np.nanmean([x[0] for x in st])), "gap": float(np.mean([x[1] for x in st])),
                    "pauses": float(np.mean([x[2] for x in st])), "n": len(ids)}
        ref = R["base_0.75"]
        ok = [s for s in R if R[s]["bleu"] >= ref["bleu"] - 0.30]
        choice = min(ok, key=lambda s: R[s]["d2"]) if ok else None
        out = {"variants": R, "reference": "base_0.75", "choice": choice}
        comp = "heal_0.75" if "heal_0.75" in R else "base_0.75"
        if "cec_0.75" in R:
            db, dd = R["cec_0.75"]["bleu"] - R["heal_0.75"]["bleu"], R["cec_0.75"]["d2"] - R["heal_0.75"]["d2"]
            out["D1"] = {"dbleu": db, "dd2": dd, "effective": bool(db >= 0.3 or (dd <= -0.05 and db >= -0.1))}
        pan = [s for s in R if s.startswith("pandora") and R[s]["bleu"] >= R[comp]["bleu"] - 0.30]
        if pan:
            b = min(pan, key=lambda s: R[s]["d2"])
            db, dd = R[b]["bleu"] - R[comp]["bleu"], R[b]["d2"] - R[comp]["d2"]
            out["D2"] = {"best": b, "comparator": comp, "dbleu": db, "dd2": dd, "effective": bool(dd <= -0.05 and db >= -0.30)}
        else:
            out["D2"] = {"best": None, "effective": False}
        res[d] = out
        print(d, json.dumps(out), flush=True)
    with open(a.out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
