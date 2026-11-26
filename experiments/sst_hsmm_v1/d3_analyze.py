"""sst_hsmm_v1 prereg change 46 stage 2d: D3 drafter vs the wave17 frontier on edev.
Metrics per variant exactly as p45_analyze (corpus BLEU, mean proxy D2 via p0rl_analyze.playback, gap, pauses).
wave17 frontier: en2zh heal_{0.60..0.75} from p45 (one engine); zh2en base_{0.60..0.75} from d3/dev_w17 (one engine, this change).
Verdict per direction: D3 effective iff some D3 point has D2 <= interp_w17_D2(at its BLEU) - 0.05 s (BLEU inside the wave17 range),
or BLEU >= interp_w17_BLEU(at its D2) + 0.30 (D2 inside the wave17 range).
Selection (pick_h10 rule): reference = wave17 heal_0.75 (en2zh, p45) / base_0.75 (zh2en, dev_w17); among D3 variants with
BLEU >= ref - 0.30 the lowest D2; none -> the highest BLEU. Also: D2 pandora_0.75_0.10 (p45, zh2en) vs the wave17 base-theta frontier.
Writes choice.json with spec per direction ("mode@theta"). Refuses to overwrite.
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

D0 = REPO / "eval_outputs/sst_hsmm_v1"


def metrics(root, d):
    import sacrebleu
    zh = d.endswith("zh")
    lang = "cmn" if zh else "eng"
    items = {r["id"]: r for r in map(json.loads, open(D0 / "edev/data" / f"items_{d}.jsonl", encoding="utf-8"))}
    rows = defaultdict(dict)
    for f in sorted(glob.glob(str(Path(root) / f"run_{d}_*" / f"blocks_{d}_*.jsonl"))):
        spec = re.match(rf"blocks_{d}_(.+)\.jsonl", Path(f).name).group(1)
        for r in map(json.loads, open(f, encoding="utf-8")):
            rows[spec][r["id"]] = r
    ids = [i for i in items if all(i in rows[s] for s in rows)]
    R = {}
    for s, X in rows.items():
        st = [playback(X[i]["blocks"], lang, items[i]["dur_ms"], len(si_units(items[i]["translation"], lang))) for i in ids]
        R[s] = {"bleu": float(sacrebleu.corpus_bleu([X[i]["translation"] for i in ids], [[items[i]["translation"] for i in ids]],
                                                    tokenize="zh" if zh else "13a").score),
                "d2": float(np.nanmean([x[0] for x in st])), "gap": float(np.mean([x[1] for x in st])),
                "pauses": float(np.mean([x[2] for x in st])), "n": len(ids)}
    return R


def interp(xs, ys, x):
    o = np.argsort(xs)
    xs, ys = np.asarray(xs)[o], np.asarray(ys)[o]
    if x < xs[0] or x > xs[-1]:
        return None
    return float(np.interp(x, xs, ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"refuse to overwrite {a.out}")
    p45 = json.load(open(D0 / "p45/p45_result.json"))
    res = {}
    for d, mode in (("en2zh", "heal"), ("zh2en", "base")):
        d3 = metrics(D0 / "d3/dev", d)
        if d == "en2zh":
            w = {k: v for k, v in p45["en2zh"]["variants"].items() if k.startswith("heal_")}
        else:
            w = metrics(D0 / "d3/dev_w17", d)
        ref = w[f"{mode}_0.75"]
        wb, wd = [v["bleu"] for v in w.values()], [v["d2"] for v in w.values()]
        pts, eff = {}, False
        for s, v in d3.items():
            d2_at = interp(wb, wd, v["bleu"])
            b_at = interp(wd, wb, v["d2"])
            dd = None if d2_at is None else v["d2"] - d2_at
            db = None if b_at is None else v["bleu"] - b_at
            pts[s] = {**v, "dD2_vs_w17_at_equal_bleu": dd, "dBLEU_vs_w17_at_equal_d2": db}
            eff |= (dd is not None and dd <= -0.05) or (db is not None and db >= 0.30)
        ok = [s for s in d3 if d3[s]["bleu"] >= ref["bleu"] - 0.30]
        choice = min(ok, key=lambda s: d3[s]["d2"]) if ok else max(d3, key=lambda s: d3[s]["bleu"])
        out = {"d3": pts, "w17": w, "reference": f"w17 {mode}_0.75", "effective": bool(eff), "choice": choice,
               "spec": choice.replace("_", "@")}
        if d == "zh2en":
            pv = p45["zh2en"]["variants"]["pandora_0.75_0.10"]
            d2_at = interp(wb, wd, pv["bleu"])
            out["D2_pandora_vs_theta"] = {"pandora": pv, "w17_d2_at_equal_bleu": d2_at,
                                          "dD2": None if d2_at is None else pv["d2"] - d2_at,
                                          "note": "p45 engine vs dev_w17 engine (different runs)"}
        res[d] = out
        print(d, json.dumps(out), flush=True)
    with open(a.out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
