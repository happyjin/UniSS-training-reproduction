"""portfolio 2026-11-27 (change 55): dev analysis of one method on edev + cvdev against the B0 reference (wave17 + h14, same grid).
Layout: <root>/dev/<devset>/run_<d>_<k>/blocks_<d>_<spec>.jsonl (method) and <ref>/dev/<devset>/... (B0). Metrics per spec as
drafter_analyze.metrics (corpus BLEU, mean proxy D2 via p0rl_analyze.playback) on the items common to all specs of both systems.
Selection (prereg change 55): among the method's specs with BLEU >= B0(mode@0.75) BLEU - 0.30 on BOTH devsets, the lowest mean D2
(over devsets); none -> the highest mean BLEU. Screening (recorded only): at the chosen spec, on BOTH devsets, BLEU >= B0 frontier
BLEU at equal D2 + 0.30, or D2 <= B0 frontier D2 at equal BLEU - 0.10 s (linear interpolation, inside the B0 range).
Change 57 (A): BLEU = the official test BLEU (wave2_s1_score.Bleu, normalize_for_bleu) unless --bleu raw.
    python3 -m experiments.sst_hsmm_v1.pf_analyze --root <method root> --ref eval_outputs/sst_hsmm_v1/pfb0 --out <choice.json>
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
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import Bleu  # noqa: E402

D0 = REPO / "eval_outputs/sst_hsmm_v1"
MODE = {"en2zh": "heal", "zh2en": "base"}
BLEU_KIND = "official"


def load(root, ds, d):
    rows = defaultdict(dict)
    for f in sorted(glob.glob(str(Path(root) / "dev" / ds / f"run_{d}_*" / f"blocks_{d}_*.jsonl"))):
        spec = re.match(rf"blocks_{d}_(.+)\.jsonl", Path(f).name).group(1)
        for r in map(json.loads, open(f, encoding="utf-8")):
            rows[spec][r["id"]] = r
    return rows


def metrics(rows, items, ids, d):
    import sacrebleu
    zh = d.endswith("zh")
    lang = "cmn" if zh else "eng"
    R = {}
    for s, X in rows.items():
        st = [playback(X[i]["blocks"], lang, items[i]["dur_ms"], len(si_units(items[i]["translation"], lang))) for i in ids]
        if BLEU_KIND == "official":          # change 57 A: the test metric (normalize_for_bleu: no punctuation, Latin split)
            bl = Bleu(lang)
            bleu = bl.score(bl.stats([X[i]["translation"] for i in ids], [items[i]["translation"] for i in ids]))
        else:                                # changes 55/56 (raw sacrebleu); kept only to reproduce old choice files
            bleu = sacrebleu.corpus_bleu([X[i]["translation"] for i in ids], [[items[i]["translation"] for i in ids]],
                                         tokenize="zh" if zh else "13a").score
        R[s] = {"bleu": float(bleu),
                "d2": float(np.nanmean([x[0] for x in st])), "n": len(ids)}
    return R


def interp(xs, ys, x):
    o = np.argsort(xs)
    xs, ys = np.asarray(xs)[o], np.asarray(ys)[o]
    if x < xs[0] or x > xs[-1]:
        return None
    return float(np.interp(x, xs, ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--items", default="", help="JSON {devset: items dir}; default edev/data, cvdev/data")
    ap.add_argument("--bleu", choices=("official", "raw"), default="official",
                    help="change 57 A: official = the test BLEU (default); raw = sacrebleu as in changes 55/56")
    a = ap.parse_args()
    global BLEU_KIND
    BLEU_KIND = a.bleu
    if Path(a.out).exists():
        raise SystemExit(f"refuse to overwrite {a.out}")
    idirs = {"edev": D0 / "edev/data", "cvdev": D0 / "cvdev/data"}
    if a.items:
        idirs.update({k: Path(v) for k, v in json.loads(a.items).items()})
    res = {}
    for d in ("en2zh", "zh2en"):
        mode = MODE[d]
        per = {}
        for ds in sorted(p.name for p in (Path(a.root) / "dev").iterdir() if p.is_dir() and p.name in idirs):
            items = {r["id"]: r for r in map(json.loads, open(idirs[ds] / f"items_{d}.jsonl", encoding="utf-8"))}
            M, B = load(a.root, ds, d), load(a.ref, ds, d)
            if not M or not B:
                continue
            ids = [i for i in items if all(i in X for X in list(M.values()) + list(B.values()))]
            per[ds] = {"m": metrics(M, items, ids, d), "b": metrics(B, items, ids, d)}
        if not per:
            continue
        ref = {ds: v["b"].get(f"{mode}_0.75") for ds, v in per.items()}
        specs = sorted(set.intersection(*[set(v["m"]) for v in per.values()]))
        ok = [s for s in specs if all(ref[ds] is not None and per[ds]["m"][s]["bleu"] >= ref[ds]["bleu"] - 0.30 for ds in per)]
        mean = lambda s, k: float(np.mean([per[ds]["m"][s][k] for ds in per]))
        choice = min(ok, key=lambda s: mean(s, "d2")) if ok else max(specs, key=lambda s: mean(s, "bleu"))
        scr, det = True, {}
        for ds, v in per.items():
            bb = [x["bleu"] for x in v["b"].values()]; bd = [x["d2"] for x in v["b"].values()]
            c = v["m"][choice]
            b_at, d_at = interp(bd, bb, c["d2"]), interp(bb, bd, c["bleu"])
            dB = None if b_at is None else c["bleu"] - b_at
            dD = None if d_at is None else c["d2"] - d_at
            same = v["b"].get(choice)
            det[ds] = {"bleu": c["bleu"], "d2": c["d2"], "dBLEU_at_equal_D2": dB, "dD2_at_equal_BLEU": dD,
                       "same_spec_dBLEU": None if same is None else c["bleu"] - same["bleu"],
                       "same_spec_dD2": None if same is None else c["d2"] - same["d2"], "n": c["n"]}
            scr &= (dB is not None and dB >= 0.30) or (dD is not None and dD <= -0.10)
        res[d] = {"spec": choice.replace("_", "@"), "choice": choice, "screen_effective": bool(scr), "at_choice": det,
                  "method": {ds: v["m"] for ds, v in per.items()}, "b0": {ds: v["b"] for ds, v in per.items()}}
        print(d, json.dumps({"spec": res[d]["spec"], "screen": scr, "at_choice": det}), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
