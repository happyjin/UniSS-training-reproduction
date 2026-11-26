"""prereg change 35 selection = pick_h6.py with a theta grid per family (default 0.80/0.90/0.95/0.98) and the family dir in the result:
    pick_h8 --fam r1:h19r1:0.30/0.60/1.00/1.50,n2:h19n2:0.50/0.60/0.70/0.75,n1:h15n1,n2:h15n2 --params h14 --out h19_choice.json
Was: prereg change 33 selection = pick_h5.py with the families / params / output from the command line:
Was: prereg change 32 selection (HSC5 families n1 / n2 / s1 / s2 on params h10, dev outputs h11<fam>/dev_<theta>_<d>):
among dev configs whose dev text BLEU >= HSC2 dev (h4/dev_0.80) - 0.30 in BOTH directions (the user accepts BLEU equal to or
slightly below HSC2), take the lowest mean (over directions) proxy speech delay; none -> smallest max BLEU deficit.
Proxy speech delay: per unit, speech start = max(block write time, playback queue end), units of 0.25 s (zh char) / 0.35 s
(en word), mean over the item's units, then over items. Proxy in-sentence pauses (queue idle > 0.3 s before a later block) and
the char-weighted write time are recorded for reference. Common dev items. Writes eval_outputs/sst_hsmm_v1/h5_choice.json."""
import argparse
import json
import sys
from pathlib import Path

import sacrebleu

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import units  # noqa: E402

S = REPO / "eval_outputs/sst_hsmm_v1"
DATA = REPO / "eval_outputs/mist_w3po_v1/wave2_S1/data"
THETAS = ("0.80", "0.90", "0.95", "0.98")
TOL = 0.30


def rd(p):
    return {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}


def wt(X, ks):
    s = 0.0
    for k in ks:
        bb = [b for b in X[k]["blocks"] if b["text"].strip()]
        L = sum(len(b["text"]) for b in bb) or 1
        s += sum(b["write_ms"] * len(b["text"]) for b in bb) / L
    return s / len(ks) / 1000


def speech(X, ks, zh):
    us = 0.25 if zh else 0.35
    tot_d, tot_p = 0.0, 0
    for k in ks:
        q, starts, pauses, first = 0.0, [], 0, True
        for b in X[k]["blocks"]:
            u = units(b["text"].strip(), zh)
            if not u:
                continue
            w = b["write_ms"] / 1000
            if not first and w > q + 0.3:
                pauses += 1
            q = max(q, w); first = False
            for _ in u:
                starts.append(q); q += us
        tot_d += sum(starts) / max(len(starts), 1); tot_p += pauses
    return tot_d / len(ks), tot_p / len(ks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    FAM = {}
    for x in a.fam.split(","):
        p = x.split(":")
        FAM[p[1]] = (p[0], p[2].split("/") if len(p) > 2 else list(THETAS))
    PARAMS = a.params
    out = S / a.out
    if out.exists():
        sys.exit(f"拒绝覆盖: {out}")
    ref = {d: {r["id"]: r["translation"] for r in map(json.loads, open(DATA / f"items_{d}.jsonl", encoding="utf-8"))} for d in ("en2zh", "zh2en")}
    base = {d: rd(S / "h4" / f"dev_0.80_{d}" / f"blocks_{d}_w1.jsonl") for d in ("en2zh", "zh2en")}
    cfgs = [(f, th) for f in FAM for th in FAM[f][1]]
    data = {(v, th, d): rd(S / v / f"dev_{th}_{d}" / f"blocks_{d}_w1.jsonl") for v, th in cfgs for d in ("en2zh", "zh2en")}
    res = {}
    for d in ("en2zh", "zh2en"):
        zh = d.endswith("zh")
        ks = [k for k in base[d] if all(k in data[(v, th, d)] for v, th in cfgs)]
        tok = "zh" if zh else "13a"
        bl = lambda X: sacrebleu.corpus_bleu([X[k]["translation"] for k in ks], [[ref[d][k] for k in ks]], tokenize=tok).score
        res[("HSC2", "0.80", d)] = (bl(base[d]), wt(base[d], ks), *speech(base[d], ks, zh), len(ks))
        for v, th in cfgs:
            X = data[(v, th, d)]
            res[(v, th, d)] = (bl(X), wt(X, ks), *speech(X, ks, zh), len(ks))
    table = []
    for v, th in cfgs:
        deficit = max(res[("HSC2", "0.80", d)][0] - res[(v, th, d)][0] for d in ("en2zh", "zh2en"))
        msd = sum(res[(v, th, d)][2] for d in ("en2zh", "zh2en")) / 2
        table.append({"variant": FAM[v][0], "fam_dir": v, "theta": float(th), "params": PARAMS, "deficit": deficit, "mean_speech_delay": msd,
                      **{f"{d}_bleu": res[(v, th, d)][0] for d in ("en2zh", "zh2en")},
                      **{f"{d}_speech_delay": res[(v, th, d)][2] for d in ("en2zh", "zh2en")},
                      **{f"{d}_pauses": res[(v, th, d)][3] for d in ("en2zh", "zh2en")},
                      **{f"{d}_wt": res[(v, th, d)][1] for d in ("en2zh", "zh2en")}, "n_items": [res[(v, th, d)][4] for d in ("en2zh", "zh2en")]})
        print(json.dumps(table[-1]), flush=True)
    ok = [r for r in table if r["deficit"] <= TOL]
    best = min(ok, key=lambda r: r["mean_speech_delay"]) if ok else min(table, key=lambda r: r["deficit"])
    hs = {d: dict(zip(("bleu", "wt", "speech_delay", "pauses", "n"), res[("HSC2", "0.80", d)])) for d in ("en2zh", "zh2en")}
    out.write_text(json.dumps({"best": best, "rule": f"deficit <= {TOL} vs HSC2 dev, min mean proxy speech delay", "hsc2_dev": hs, "table": table}, indent=1))
    print("best", json.dumps(best), "\nhsc2", json.dumps(hs), flush=True)


if __name__ == "__main__":
    main()
