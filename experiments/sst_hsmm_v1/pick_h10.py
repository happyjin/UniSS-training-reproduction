"""prereg change 37-4 selection on the Emilia dev (edev): families 'variant:dir:thetas:params:verifier' (thetas '/'-separated),
reference 'dir:theta' (the p3h8 configuration on edev). Rule: dev text BLEU >= reference - 0.30 in BOTH directions, then the lowest
mean (over directions) proxy speech delay (pick_h5.speech); none -> smallest max BLEU deficit. Common items of all configs.
    pick_h10 --fam n2:h22a:0.65/0.70:h14:v,n2:h22b:0.75/0.80:h14:v,n2:h24a:0.65/0.70:h23:v2,... --ref h22b:0.75 --out h24_choice.json
"""
import argparse
import json
import sys
from pathlib import Path

import sacrebleu

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.sst_hsmm_v1.pick_h5 import speech, wt  # noqa: E402

S = REPO / "eval_outputs/sst_hsmm_v1"
DATA = S / "edev/data"
TOL = 0.30


def rd(p):
    return {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = S / a.out
    if out.exists():
        sys.exit(f"拒绝覆盖: {out}")
    cfgs = []
    for x in a.fam.split(","):
        v, fd, ths, prm, ver = x.split(":")
        cfgs += [(v, fd, th, prm, ver) for th in ths.split("/")]
    rdir, rth = a.ref.split(":")
    ref = {d: {r["id"]: r["translation"] for r in map(json.loads, open(DATA / f"items_{d}.jsonl", encoding="utf-8"))} for d in ("en2zh", "zh2en")}
    data = {(fd, th, d): rd(S / fd / f"dev_{th}_{d}" / f"blocks_{d}_w1.jsonl") for _, fd, th, _, _ in cfgs for d in ("en2zh", "zh2en")}
    res = {}
    for d in ("en2zh", "zh2en"):
        zh = d.endswith("zh")
        ks = [k for k in ref[d] if all(k in data[(fd, th, d)] for _, fd, th, _, _ in cfgs)]
        tok = "zh" if zh else "13a"
        for _, fd, th, _, _ in cfgs:
            X = data[(fd, th, d)]
            bl = sacrebleu.corpus_bleu([X[k]["translation"] for k in ks], [[ref[d][k] for k in ks]], tokenize=tok).score
            res[(fd, th, d)] = (bl, wt(X, ks), *speech(X, ks, zh), len(ks))
    table = []
    for v, fd, th, prm, ver in cfgs:
        deficit = max(res[(rdir, rth, d)][0] - res[(fd, th, d)][0] for d in ("en2zh", "zh2en"))
        msd = sum(res[(fd, th, d)][2] for d in ("en2zh", "zh2en")) / 2
        table.append({"variant": v, "fam_dir": fd, "theta": float(th), "params": prm, "verifier": f"eval_outputs/sst_hsmm_v1/{ver}/lora/a1.0000",
                      "deficit": deficit, "mean_speech_delay": msd,
                      **{f"{d}_{k}": res[(fd, th, d)][i] for d in ("en2zh", "zh2en") for i, k in ((0, "bleu"), (2, "speech_delay"), (3, "pauses"), (1, "wt"))},
                      "n_items": [res[(fd, th, d)][4] for d in ("en2zh", "zh2en")]})
        print(json.dumps(table[-1]), flush=True)
    ok = [r for r in table if r["deficit"] <= TOL]
    best = min(ok, key=lambda r: r["mean_speech_delay"]) if ok else min(table, key=lambda r: r["deficit"])
    out.write_text(json.dumps({"best": best, "ref": a.ref, "rule": f"deficit <= {TOL} vs ref on edev, min mean proxy speech delay", "table": table}, indent=1))
    print("best", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
