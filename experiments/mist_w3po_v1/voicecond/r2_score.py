"""R2 scoring (reports/mist_w3po_v1/robust_r2_contrast_prereg.zh-CN.md): point estimates, no bootstrap.
Systems base / enc_wise05 / r2_wise05 (Thinker LA2 blocks under <root>/eval_<sys>/thinker_<d>/blocks_<d>_w1.jsonl) on
<root>/data (devU set u + wave2_S1 set b). Per direction x set {u, b} x cond {clean, noisy}: text BLEU (paper
normalisation: evaluation.text_metrics), length ratio (normalised hyp / ref length), repetition rate (normalised hyp
longer than 3x the reference), text LAAL (s; unit = character for zh, word for en; delay of unit i = write_ms of the block
that wrote it; LAAL with max(|hyp|, |ref|) as in Papi et al.). Rules (written before any number):
  (1) vs enc_wise05: for both directions and both sets, gap (noisy - clean) not worse by more than 0.3, clean BLEU >= -0.3,
      |length ratio - 1| smaller, repetition rate not higher;
  (2) vs base: some direction x set with noisy BLEU > +0.3.
    qwen3omni python experiments/mist_w3po_v1/voicecond/r2_score.py --root eval_outputs/mist_w3po_v1/r2 --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from evaluation.text_metrics import corpus_bleu, normalize_for_bleu  # noqa: E402

LANG = {"en2zh": "cmn", "zh2en": "eng"}
SYS = ("base", "enc_wise05", "r2_wise05")


def units(text: str, lang: str) -> list:
    n = normalize_for_bleu(text, lang)
    return n.split()                                                 # zh: characters (space-joined); en: words


def laal(row: dict, ref: str, lang: str, dur_ms: int) -> float:
    delays = []
    for b in row["blocks"]:
        delays += [b["write_ms"]] * len(units(b["text"], lang))
    if not delays:
        return float("nan")
    ny, nr = len(delays), len(units(ref, lang))
    rate = dur_ms / max(ny, nr, 1)
    tot, tau = 0.0, 0
    for i, dl in enumerate(delays):
        tot += dl - i * rate
        tau = i + 1
        if dl >= dur_ms:
            break
    return tot / tau / 1000.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    res = {"cells": {}}
    for d in ("en2zh", "zh2en"):
        lang = LANG[d]
        meta = {m["id"]: m for m in map(json.loads, open(root / "data" / f"meta_{d}.jsonl"))}
        ref = {r["id"]: r["translation"] for r in map(json.loads, open(root / "data" / f"items_{d}.jsonl"))}
        H = {s: {r["id"]: r for r in map(json.loads, open(root / f"eval_{s}" / f"thinker_{d}" / f"blocks_{d}_w1.jsonl"))}
             for s in SYS}
        for st in ("u", "b"):
            for cond in ("clean", "noisy"):
                ids = sorted(i for i, m in meta.items() if m["set"] == st and m["cond"] == cond)
                cell = {}
                for s in SYS:
                    hyp = [H[s][i]["translation"] for i in ids]
                    rr = [ref[i] for i in ids]
                    bl = corpus_bleu(hyp, rr, language=lang)
                    lh = [len(units(x, lang)) for x in hyp]
                    lr = [len(units(x, lang)) for x in rr]
                    lat = [laal(H[s][i], ref[i], lang, meta[i]["dur_ms"]) for i in ids]
                    lat = [x for x in lat if x == x]
                    cell[s] = {"bleu": round(bl["score"], 2), "len_ratio": round(sum(lh) / max(1, sum(lr)), 3),
                               "rep_rate": round(sum(x > 3 * max(1, y) for x, y in zip(lh, lr)) / len(ids), 4),
                               "laal": round(sum(lat) / max(1, len(lat)), 3), "n": len(ids)}
                res["cells"][f"{d}/{st}/{cond}"] = cell
    C = res["cells"]
    r1, r2 = [], []
    for d in ("en2zh", "zh2en"):
        for st in ("u", "b"):
            c, n = C[f"{d}/{st}/clean"], C[f"{d}/{st}/noisy"]
            gap = {s: n[s]["bleu"] - c[s]["bleu"] for s in SYS}
            chk = {"gap": gap["r2_wise05"] >= gap["enc_wise05"] - 0.3,
                   "clean": c["r2_wise05"]["bleu"] - c["enc_wise05"]["bleu"] >= -0.3,
                   "len": all(abs(x["r2_wise05"]["len_ratio"] - 1) <= abs(x["enc_wise05"]["len_ratio"] - 1) for x in (c, n)),
                   "rep": all(x["r2_wise05"]["rep_rate"] <= x["enc_wise05"]["rep_rate"] for x in (c, n))}
            res.setdefault("gaps", {})[f"{d}/{st}"] = {s: round(v, 2) for s, v in gap.items()}
            if not all(chk.values()):
                r1.append([f"{d}/{st}", chk])
            if n["r2_wise05"]["bleu"] - n["base"]["bleu"] > 0.3:
                r2.append([f"{d}/{st}", round(n["r2_wise05"]["bleu"] - n["base"]["bleu"], 2)])
    res["verdict"] = {"rule1_failures": r1, "rule1_PASS": not r1, "rule2_noisy_gains_vs_base": r2, "rule2_PASS": bool(r2),
                      "effective": not r1 and bool(r2)}
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    for k, v in C.items():
        print(k, " | ".join(f"{s} {v[s]['bleu']:.2f} r{v[s]['len_ratio']:.2f} rep{v[s]['rep_rate']:.3f} L{v[s]['laal']:.2f}" for s in SYS))
    print(json.dumps(res["gaps"]), json.dumps(res["verdict"]))


if __name__ == "__main__":
    main()
