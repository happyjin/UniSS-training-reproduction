"""Robust line, combination self-check (reports/mist_w3po_v1/robust_combo_plan_20261120.zh-CN.md, section 2): a
combo_si_thinker run with one switch configuration must reproduce the wave whose method it re-implements, up to vLLM's
own nondeterminism. Same yardstick as wave15_selfcheck.py / wave16_tools.py agree: per direction, the share of items whose
blocks (write_ms, text, final) equal those of the reference run; PASS when that share (mean over the references if several)
>= the smallest pairwise blocks-equal share among the 3 base LA2 runs (S1, wave5/base_rerun{1,2}; recomputed here, written
in the plan before any combo run: en2zh 0.9035, zh2en 0.9095). Translation-equal shares are listed too.

Planned pairs (plan section 2):
  SC-N  --second-view none  --assign wave16 assign  (LoRA path)      vs  wave16/eval_ncwi
  SC-C  --second-view enh   --thinker wave14/merged_unist_wise05     vs  wave15/eval_cvla_unist
  (SC-0 --second-view none, base: vs the 3 base runs; only if SC-N or SC-C fails, to localise the fault)
Exit 0 = PASS (every direction), 1 = FAIL.

    python experiments/mist_w3po_v1/voicecond/combo_selfcheck.py --run <root> --ref <root> [--ref <root> ...] --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BASES = {"S1": REPO / "eval_outputs/mist_w3po_v1/wave2_S1", "r1": REPO / "eval_outputs/mist_w3po_v1/wave5/base_rerun1",
         "r2": REPO / "eval_outputs/mist_w3po_v1/wave5/base_rerun2"}


def load(root: Path, d: str) -> dict:
    return {r["id"]: (r["translation"], json.dumps(r["blocks"], ensure_ascii=False))
            for r in map(json.loads, open(root / f"thinker_{d}" / f"blocks_{d}_w1.jsonl", encoding="utf-8"))}


def share(a: dict, b: dict, j: int) -> float:
    assert set(a) == set(b), (len(a), len(b), len(set(a) ^ set(b)))
    return round(sum(a[i][j] == b[i][j] for i in a) / len(a), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ref", action="append", required=True)
    ap.add_argument("--dirs", default="en2zh,zh2en")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    res, ok = {"run": a.run, "refs": a.ref}, True
    for d in a.dirs.split(","):
        base = {k: load(v, d) for k, v in BASES.items()}
        ks = list(base)
        pair = {f"{ks[i]}-{ks[k]}": share(base[ks[i]], base[ks[k]], 1) for i in range(3) for k in range(i + 1, 3)}
        run = load(Path(a.run), d)
        vs = {r: share(run, load(Path(r), d), 1) for r in a.ref}
        tr = {r: share(run, load(Path(r), d), 0) for r in a.ref}
        mean = round(sum(vs.values()) / len(vs), 4)
        thr = min(pair.values())
        res[d] = {"blocks_equal": vs, "mean": mean, "translation_equal": tr, "base_pairwise_blocks_equal": pair,
                  "threshold_min_pairwise": thr, "PASS": mean >= thr}
        ok &= res[d]["PASS"]
    res["PASS"] = ok
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
