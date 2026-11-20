"""Robust wave 15 CV-LA self-check (reports/mist_w3po_v1/robust_wave15_cvla_prereg.zh-CN.md): the --view1 same run (v1 =
v0's own audio) must reproduce LA2 up to vLLM's own nondeterminism. Per direction: the share of items whose blocks
(write_ms, text, final) equal those of each of the 3 base LA2 runs (S1, wave5/base_rerun{1,2}); PASS when the mean of
the 3 shares >= the smallest pairwise share among the 3 base runs (written before the run: en2zh 0.9035, zh2en 0.9095,
recomputed here). Also listed: translation-equal shares and the run's own share of non-final reads with d1 != d0.
Exit 0 = PASS (both directions), 1 = FAIL.
    python experiments/mist_w3po_v1/voicecond/wave15_selfcheck.py --run eval_outputs/mist_w3po_v1/wave15/eval_same_base \
        --out eval_outputs/mist_w3po_v1/wave15/selfcheck.json
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
    assert set(a) == set(b)
    return round(sum(a[i][j] == b[i][j] for i in a) / len(a), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    res, ok = {}, True
    for d in ("en2zh", "zh2en"):
        base = {k: load(v, d) for k, v in BASES.items()}
        run = load(Path(a.run), d)
        ks = list(base)
        pair = {f"{ks[i]}-{ks[k]}": share(base[ks[i]], base[ks[k]], 1) for i in range(3) for k in range(i + 1, 3)}
        vs = {k: share(run, base[k], 1) for k in ks}
        mean = round(sum(vs.values()) / 3, 4)
        st = json.load(open(Path(a.run) / f"thinker_{d}" / f"blocks_{d}_w1.cvla.json"))
        r = {"blocks_equal_same_vs_base": vs, "mean": mean, "base_pairwise_blocks_equal": pair,
             "threshold_min_pairwise": min(pair.values()),
             "translation_equal_same_vs_base": {k: share(run, base[k], 0) for k in ks},
             "base_pairwise_translation_equal": {f"{ks[i]}-{ks[k]}": share(base[ks[i]], base[ks[k]], 0)
                                                 for i in range(3) for k in range(i + 1, 3)},
             "nonfinal_reads": st["reads_nonfinal"], "v1_differs_share": round(st["v1_differs"] / max(1, st["reads_nonfinal"]), 5),
             "cut_reads": st["cut_reads"], "PASS": mean >= min(pair.values())}
        ok &= r["PASS"]
        res[d] = r
    res["PASS"] = ok
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
