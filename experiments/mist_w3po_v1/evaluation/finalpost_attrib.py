"""Attribution of the final-read causal P gain (finalpost dev test), CPU only. The flag changes only final reads, but
bf16 MoE numerics depend on the batch: a changed final-read item can flip a greedy token of another window's NON-final
read in the same batch, and that window then diverges (the flag-off rerun is bit-identical to the reference, so this
noise is not in it). Windows of cand vs ref are split into
  same       identical output
  final_only every non-final read identical, the final read differs   (the clean effect of the flag)
  perturbed  some earlier read differs                                  (batch-composition noise + the flag)
and two hybrids are scored with the online scorer (overlap_score.score, online=True, as score_run): ref with cand's
output on the final_only windows only, and ref with cand's output on the perturbed windows only; each is compared with
ref by the paired meeting bootstrap (compare.py). Writes a new json.

    python -m experiments.mist_w3po_v1.evaluation.finalpost_attrib --ref <dir> --cand <dir> --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.mist_w3po_v1.evaluation.compare import compare

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "reports/gstar_si_v8"))
sys.path.insert(0, str(REPO / "reports/gstar_si_v6"))
WIN = "eval_outputs/mist_w3po_v1/data/windows_zh2en_v1/windows_dev_online.jsonl"
REF_WIN = "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_validation_ref.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    import overlap_score as O
    from meeting_score import universe

    W = [json.loads(l) for l in open(WIN, encoding="utf-8")]
    RW = [json.loads(l) for l in open(REF_WIN, encoding="utf-8")]
    U = universe("zh2en", {w["meeting"] for w in W})
    R = {r["id"]: r for r in map(json.loads, open(Path(a.ref) / "hyp.jsonl", encoding="utf-8"))}
    C = {r["id"]: r for r in map(json.loads, open(Path(a.cand) / "hyp.jsonl", encoding="utf-8"))}
    cls = {}
    for w in W:
        r, c = R[w["id"]], C[w["id"]]
        lr, lc = r["read_log"], c["read_log"]
        if r["raw"] == c["raw"] and lr == lc:
            cls[w["id"]] = "same"
        elif len(lr) == len(lc) and all(x["new"] == y["new"] for x, y in zip(lr[:-1], lc[:-1])):
            cls[w["id"]] = "final_only"
        else:
            cls[w["id"]] = "perturbed"
    base = O.score("zh2en", W, R, RW, U, online=True)
    res = {"counts": {k: sum(v == k for v in cls.values()) for k in ("same", "final_only", "perturbed")}, "hybrids": {}}
    for name, keep in (("final_only", "final_only"), ("perturbed", "perturbed")):
        H = {i: (C[i] if cls[i] == keep else R[i]) for i in R}
        s = O.score("zh2en", W, H, RW, U, online=True)
        cmp_ = compare(base, s, regs=("all", "ov", "1spk"))
        res["hybrids"][name] = {"compare_vs_ref": cmp_,
                                "short_recall": {k: v["recall"] for k, v in s["short_recall"].items()},
                                "omitted_ov": s["omitted_ov"], "omitted_1spk": s["omitted_1spk"],
                                "collapse": s["collapse"], "CR": s["CR"]}
        print(name, res["counts"][keep], {k: (cmp_[k]["diff"], cmp_[k]["ci95"], cmp_[k]["p_le0"]) for k in ("all", "ov", "1spk")},
              flush=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
