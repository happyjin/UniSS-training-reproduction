"""Paper v9 P1-1 (review M07): sensitivity of the overlap comparison of MIST-Simul (S) vs the speaker-blind Simul model with
causal relabelling (Rel) to the scoring choices fixed in prereg_v8: overlap share of a unit (10/20/30%), assignment collar
(0/0.5/1/2 s) and minimum overlap of a concurrent pair (0.2/0.5/1.0 s). Whole meetings, online, the v8 outputs;
AliMeeting with the DiariST human references. Per cell: tcpBLEU-ov of S and Rel, their difference with a 95% meeting-
bootstrap interval (1,000 resamples, corpus BLEU), and the concurrent-recall difference. By construction every segment
goes to one reference unit (a proportional split only divides a segment between units of the same speaker), so a generic
short reply cannot be counted for several units. Writes <out> (refuses to overwrite).

    /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python reports/gstar_si_v9/sensitivity.py --out <json>
"""
import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "reports/gstar_si_v8"))
sys.path.insert(0, str(REPO / "reports/gstar_si_v6"))
sys.path.insert(0, str(REPO / "paper/gstar_si_v8/scripts"))
EV = REPO / "eval_outputs/uniss_qwen3_omni_v1"
ON = REPO / "data/online_v6_20261031"
REF = {"en2zh": "data/ami_multispk_si_v1/e2e_gstar_v1/windows_test_ref.jsonl",
       "zh2en": "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_ref.jsonl",
       "ali": "data/diarist_alimeeting_refs_v1/windows_test_ref_human.jsonl"}
HYP = {"S": lambda st: EV / f"gstar_v7_20261101/on_{st}_S2dl_full/hyp.jsonl",
       "Rel": lambda st: EV / f"gstar_v8_20261103/runs/on_{st}_S2dlnosd/hyp.jsonl"}


def cell(args):
    st, share, collar, pmin = args
    import overlap_score as O
    from meeting_score import universe
    O.OV_SHARE, O.PAIR_MIN = share, pmin
    RW = list(map(json.loads, open(REPO / REF[st], encoding="utf-8")))
    res = {}
    for name in ("S", "Rel"):
        H = {h["id"]: h for h in map(json.loads, open(HYP[name](st), encoding="utf-8"))}
        W = [w for w in map(json.loads, open(ON / f"windows_test_online_{st}.jsonl")) if w["id"] in H]
        r = O.score(st, W, H, RW, universe(st, {w["meeting"] for w in W}), online=True, relabel=name == "Rel",
                    collar=collar)
        res[name] = r
    from bootstrap_v7 import bleu_vec
    ms = sorted(set(res["S"]["per_meeting"]) & set(res["Rel"]["per_meeting"]))
    A = np.array([res["S"]["per_meeting"][m]["ov_stats"] or [0.0] * 10 for m in ms])
    B = np.array([res["Rel"]["per_meeting"][m]["ov_stats"] or [0.0] * 10 for m in ms])
    rng = np.random.default_rng(20261104)
    J = rng.integers(0, len(ms), (1000, len(ms)))
    d = np.array([bleu_vec(A[j].sum(0))[0] - bleu_vec(B[j].sum(0))[0] for j in J])
    return {"set": st, "ov_share": share, "collar": collar, "pair_min": pmin,
            "ov_S": res["S"]["tcpBLEU_ov"], "ov_Rel": res["Rel"]["tcpBLEU_ov"],
            "ov_diff": round(float(res["S"]["tcpBLEU_ov"] - res["Rel"]["tcpBLEU_ov"]), 2),
            "ov_diff_ci": [round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)],
            "CR_S": res["S"]["CR"], "CR_Rel": res["Rel"]["CR"], "CR_diff": round(res["S"]["CR"] - res["Rel"]["CR"], 4),
            "pairs": res["S"]["pairs"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    grid = list(itertools.product(("en2zh", "zh2en", "ali"), (0.1, 0.2, 0.3), (0.0, 0.5, 1.0, 2.0), (0.2, 0.5, 1.0)))
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(cell, grid))
    out.write_text(json.dumps(rows, indent=1))
    for st in ("en2zh", "zh2en", "ali"):
        r = [x for x in rows if x["set"] == st]
        print(st, "ov diff range", min(x["ov_diff"] for x in r), max(x["ov_diff"] for x in r),
              "CI excl. 0:", sum(x["ov_diff_ci"][0] > 0 for x in r), "/", len(r),
              "CR diff range", min(x["CR_diff"] for x in r), max(x["CR_diff"] for x in r))


if __name__ == "__main__":
    main()
