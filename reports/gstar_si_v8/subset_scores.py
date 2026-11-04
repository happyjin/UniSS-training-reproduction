"""Paper v8 (prereg_v8 amendments 2-3, exploratory): MIST-Simul, fixed lag 2 s and LA2 on the same prefix-trained model,
scored on the v7 meeting subset (8 AMI / 10 AliMeeting meetings) with overlap_score.py. Writes
scores_ov2/<set>_sub_<system>.json (existing files are kept).

    /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python reports/gstar_si_v8/subset_scores.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "reports/gstar_si_v8"))
sys.path.insert(0, str(REPO / "reports/gstar_si_v6"))
import overlap_score as O  # noqa: E402
from meeting_score import universe  # noqa: E402

EV = REPO / "eval_outputs/uniss_qwen3_omni_v1"
SO = EV / "gstar_v8_20261103/scores_ov2"
ON = REPO / "data/online_v6_20261031"
REF = {"en2zh": "data/ami_multispk_si_v1/e2e_gstar_v1/windows_test_ref.jsonl",
       "zh2en": "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_ref.jsonl",
       "ali": "data/alimeeting_si_v1/windows_test_ref.jsonl"}


def main():
    sub = json.load(open(ON / "subset_meetings.json"))
    for st in ("en2zh", "zh2en", "ali"):
        meet = set(sub["ali" if st == "ali" else "ami"])
        for tag, h in (("on_S2dl", EV / f"gstar_v7_20261101/on_{st}_S2dl_full/hyp.jsonl"),
                       ("lag2", EV / f"gstar_v8_20261103/runs/on_{st}_S2dllag2/hyp.jsonl"),
                       ("pla2", EV / f"gstar_v8_20261103/runs/on_{st}_S2dlLA2/hyp.jsonl")):
            out = SO / f"{st}_sub_{tag}.json"
            if out.exists() or not h.exists():
                print(st, tag, "kept" if out.exists() else "no output yet")
                continue
            H = {x["id"]: x for x in map(json.loads, open(h, encoding="utf-8"))}
            W = [w for w in map(json.loads, open(ON / f"windows_test_online_{st}.jsonl")) if w["id"] in H and w["meeting"] in meet]
            RW = list(map(json.loads, open(REPO / REF[st], encoding="utf-8")))
            r = O.score(st, W, H, RW, universe(st, {w["meeting"] for w in W}), online=True)
            with open(out, "x") as fh:
                json.dump(r, fh, indent=1)
            print(st, tag, r["meetings"], r["tcpBLEU_all"], r["tcpBLEU_ov"], (r.get("lag_ov") or {}).get("p50"))


if __name__ == "__main__":
    main()
