"""Paper v8 A3: the overlap-aware scores (overlap_score.py) of every existing v6/v7 system, on its own windows (reference
windows, or the whole-meeting online windows), into eval_outputs/.../gstar_v8_20261103/scores_ov/<set>_<system>.json.
An existing output file is kept (never overwritten); a system without outputs is skipped.

    /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python reports/gstar_si_v8/rescore_existing.py [--jobs 12]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper/gstar_si_v8/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

EV = REPO / "eval_outputs/uniss_qwen3_omni_v1"
OUT = EV / "gstar_v8_20261103/scores_ov"
ON = REPO / "data/online_v6_20261031"
REF = {"en2zh": REPO / "data/ami_multispk_si_v1/e2e_gstar_v1/windows_test_ref.jsonl",
       "zh2en": REPO / "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_ref.jsonl",
       "ali": REPO / "data/alimeeting_si_v1/windows_test_ref.jsonl"}
REFWIN = ["C-ft2", "C-turn", "OC", "CC", "SC", "O2-nosd", "O2-nosd-relabel", "O2", "O2_LA2_d1", "S2dl_d1",
          "S2dl_LA2_full", "S2dlZ_LA2_d1", "S2dl_seed2_full", "S2dl_seed3_d1", "S2dlZ_seed2_d1"]


def tasks():
    from systems_v7 import hyp_dir, windows
    out = []
    for dr in ("en2zh", "zh2en", "ali"):
        for k in REFWIN:
            try:
                d = hyp_dir(dr, k)
            except Exception:
                d = None
            if d is None or not (Path(d) / "hyp.jsonl").exists():
                continue
            out.append((dr, k, str(Path(d) / "hyp.jsonl"), str(windows(dr, k)), False))
        for k in ("S2dl", "O2_LA2"):
            h = EV / f"gstar_v7_20261101/on_{dr}_{k}_full/hyp.jsonl"
            if h.exists():
                out.append((dr, f"on_{k}", str(h), str(ON / f"windows_test_online_{dr}.jsonl"), True))
    return out


def one(t):
    import overlap_score as O
    from meeting_score import universe
    dr, k, hyp, win, online = t
    for split in (True, False):
        o = OUT / f"{dr}_{k}{'' if split else '_nosplit'}.json"
        if o.exists():
            continue
        H = {h["id"]: h for h in map(json.loads, open(hyp, encoding="utf-8"))}
        W = [w for w in map(json.loads, open(win, encoding="utf-8")) if w["id"] in H]
        if not W:
            return f"{dr} {k}: no windows"
        RW = list(map(json.loads, open(REF[dr], encoding="utf-8")))
        U = universe(dr, {w["meeting"] for w in W})
        r = O.score(dr, W, H, RW, U, online=online, split=split)
        r["system"], r["hyp"], r["windows"] = k, hyp, win
        with open(o, "x", encoding="utf-8") as fh:
            json.dump(r, fh, indent=1, ensure_ascii=False)
    return f"{dr} {k}: done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    T = tasks()
    print(len(T), "systems", flush=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        for msg in ex.map(one, T):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
