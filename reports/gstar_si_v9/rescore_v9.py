"""Paper v9: merge the v9 whole-meeting runs and score every system with overlap_score.py on every reference set.

Merge: for each system the hyp_*.jsonl parts of all its run dirs (v8 and v9) are joined by window id into
gstar_v9_20261104/merged/<set>_<system>/hyp.jsonl once every window of the system's window file is present (an existing
merged file is kept). D2 is relabelled to the turns' slots (d1_d2_labels.py d2) after merging.
Reference sets: std (the v8 reference windows; AliMeeting: machine-translated), full (every utterance: En->Zh v8 B1,
Zh->En human AMI English, AliMeeting human), hum (AliMeeting reference windows with the DiariST human translations).
Offline reference-window systems of v8 are scored on the AliMeeting human sets too. Scores go to
gstar_v9_20261104/scores/<set>_<system>_<refset>.json (existing files kept).

    /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python reports/gstar_si_v9/rescore_v9.py [--jobs 24]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "reports/gstar_si_v8"))
sys.path.insert(0, str(REPO / "reports/gstar_si_v6"))
EV = REPO / "eval_outputs/uniss_qwen3_omni_v1"
V7, V8, V9 = EV / "gstar_v7_20261101", EV / "gstar_v8_20261103", EV / "gstar_v9_20261104"
ON = REPO / "data/online_v6_20261031"
TURNS = REPO / "data/turns_v9_20261104"
HUM = REPO / "data/diarist_alimeeting_refs_v1"
PY = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python"
REF = {
    "en2zh": {"std": "data/ami_multispk_si_v1/e2e_gstar_v1/windows_test_ref.jsonl",
              "full": "data/ami_multispk_si_v1/e2e_gstar_v1/windows_test_fullref_v8.jsonl"},
    "zh2en": {"std": "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_ref.jsonl",
              "full": "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_fullref_v9.jsonl"},
    "ali": {"std": "data/alimeeting_si_v1/windows_test_ref.jsonl",
            "hum": "data/diarist_alimeeting_refs_v1/windows_test_ref_human.jsonl",
            "full": "data/diarist_alimeeting_refs_v1/windows_test_fullref_human.jsonl"},
}
SETS = ("en2zh", "zh2en", "ali")


def online(st):
    return str(ON / f"windows_test_online_{st}.jsonl")


def parts(dirs):
    got = {}
    for d in dirs:
        for f in sorted(Path(d).glob("hyp_*.jsonl")):
            for ln in open(f, encoding="utf-8"):
                try:
                    h = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                got.setdefault(h["id"], ln if ln.endswith("\n") else ln + "\n")
    return got


def merge(st, name, dirs, windows):
    out = V9 / "merged" / f"{st}_{name}" / "hyp.jsonl"
    if out.exists():
        return out
    want = [json.loads(ln)["id"] for ln in open(windows)]
    got = parts(dirs)
    if not set(want) <= set(got):
        print(f"{st} {name}: {len(set(want) & set(got))}/{len(want)} windows, not merged", flush=True)
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        fh.writelines(got[i] for i in want)
    print(f"{st} {name}: merged {len(want)}", flush=True)
    return out


def systems():
    """(set, name, hyp, windows, online, relabel)"""
    T = []
    for st in SETS:
        T.append((st, "S", V7 / f"on_{st}_S2dl_full/hyp.jsonl", online(st), True, False))
        T.append((st, "OLA2", V7 / f"on_{st}_O2_LA2_full/hyp.jsonl", online(st), True, False))
        T.append((st, "Rel", V8 / f"runs/on_{st}_S2dlnosd/hyp.jsonl", online(st), True, True))
        T.append((st, "LA2", V8 / f"runs/on_{st}_S2dlLA2/hyp.jsonl", online(st), True, False))
        T.append((st, "D1", V9 / f"labels/{st}_D1.jsonl", online(st), True, False))
        for tag in ("seed2", "seed3"):
            h = V8 / f"runs/on_{st}_S2dl_{tag}/hyp.jsonl"
            if h.exists():
                T.append((st, f"S_{tag}", h, online(st), True, False))
        for L in ("lag1", "lag2", "lag3", "blk2"):
            dirs = [V9 / f"runs/on_{st}_S2dl{L}"] + ([V8 / f"runs/on_{st}_S2dllag2"] if L == "lag2" else [])
            m = merge(st, L, dirs, online(st))
            if m:
                T.append((st, L.capitalize(), m, online(st), True, False))
        tw = str(TURNS / f"windows_test_turns_{st}.jsonl")
        m = merge(st, "D2raw", [V9 / f"runs/turn_{st}_D2"], tw)
        if m:
            lab = V9 / f"labels/{st}_D2.jsonl"
            if not lab.exists():
                subprocess.run([PY, str(REPO / "reports/gstar_si_v9/d1_d2_labels.py"), "d2", "--windows", tw,
                                "--hyp", str(m), "--out", str(lab)], check=True)
            T.append((st, "D2", lab, tw, True, False))
        for k in range(2, 5):   # P1-4 speaker-blind seeds, when decoded
            h = V9 / f"merged/{st}_Rel_seed{k}/hyp.jsonl"
            if h.exists():
                T.append((st, f"Rel_seed{k}", h, online(st), True, True))
    # P1-6 SDM far field (En->Zh)
    for name, rel in (("S_sdm", False), ("Rel_sdm", True)):
        h = V9 / f"merged/en2zh_{name}/hyp.jsonl"
        if h.exists():
            T.append(("en2zh", name, h, str(V9 / "sdm/windows_test_online_sdm.jsonl"), True, rel))
    return [t for t in T if Path(t[2]).exists()]


def offline_ali():
    """v8 reference-window systems on AliMeeting (for the human references)."""
    from rescore_existing import tasks
    return [("ali", k, h, w, False, k.endswith("relabel") and False) for st, k, h, w, onl in tasks() if st == "ali" and not onl]


def one(t):
    import overlap_score as O
    from meeting_score import universe
    st, name, hyp, win, onl, rel, refset = t
    o = V9 / "scores" / f"{st}_{name}_{refset}.json"
    if o.exists():
        return f"{o.name}: kept"
    H = {h["id"]: h for h in map(json.loads, open(hyp, encoding="utf-8"))}
    W = [w for w in map(json.loads, open(win, encoding="utf-8")) if w["id"] in H]
    RW = list(map(json.loads, open(REPO / REF[st][refset], encoding="utf-8")))
    U = universe(st, {w["meeting"] for w in W})
    r = O.score(st, W, H, RW, U, online=onl, relabel=rel)
    r["system"], r["hyp"], r["windows"], r["refset"] = name, str(hyp), str(win), refset
    with open(o, "x", encoding="utf-8") as fh:
        json.dump(r, fh, indent=1, ensure_ascii=False)
    return f"{o.name}: tcp {r['tcpBLEU_all']} ov {r['tcpBLEU_ov']} CR {r['CR']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args()
    (V9 / "scores").mkdir(parents=True, exist_ok=True)
    S = systems()
    todo = [(st, n, str(h), w, onl, rel, rs) for st, n, h, w, onl, rel in S for rs in REF[st]]
    todo += [(st, n, str(h), w, onl, rel, rs) for st, n, h, w, onl, rel in offline_ali() for rs in ("hum", "std")]   # reference-window outputs cover only the windows: not scored on full coverage
    print(len(todo), "scorings", flush=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        for msg in ex.map(one, todo):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
