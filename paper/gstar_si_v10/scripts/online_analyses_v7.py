"""Paper v7 analyses of the whole-meeting online runs (prereg_v7 §1 exploratory, §5; v6 review M16, M18, T02, T04, T11,
T12). On the meeting clock, each scored reference utterance (non-hesitation utterances of the reference windows, as
meeting_score.py) is covered by an output segment of the run if the segment overlaps it within 0.5 s; the output slot
is mapped to a speaker once per meeting (chrF Hungarian on the whole meeting's text, as gstar_score). Each segment is
assigned to the one utterance it overlaps most (0.5 s collar); an utterance is covered if a segment is assigned to it.

  close        windows closed by 1 s of silence vs forced at 20 s: share of windows, utterance coverage of the
               utterances that lie (by midpoint) in them, and coverage of utterances within 2 s of a window boundary
               vs inside; cross-window repetition (an utterance covered by segments of two different windows); the
               closing latency (close_detect_ms - end of the last reference speech in the window)
  drift      M18: per 5-minute bin of the meeting, attribution accuracy of covered utterances (the segment's mapped
               speaker is the utterance's), and of re-entering speakers (silent >= 60 s before)
  hol        T04: segments nested in another speaker's segment of the same window (start later, end earlier): commit
               lag after their own end vs other segments' (the decoder writes segments by start time)

    python scripts/online_analyses_v7.py <set> <hyp.jsonl> <online windows> <ref windows> <name>
      -> generated/v7_online/<set>_<name>.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.uniss_qwen3_omni_v1.evaluation.gstar_score import match  # noqa: E402

GEN = Path(__file__).resolve().parents[1] / "generated" / "v7_online"


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main():
    st, hyp, onw, refw, name = sys.argv[1:6]
    sep = "" if st == "en2zh" else " "
    W = {w["id"]: w for w in load(onw)}
    H = {h["id"]: h for h in load(hyp)}
    utts = defaultdict(list)
    seen = set()
    for w in load(refw):
        for u in w["utts"]:
            if u["hesitation"] or u["record_id"] in seen:
                continue
            seen.add(u["record_id"])
            utts[w["meeting"]].append({"spk": u["speaker_id"], "a": w["start_s"] + u["begin_s"], "b": w["start_s"] + u["end_s"],
                                       "text": u.get("ref", u["zh"]).strip()})
    segs = defaultdict(list)
    for wid, h in H.items():
        w = W[wid]
        for s in h["segments"]:
            segs[w["meeting"]].append({"win": wid, "spk": s["spk"], "a": w["start_s"] + s["start_s"], "b": w["start_s"] + s["end_s"],
                                       "text": s["text"], "commit": w["start_s"] + s.get("write_ms", 0) / 1000,
                                       "rel_a": s["start_s"], "rel_b": s["end_s"]})
    res = {"close": defaultdict(lambda: defaultdict(float)), "drift": defaultdict(lambda: [0, 0]), "reentry": [0, 0],
           "hol": {"nested": [], "other": []}}
    close_lat = defaultdict(list)
    meetings = {w["meeting"] for w in W.values()}
    for m in sorted(meetings):
        U, S = sorted(utts[m], key=lambda u: u["a"]), segs[m]
        assigned = defaultdict(list)
        starts = np.array([u["a"] for u in U])
        for s in S:
            if not s["text"].strip():
                continue
            lo = np.searchsorted(starts, s["a"] - 30)
            best, bo = None, 0.0
            for i in range(lo, len(U)):
                u = U[i]
                if u["a"] > s["b"] + 0.5:
                    break
                o = min(s["b"], u["b"] + 0.5) - max(s["a"], u["a"] - 0.5)
                if o > bo:
                    best, bo = i, o
            if best is not None:
                assigned[best].append(s)
        hm, rm = defaultdict(str), defaultdict(str)
        for s in S:
            hm[s["spk"]] += sep + s["text"]
        for u in U:
            rm[u["spk"]] += sep + u["text"]
        mp = match(dict(hm), dict(rm))
        wins = sorted((w for w in W.values() if w["meeting"] == m), key=lambda w: w["start_s"])
        last_end = {}
        for i, u in enumerate(U):
            cov = assigned.get(i, [])
            mid = (u["a"] + u["b"]) / 2
            w = next((w for w in wins if w["start_s"] <= mid <= w["start_s"] + w["dur_ms"] / 1000), None)
            kind = "none" if w is None else ("cap" if w["dur_ms"] >= 20000 else "silence")
            near = w is not None and min(mid - w["start_s"], w["start_s"] + w["dur_ms"] / 1000 - mid) <= 2.0
            c = res["close"][kind]
            c["utts"] += 1
            c["covered"] += bool(cov)
            c["near_boundary"] += near
            c["near_covered"] += near and bool(cov)
            c["two_windows"] += len({s["win"] for s in cov}) > 1
            if cov:
                ok = any(mp.get(s["spk"]) == u["spk"] for s in cov)
                b5 = int(u["a"] // 300)
                res["drift"][b5][0] += 1
                res["drift"][b5][1] += ok
                if u["a"] - last_end.get(u["spk"], -1e9) >= 60 and u["spk"] in last_end:
                    res["reentry"][0] += 1
                    res["reentry"][1] += ok
            last_end[u["spk"]] = max(last_end.get(u["spk"], 0), u["b"])
        for w in wins:
            e = w["start_s"] + w["dur_ms"] / 1000
            inside = [u["b"] for u in U if w["start_s"] <= u["b"] <= e]
            if inside and "close_detect_ms" in w:
                close_lat["cap" if w["dur_ms"] >= 20000 else "silence"].append(w["start_s"] + w["close_detect_ms"] / 1000 - max(inside))
        byw = defaultdict(list)
        for s in S:
            byw[s["win"]].append(s)
        for ss in byw.values():
            for s in ss:
                nested = any(o is not s and o["spk"] != s["spk"] and o["a"] < s["a"] and o["b"] > s["b"] for o in ss)
                res["hol"]["nested" if nested else "other"].append(s["commit"] - s["b"])
    out = {"set": st, "name": name, "close": {}, "close_latency_s": {}, "drift_5min": {}, "reentry": {}, "hol": {}}
    nw = len(W)
    out["windows_cap_share"] = round(sum(w["dur_ms"] >= 20000 for w in W.values()) / max(nw, 1), 4)
    for k, c in res["close"].items():
        out["close"][k] = {"utts": int(c["utts"]), "coverage": round(c["covered"] / max(c["utts"], 1), 4),
                           "near_boundary_share": round(c["near_boundary"] / max(c["utts"], 1), 4),
                           "coverage_near_boundary": round(c["near_covered"] / max(c["near_boundary"], 1), 4),
                           "coverage_inside": round((c["covered"] - c["near_covered"]) / max(c["utts"] - c["near_boundary"], 1), 4),
                           "repeated_in_two_windows": round(c["two_windows"] / max(c["covered"], 1), 4)}
    for k, v in close_lat.items():
        out["close_latency_s"][k] = {"n": len(v), "p50": round(float(np.median(v)), 2), "p90": round(float(np.percentile(v, 90)), 2)}
    for b, (n, ok) in sorted(res["drift"].items()):
        out["drift_5min"][f"{5 * b}-{5 * b + 5}min"] = {"covered_utts": n, "attribution": round(ok / max(n, 1), 4)}
    out["reentry"] = {"utts": res["reentry"][0], "attribution": round(res["reentry"][1] / max(res["reentry"][0], 1), 4)}
    for k, v in res["hol"].items():
        if v:
            out["hol"][k] = {"segments": len(v), "commit_after_end_p50_s": round(float(np.median(v)), 2),
                             "p90_s": round(float(np.percentile(v, 90)), 2)}
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / f"{st}_{name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out)[:1500])


if __name__ == "__main__":
    main()
