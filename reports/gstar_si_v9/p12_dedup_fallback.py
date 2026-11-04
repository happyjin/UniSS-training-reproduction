"""Paper v9 P1-2 (review M08; prereg_v9 amendment 1): online de-duplication x short-turn fallback, 2x2, on MIST-Simul's
whole-meeting output (no retraining).

  dedup     within a meeting, a segment whose slot had a kept segment with chrF >= 50 committed in the 5 s before it is
            dropped
  fallback  a D2 segment (the turn-segmented speaker-blind decode, labelled with its turn's slot) of <= 1.5 s source is
            added when no MIST-Simul segment of the same slot overlaps it within +-0.5 s and no MIST-Simul text within
            +-0.5 s has chrF >= 50 with it; it keeps its own commit time (the turn's read)
Output: one hyp per variant on the meeting clock as single pseudo-windows per meeting (start 0, commit times absolute),
scored by overlap_score.py with --online off and commit times already absolute. Writes <out>/<set>_<variant>.jsonl and
<out>/<set>_windows.jsonl (refuses to overwrite).

    python3 reports/gstar_si_v9/p12_dedup_fallback.py --set en2zh --out eval_outputs/.../gstar_v9_20261104/p12
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from sacrebleu.metrics import CHRF

REPO = Path(__file__).resolve().parents[2]
EV = REPO / "eval_outputs/uniss_qwen3_omni_v1"
ON = REPO / "data/online_v6_20261031"
TURNS = REPO / "data/turns_v9_20261104"
CH = CHRF()


def absolute(windows, hyp, slot_of=None):
    """meeting -> [segment with absolute a, b, commit]; commit as overlap_score (clamped to opening / close detection)."""
    W = {w["id"]: w for w in map(json.loads, open(windows, encoding="utf-8"))}
    out = defaultdict(list)
    for h in map(json.loads, open(hyp, encoding="utf-8")):
        w = W.get(h["id"])
        if w is None:
            continue
        for s in h["segments"]:
            if not s["text"].strip():
                continue
            c = None
            if "write_ms" in s:
                c = max(s["write_ms"], w["open_detect_ms"])
                if s["write_ms"] >= w["dur_ms"]:
                    c = max(c, w["close_detect_ms"])
                c = w["start_s"] + c / 1000
            out[w["meeting"]].append({"spk": s["spk"], "a": w["start_s"] + s["start_s"],
                                      "b": w["start_s"] + max(s["end_s"], s["start_s"] + 0.08), "text": s["text"].strip(),
                                      "commit": c})
    return out


def dedup(segs):
    kept, out = defaultdict(list), []
    for s in sorted(segs, key=lambda s: (s["commit"] or 0.0)):
        prev = [t for c, t in kept[s["spk"]] if (s["commit"] or 0.0) - c <= 5.0]
        if any(CH.sentence_score(s["text"], [t]).score >= 50 for t in prev):
            continue
        kept[s["spk"]].append((s["commit"] or 0.0, s["text"]))
        out.append(s)
    return out


def fallback(segs, d2):
    add = []
    for c in d2:
        if c["b"] - c["a"] > 1.5:
            continue
        near = [s for s in segs if min(c["b"] + 0.5, s["b"]) > max(c["a"] - 0.5, s["a"])]
        if any(s["spk"] == c["spk"] for s in near):
            continue
        if any(CH.sentence_score(c["text"], [s["text"]]).score >= 50 for s in near):
            continue
        add.append(dict(c, fallback=True))
    return segs + add


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st = a.set
    S = absolute(ON / f"windows_test_online_{st}.jsonl", EV / f"gstar_v7_20261101/on_{st}_S2dl_full/hyp.jsonl")
    D2 = absolute(TURNS / f"windows_test_turns_{st}.jsonl", EV / f"gstar_v9_20261104/labels/{st}_D2.jsonl")
    S = {m: v for m, v in S.items() if m in D2}   # D2 exists on the subset meetings only (prereg_v9 amendment 2)
    wf = out / f"{st}_windows.jsonl"
    if not wf.exists():
        with open(wf, "x") as fh:
            for m in sorted(S):
                end = max(s["b"] for s in S[m]) + 60
                fh.write(json.dumps({"id": f"{m}_meeting", "meeting": m, "start_s": 0.0, "dur_ms": int(end * 1000),
                                     "open_detect_ms": 0, "close_detect_ms": int(end * 1000)}) + "\n")
    stats = {}
    for name, f in (("base", lambda m: S[m]), ("dedup", lambda m: dedup(S[m])),
                    ("fallback", lambda m: fallback(S[m], D2.get(m, []))),
                    ("dedup_fallback", lambda m: fallback(dedup(S[m]), D2.get(m, [])))):
        o = out / f"{st}_{name}.jsonl"
        if o.exists():
            continue
        n_seg = n_fb = 0
        with open(o, "x", encoding="utf-8") as fh:
            for m in sorted(S):
                segs = f(m)
                n_seg += len(segs)
                n_fb += sum(1 for s in segs if s.get("fallback"))
                fh.write(json.dumps({"id": f"{m}_meeting", "segments": [
                    {"spk": s["spk"], "start_s": round(s["a"], 3), "end_s": round(s["b"], 3), "text": s["text"],
                     **({"write_ms": int(round(s["commit"] * 1000))} if s["commit"] is not None else {})}
                    for s in sorted(segs, key=lambda s: s["a"])]}, ensure_ascii=False) + "\n")
        stats[name] = {"segments": n_seg, "fallback_added": n_fb}
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
