"""Paper v9 overlap-capable controls (prereg_v9 §3), as labelled hyp files for overlap_score.py (scored with relabel off).

  d1   multi-label split of the speaker-blind Simul output (Rel): per segment [a, b], with the diarizer frames available at
       its commit (<= commit - 1.04 s): if two or more slots are each active (> 0.5) for >= 0.24 s inside the visible
       span, the span is cut where the dominant active slot changes (runs < 0.24 s merged into the previous run), the
       text is split in proportion to the runs' durations (characters for Chinese, words for English; the unseen tail
       goes to the last run), and each piece gets its run's slot; otherwise the segment gets Rel's single causal label
       (overlap_score.relabel_causal; "unk" when no frame is available). Dedup: within a meeting, a piece whose slot had
       a kept piece with chrF >= 50 committed in the 5 s before it is dropped.
  d2   turn-segmented decode: every segment of a turn window gets the turn's slot.

    python3 reports/gstar_si_v9/d1_d2_labels.py d1 --set en2zh --windows <online windows> --hyp <Rel hyp> --out <hyp>
    python3 reports/gstar_si_v9/d1_d2_labels.py d2 --windows <turn windows> --hyp <D2 hyp> --out <hyp>
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "reports/gstar_si_v8"))
import overlap_score as O  # noqa: E402

MINRUN = 3   # frames: 0.24 s


def runs_of(act):
    """dominant active slot per frame -> [(slot, n_frames)], short runs merged into the previous one."""
    dom, last = [], None
    for row in act:
        on = np.flatnonzero(row[1] > 0.5)
        if len(on):
            last = int(on[np.argmax(row[1][on])])
        dom.append(last)
    runs = []
    for s in dom:
        if s is None:
            continue
        if runs and runs[-1][0] == s:
            runs[-1][1] += 1
        else:
            runs.append([s, 1])
    merged = []
    for s, n in runs:
        if merged and (n < MINRUN or merged[-1][0] == s):
            merged[-1][1] += n
        else:
            merged.append([s, n])
    return merged


def split_segment(s, post, ws, commit, eng):
    lim = len(post) if commit is None else int((commit - O.LOOKAHEAD - ws) / O.FRAME)
    lim = min(lim, len(post))
    a, b = ws + s["start_s"], ws + max(s["end_s"], s["start_s"] + 0.08)
    r0, r1 = int((a - ws) / O.FRAME), int(np.ceil((b - ws) / O.FRAME))
    seg = {"a": a, "b": b}
    r1v = min(r1, lim)
    if r1v - r0 >= 1:
        vis = post[max(r0, 0):r1v].astype(np.float32)
        busy = (vis > 0.5).sum(0)
        if (busy >= MINRUN).sum() >= 2:
            runs = runs_of(list(enumerate(vis)))
            if len(runs) >= 2:
                pc = O.pieces(s["text"], eng)
                tot = sum(n for _, n in runs) + max(0, r1 - r1v)
                out, i0, t0 = [], 0, s["start_s"]
                for j, (slot, n) in enumerate(runs):
                    n_eff = n + (max(0, r1 - r1v) if j == len(runs) - 1 else 0)
                    i1 = len(pc) if j == len(runs) - 1 else i0 + int(round(len(pc) * n_eff / tot))
                    t1 = s["end_s"] if j == len(runs) - 1 else t0 + n_eff * O.FRAME
                    if i1 > i0:
                        out.append(dict(s, spk=slot + 1, start_s=round(t0, 2), end_s=round(t1, 2),
                                        text=(" " if eng else "").join(pc[i0:i1]), d1_split=True))
                    i0, t0 = i1, t1
                if out:
                    return out
    k = O.relabel_causal(seg, post, ws, commit)
    return [dict(s, spk=k if k else "unk")]


def d1(a):
    from sacrebleu.metrics import CHRF
    chrf = CHRF()
    eng = a.set != "en2zh"
    W = {w["id"]: w for w in map(json.loads, open(a.windows, encoding="utf-8"))}
    H = [h for h in map(json.loads, open(a.hyp, encoding="utf-8")) if h["id"] in W]
    by_m = defaultdict(list)
    for h in H:
        w = W[h["id"]]
        post = np.load(w["post"]["lowlat"])
        new = []
        for s in h["segments"]:
            c = None
            if "write_ms" in s:
                c = max(s["write_ms"], w["open_detect_ms"])
                if s["write_ms"] >= w["dur_ms"]:
                    c = max(c, w["close_detect_ms"])
                c = w["start_s"] + c / 1000
            for p in split_segment(s, post, w["start_s"], c, eng):
                p["_commit"] = c if c is not None else w["start_s"] + w["dur_ms"] / 1000
                new.append(p)
        h["segments"] = new
        by_m[w["meeting"]].append(h)
    n_drop = n_split = 0
    for m, hs in by_m.items():
        kept = defaultdict(list)   # slot -> [(commit, text)]
        for p in sorted((p for h in hs for p in h["segments"]), key=lambda p: p["_commit"]):
            n_split += bool(p.get("d1_split"))
            prev = [t for c, t in kept[p["spk"]] if p["_commit"] - c <= 5.0]
            if p["text"].strip() and any(chrf.sentence_score(p["text"], [t]).score >= 50 for t in prev):
                p["_drop"] = True
                n_drop += 1
            else:
                kept[p["spk"]].append((p["_commit"], p["text"]))
    with open(a.out, "x", encoding="utf-8") as fh:
        for hs in by_m.values():
            for h in hs:
                h["segments"] = [{k: v for k, v in p.items() if not k.startswith("_")} for p in h["segments"]
                                 if not p.get("_drop")]
                fh.write(json.dumps(h, ensure_ascii=False) + "\n")
    print(json.dumps({"windows": len(H), "split_pieces": n_split, "dropped_duplicates": n_drop}))


def d2(a):
    W = {w["id"]: w for w in map(json.loads, open(a.windows, encoding="utf-8"))}
    n = 0
    with open(a.out, "x", encoding="utf-8") as fh:
        for h in map(json.loads, open(a.hyp, encoding="utf-8")):
            if h["id"] not in W:
                continue
            h["segments"] = [dict(s, spk=W[h["id"]]["slot"]) for s in h["segments"]]
            fh.write(json.dumps(h, ensure_ascii=False) + "\n")
            n += 1
    print(json.dumps({"windows": n}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("d1")
    for k in ("--set", "--windows", "--hyp", "--out"):
        p.add_argument(k, required=True)
    p = sub.add_parser("d2")
    for k in ("--windows", "--hyp", "--out"):
        p.add_argument(k, required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"拒绝覆盖已有输出: {a.out}")
    {"d1": d1, "d2": d2}[a.cmd](a)
