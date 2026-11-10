"""Types of the non-recalled scored utterances of a dev decode (CPU; reports/mist_w3po_v1/metrics_first_si_analysis.zh-CN.md §4).

Same assignment as metric_headroom (= overlap_score.score, online). Every scored utterance that is not recalled is put
in the first matching class:
  NO_WINDOW     less than half of it lies inside any online window (the window detector never opened on it)
  CONTENT_ELSEWHERE  >= half of its content words (lower-cased, >= 3 letters; all words if it has none) appear in the
                     text of segments of any speaker overlapping [a - 2 s, b + 2 s]: the content was produced but credited
                     elsewhere (other label, merged into another speaker's segment, or outside the 1 s recall collar)
  CONTENT_ABSENT     otherwise: the content was not produced (true omission)
by duration band (<= 1 s / > 1 s) and region (iso / ov by the 20% rule), with their reference words.
Writes <out> (json; refuses to overwrite). Dev only.

    python -m experiments.mist_w3po_v1.evaluation.metric_omission_types --run <decode dir> --out <json>
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from experiments.mist_w3po_v1.evaluation import metric_headroom as M


def norm(t):
    import re
    return re.findall(r"[a-z0-9']+", t.lower())


def cwords(t):
    w = norm(t)
    c = [x for x in w if len(x) >= 3]
    return c or w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    from meeting_score import universe
    import overlap_score as O
    W = [json.loads(l) for l in open(M.REPO / M.WIN, encoding="utf-8")]
    H = {h["id"]: h for h in map(json.loads, open(Path(args.run) / "hyp.jsonl", encoding="utf-8"))}
    RW = [json.loads(l) for l in open(M.REPO / M.REF_WIN, encoding="utf-8")]
    U = universe(M.DR, {w["meeting"] for w in W})
    sc = M.scored_utts(W, RW)
    As = M.assign_all(W, H, sc, U)
    segs = M.build_segs(W, H)
    wins = defaultdict(list)
    for w in W:
        wins[w["meeting"]].append((w["start_s"], w["start_s"] + w["dur_ms"] / 1000))
    T = defaultdict(lambda: {"n": 0, "words": 0})
    tot_words = 0
    for m, A in As.items():
        for rid, v in A["sc"].items():
            tot_words += len(v["ref"].split())
            if A["rec"][rid]:
                continue
            d = max(v["b"] - v["a"], 1e-6)
            inwin = sum(O.overlap(v["a"], v["b"], a, b) for a, b in wins[m]) / d
            near = [s for s in segs[m] if s["text"] and O.overlap(s["a"] - 2, s["b"] + 2, v["a"], v["b"]) > 0]
            cw = cwords(v["ref"])
            pool = set(w for s in near for w in norm(s["text"]))
            hit = sum(w in pool for w in cw) / max(len(cw), 1)
            if inwin < 0.5:
                c = "NO_WINDOW"
            elif cw and hit >= 0.5:
                c = "CONTENT_ELSEWHERE"
            else:
                c = "CONTENT_ABSENT"
            band = "le1" if d <= 1.0 else "gt1"
            reg = "ov" if v["uov"] else "iso"
            for k in (f"{band}_{reg}_{c}", f"{band}_{c}", f"all_{c}"):
                T[k]["n"] += 1
                T[k]["words"] += len(v["ref"].split())
    res = {"run": args.run, "ref_words": tot_words,
           "types": {k: dict(v, word_share=round(v["words"] / tot_words, 4)) for k, v in sorted(T.items())}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps({k: (v["n"], v["words"]) for k, v in res["types"].items() if k.count("_") == 1 or k.startswith("all")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
