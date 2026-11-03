"""Paper v8 C2 (prereg_v8 amendment 1): diarizer-turn cascade fallback for short turns MIST leaves out.

For every window, a cascade segment (same diarizer, same slot numbering) of <= 1.5 s source duration is added to the
MIST output when no MIST segment of the same slot overlaps it within +-0.5 s and no MIST text within +-0.5 s has
chrF >= 50 with it. The added segment's commit is max(cascade commit, the MIST window's read at that time).
Writes <out>/hyp.jsonl (refuses to overwrite) and <out>/added.json.

    python reports/gstar_si_v8/c2_fallback.py --mist <hyp.jsonl> --cascade <hyp.jsonl> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from sacrebleu.metrics import CHRF

MAXDUR, PAD, CHRF_MAX = 1.5, 0.5, 50.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mist", required=True)
    ap.add_argument("--cascade", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "hyp.jsonl").exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    out.mkdir(parents=True, exist_ok=True)
    chrf = CHRF()
    C = {h["id"]: h for h in map(json.loads, open(a.cascade, encoding="utf-8"))}
    n_add = n_win = 0
    with open(out / "hyp.jsonl", "x", encoding="utf-8") as fh:
        for h in map(json.loads, open(a.mist, encoding="utf-8")):
            segs = list(h["segments"])
            add = []
            for c in (C.get(h["id"]) or {}).get("segments", []):
                if not c["text"].strip() or c["end_s"] - c["start_s"] > MAXDUR:
                    continue
                a0, a1 = c["start_s"] - PAD, c["end_s"] + PAD
                near = [s for s in segs if min(a1, s["end_s"]) > max(a0, s["start_s"])]
                if any(s["spk"] == c["spk"] for s in near):
                    continue
                if any(chrf.sentence_score(c["text"], [s["text"]]).score >= CHRF_MAX for s in near if s["text"].strip()):
                    continue
                s = dict(c)
                if "write_ms" in c or any("write_ms" in x for x in segs):
                    wm = c.get("write_ms", int(math.ceil(c["end_s"])) * 1000)
                    s["write_ms"] = max(wm, int(math.ceil(c["end_s"])) * 1000)
                s["fallback"] = True
                add.append(s)
            n_add += len(add)
            n_win += bool(add)
            h["segments"] = sorted(segs + add, key=lambda s: (s["start_s"], s["spk"]))
            fh.write(json.dumps(h, ensure_ascii=False) + "\n")
    (out / "added.json").write_text(json.dumps({"added_segments": n_add, "windows_with_additions": n_win}))
    print(n_add, "segments added in", n_win, "windows")


if __name__ == "__main__":
    main()
