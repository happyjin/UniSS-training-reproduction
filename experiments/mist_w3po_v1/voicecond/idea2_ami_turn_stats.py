"""Idea 2: turn-taking statistics from the AMI manual segment annotations (train meetings of the AMI full-corpus ASR
partition only; dev/test meetings excluded). Only timing statistics are taken (no audio, no text).

Turns: all segments of a meeting sorted by start; consecutive segments of one speaker form one turn. For each turn
change we record the transition gap = next turn start - previous turn end (< 0: overlap). Writes
eval_outputs/mist_w3po_v1/voicecond_idea2/ami_turn_stats.json (refuses to overwrite):
  turn_dur[]  (s), gap[] (s), per-meeting overlap fraction (time with >= 2 speakers / speech time), speakers per meeting.
"""
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ZIP = REPO / "data/ami_multispk_si_v1/annotations/ami_public_manual_1.6.2.zip"
OUT = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea2/ami_turn_stats.json"
HELD = ("ES2011", "IS1008", "TS3004", "IB4001", "IB4002", "IB4003", "IB4004", "IB4010", "IB4011",   # dev
        "ES2004", "IS1009", "TS3003", "EN2002")                                                     # test
RX = re.compile(r'transcriber_start="([0-9.]+)"\s+transcriber_end="([0-9.]+)"')


def main():
    if OUT.exists():
        sys.exit(f"refuse to overwrite {OUT}")
    segs = {}
    with zipfile.ZipFile(ZIP) as z:
        for n in z.namelist():
            m = re.match(r"segments/(\w+)\.([A-Z])\.segments\.xml$", n)
            if not m or m.group(1).startswith(HELD):
                continue
            for s, e in RX.findall(z.read(n).decode("latin-1")):
                s, e = float(s), float(e)
                if e - s > 0.05:
                    segs.setdefault(m.group(1), []).append((s, e, m.group(2)))
    turn_dur, gaps, ovl, nspk = [], [], [], []
    for mt, ss in sorted(segs.items()):
        ss.sort()
        nspk.append(len({x[2] for x in ss}))
        cur = list(ss[0])
        for s, e, k in ss[1:]:
            if k == cur[2]:
                cur[1] = max(cur[1], e)
            else:
                turn_dur.append(cur[1] - cur[0]); gaps.append(s - cur[1]); cur = [s, e, k]
        turn_dur.append(cur[1] - cur[0])
        T = int(max(e for _, e, _ in ss) * 100) + 1
        cnt = np.zeros(T, np.int16)
        for s, e, _ in ss:
            cnt[int(s * 100):int(e * 100)] += 1
        ovl.append(float((cnt >= 2).sum() / max(1, (cnt >= 1).sum())))
    td, gp = np.array(turn_dur), np.array(gaps)
    q = lambda a: {f"p{p}": round(float(np.percentile(a, p)), 3) for p in (5, 10, 25, 50, 75, 90, 95)}
    res = {"meetings": len(segs), "turns": len(td), "turn_dur_q": q(td), "gap_q": q(gp),
           "frac_gap_negative": round(float((gp < 0).mean()), 4), "overlap_frac_mean": round(float(np.mean(ovl)), 4),
           "speakers_per_meeting": {str(k): nspk.count(k) for k in sorted(set(nspk))},
           "turn_dur": [round(float(x), 3) for x in td], "gap": [round(float(x), 3) for x in gp]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res))
    print(json.dumps({k: v for k, v in res.items() if k not in ("turn_dur", "gap")}, indent=1))


if __name__ == "__main__":
    main()
