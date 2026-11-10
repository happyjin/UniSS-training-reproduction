"""Speech-gap plan C2 (text_to_speech_gap_plan.zh-CN.md §5 C2), CPU, dev only: what a shorter window-closing gap would
do, before any decode. build_windows_zh2en.online_windows closes a window after GAP = 1.0 s without diarizer activity;
the close is known LOOKAHEAD = 1.04 s after the deciding frame, so a window's final read waits
close_detect - audio end ~ GAP + 0.08 + 1.04 - 0.3 s.

Per gap g: windows, the mean detector wait (close_detect_ms - dur_ms, as the window jsonl), and how many placed,
non-hesitation reference utterances (zh_mix_v8, the dev meetings of windows_dev_online.jsonl) a window boundary cuts
(the utterance's [zh_begin, zh_end] is not inside one window's audio span) -- the text-risk proxy. No BLEU is used.

    python -m experiments.mist_w3po_v1.evaluation.c2_gap_stats --gaps 1.0,0.7,0.5,0.3 --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.mist_w3po_v1.data.build_windows_zh2en import FRAME, LOOKAHEAD, online_windows

DATA = Path("data/ami_multispk_si_v1")
WIN = "eval_outputs/mist_w3po_v1/data/windows_zh2en_v1/windows_dev_online.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gaps", default="1.0,0.7,0.5,0.3")
    ap.add_argument("--diar", default="nemotron_lowlat_zh8")
    ap.add_argument("--mix", default="zh_mix_v8")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"拒绝覆盖已有输出: {a.out}")
    meetings = sorted({json.loads(l)["meeting"] for l in open(WIN)})
    res = {}
    for g in map(float, a.gaps.split(",")):
        nw = 0
        waits, spans = [], 0
        cut = tot = 0
        for m in meetings:
            post = torch.load(DATA / "diar" / a.diar / f"{m}.post.pt", weights_only=False)["post80"].float().numpy()
            sess_s = len(post) * FRAME   # (the builder uses the audio length; differs by < 1 frame)
            spans_m = []
            for st, en, j in online_windows(post, g):
                ws, we = max(0.0, st * FRAME - 0.2), min(sess_s, en * FRAME + 0.3)
                close = min(sess_s, j * FRAME + LOOKAHEAD) if j < len(post) else sess_s
                waits.append(max(0.0, close - we))
                spans_m.append((ws, we))
            nw += len(spans_m)
            for u in map(json.loads, open(DATA / a.mix / "utts" / f"{m}.jsonl", encoding="utf-8")):
                if u["hes"] or not u["placed"]:
                    continue
                tot += 1
                b, e = u["zh_begin"], u["zh_end"]
                cov = [min(we, e) - max(ws, b) for ws, we in spans_m]
                if not any(c >= 0.9 * (e - b) for c in cov):
                    cut += 1
        res[str(g)] = {"windows": nw, "detector_wait_mean_s": round(float(np.mean(waits)), 3),
                       "detector_wait_p50_s": round(float(np.median(waits)), 3),
                       "utts": tot, "utts_cut": cut, "cut_rate": round(cut / max(tot, 1), 4)}
        print(g, res[str(g)], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "x"), indent=1)


if __name__ == "__main__":
    main()
