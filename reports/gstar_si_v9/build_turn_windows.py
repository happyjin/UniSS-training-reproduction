"""Paper v9 D2 (prereg_v9 §3): turn-segmented input for the speaker-blind prefix model.

Every slot of the low-latency diarizer gives turns by the v7 cascade rule (activity > 0.5, gaps < 0.4 s closed, at least
0.24 s, padded 0.1 s on both sides; turns longer than 20 s cut at 20 s); turns of different slots may overlap. Each turn
is one online window of the mixed audio (no separation): its opening is known 1.04 s after the turn's first active frame
and its end 1.04 s after its last, as for build_online_windows.py. The window carries "slot": every segment decoded in it
is attributed to that slot (turn_relabel.py). Writes <out>/windows_test_turns_<set>.jsonl, audio/ and post/ (refuses
to overwrite the windows file).

    python3 reports/gstar_si_v9/build_turn_windows.py --set en2zh|zh2en|ali --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "reports/gstar_si_v6"))
from build_online_windows import SETS  # noqa: E402

FRAME, LOOKAHEAD, GAP, MINLEN, PAD, CAP = 0.08, 1.04, 0.4, 0.24, 0.1, 20.0


def turns(act: np.ndarray) -> list[tuple[int, int]]:
    """[first frame, last frame + 1) runs of one slot after closing gaps < GAP and dropping runs < MINLEN."""
    idx = np.flatnonzero(act)
    if not len(idx):
        return []
    runs, st, last = [], idx[0], idx[0]
    for i in idx[1:]:
        if (i - last - 1) * FRAME < GAP:
            last = i
        else:
            runs.append((st, last + 1))
            st = last = i
    runs.append((st, last + 1))
    out = []
    for a, b in runs:
        if (b - a) * FRAME < MINLEN:
            continue
        cap = int(round(CAP / FRAME))
        for c in range(a, b, cap):
            out.append((c, min(b, c + cap)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    wf = out / f"windows_test_turns_{a.set}.jsonl"
    if wf.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {wf}")
    meetings, audio_of, post_of = SETS[a.set]
    (out / "audio" / a.set).mkdir(parents=True, exist_ok=True)
    (out / "post" / a.set).mkdir(parents=True, exist_ok=True)
    n = 0
    with open(wf, "x", encoding="utf-8") as fh:
        for m in meetings():
            post = torch.load(post_of(m), weights_only=False)["post80"].float().numpy()
            x, sr = sf.read(str(audio_of(m)), dtype="float32", always_2d=True)
            x = x.mean(1)
            sess = len(x) / sr
            T = []
            for k in range(post.shape[1]):
                T += [(st, en, k) for st, en in turns(post[:, k] > 0.5)]
            T.sort()
            for i, (st, en, k) in enumerate(T):
                ws, we = max(0.0, st * FRAME - PAD), min(sess, en * FRAME + PAD)
                wid = f"{m}_tu_{i:05d}"
                wav = out / "audio" / a.set / f"{wid}.wav"
                sf.write(str(wav), x[round(ws * sr): round(we * sr)], sr)
                f0, nf = round(ws / FRAME), int(np.ceil((we - ws) / FRAME))
                s = post[f0: f0 + nf]
                if len(s) < nf:
                    s = np.concatenate([s, np.zeros((nf - len(s), post.shape[1]), dtype=s.dtype)])
                pp = out / "post" / a.set / f"{wid}.npy"
                np.save(pp, s.astype(np.float16))
                fh.write(json.dumps({"id": wid, "meeting": m, "split": "test", "wtype": "turn", "slot": k + 1,
                                     "dir": "en2zh" if a.set == "en2zh" else "zh2en", "start_s": round(ws, 3),
                                     "dur_ms": round((we - ws) * 1000), "audio": str(wav.resolve()),
                                     "post": {"lowlat": str(pp.resolve())},
                                     "open_detect_ms": round((st * FRAME + LOOKAHEAD - ws) * 1000),
                                     "close_detect_ms": round((min(sess, en * FRAME + LOOKAHEAD) - ws) * 1000),
                                     "utts": [], "strict_review": False}) + "\n")
                n += 1
            print(m, len(T), "turns", flush=True)
    print(json.dumps({"set": a.set, "windows": n}))


if __name__ == "__main__":
    main()
