"""Paper v6, review M01 (prereg_v6 §1): whole test meetings cut into windows online, with no reference boundary.

The rule is the diarizer-window rule the training data has used since v2 (data_clean/build_ami_gstar_si_rows.py
diar_windows), on the low-latency streaming posteriors: a window opens at the first frame where any slot is > 0.5 and
closes after 1.0 s without activity or when it reaches 20 s; its audio is [open - 0.2 s, last active frame + 0.3 s].
Unlike the training windows, nothing is filtered: every window of every test meeting is kept, so the whole meeting
is covered. Per window two detection times (relative to the window start, ms) bound when its output can exist:
  open_detect_ms   the opening frame is known 1.04 s after it (the preset's look-ahead)
  close_detect_ms  the closing decision (1 s of silence, or the 20 s cap) is known 1.04 s after the frame that decides it
The streaming decoder reads the window as usual; commit times are later clamped to >= open_detect_ms, and the last
read's commits are moved to close_detect_ms (online_post.py).

    python3 reports/gstar_si_v6/build_online_windows.py --set en2zh|zh2en|ali --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/ami_multispk_si_v1"
A = REPO / "data/alimeeting_si_v1"
FRAME, LOOKAHEAD, GAP, CAP = 0.08, 1.04, 1.0, 20.0
SETS = {
    "en2zh": (lambda: sorted(m for m, s in json.load(open(D / "meetings.json")).items() if s == "test"),
              lambda m: D / f"session_audio/ihm_mix/{m}.Mix-Headset.wav", lambda m: D / f"diar/nemotron_lowlat/{m}.post.pt"),
    "zh2en": (lambda: sorted(m for m, s in json.load(open(D / "meetings.json")).items() if s == "test"),
              lambda m: D / f"session_audio/zh_mix_v8/{m}.wav", lambda m: D / f"diar/nemotron_lowlat_zh8/{m}.post.pt"),
    "ali": (lambda: sorted(json.load(open(A / "meetings.json"))),
            lambda m: A / f"session_audio/near_mix/{m}.wav", lambda m: A / f"diar/nemotron_lowlat/{m}.post.pt"),
}


def online_windows(post: np.ndarray) -> list[tuple[int, int, int]]:
    """(first active frame, last active frame + 1, frame whose arrival closes the window) per window."""
    act = (post > 0.5).any(1)
    gap, cap = round(GAP / FRAME), round(CAP / FRAME)
    out, i, n = [], 0, len(act)
    while i < n:
        if not act[i]:
            i += 1
            continue
        st, last, j = i, i, i
        while j < n and j - st < cap and j - last <= gap:
            if act[j]:
                last = j
            j += 1
        out.append((st, last + 1, j))
        i = last + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    wf = out / f"windows_test_online_{args.set}.jsonl"
    if wf.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {wf}")
    meetings, audio_of, post_of = SETS[args.set]
    (out / "audio" / args.set).mkdir(parents=True, exist_ok=True)
    (out / "post" / args.set).mkdir(parents=True, exist_ok=True)
    n_tot, dur_tot = 0, 0.0
    with open(wf, "x", encoding="utf-8") as fh:
        for m in meetings():
            post = torch.load(post_of(m), weights_only=False)["post80"].float().numpy()
            x, sr = sf.read(str(audio_of(m)), dtype="float32", always_2d=True)
            x = x.mean(1)
            sess_s = len(x) / sr
            for k, (st, en, j) in enumerate(online_windows(post)):
                ws, we = max(0.0, st * FRAME - 0.2), min(sess_s, en * FRAME + 0.3)
                close = min(sess_s, j * FRAME + LOOKAHEAD) if j < len(post) else sess_s
                wid = f"{m}_on_{k:04d}"
                wav = out / "audio" / args.set / f"{wid}.wav"
                sf.write(str(wav), x[round(ws * sr): round(we * sr)], sr)
                f0, nf = round(ws / FRAME), int(np.ceil((we - ws) / FRAME))
                s = post[f0: f0 + nf]
                if len(s) < nf:
                    s = np.concatenate([s, np.zeros((nf - len(s), post.shape[1]), dtype=s.dtype)])
                pp = out / "post" / args.set / f"{wid}.npy"
                np.save(pp, s.astype(np.float16))
                rec = {"id": wid, "meeting": m, "split": "test", "wtype": "online",
                       "dir": "en2zh" if args.set == "en2zh" else "zh2en", "start_s": round(ws, 3),
                       "dur_ms": round((we - ws) * 1000), "audio": str(wav.resolve()),
                       "post": {"lowlat": str(pp.resolve())},
                       "open_detect_ms": round((st * FRAME + LOOKAHEAD - ws) * 1000),
                       "close_detect_ms": round((close - ws) * 1000), "utts": [], "strict_review": False}
                fh.write(json.dumps(rec) + "\n")
                n_tot += 1
                dur_tot += we - ws
            print(f"{m}: {k + 1} windows", flush=True)
    print(json.dumps({"set": args.set, "windows": n_tot, "hours": round(dur_tot / 3600, 2)}))


if __name__ == "__main__":
    main()
