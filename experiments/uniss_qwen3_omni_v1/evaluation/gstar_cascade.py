"""Cascade baselines for G-STAR-SI (paper v2): diarize, cut, translate each turn, label it with the diarizer's slot.

  segment   every test window -> its Nemotron-3 speaker turns (posterior > 0.5 per slot on the window's 80 ms slice,
            gaps < --fill s closed, runs >= --min-s kept), each cut from the window audio with --pad s on both sides
            and written as a pseudo-window "<window>#<j>" (its own posterior slice, so --inst gstar can read it)
  assemble  translations of the pseudo-windows (gstar_si_stream.py hyp.jsonl) -> one hypothesis per window:
            a segment per turn, speaker = the turn's slot, start / end = the turn's, text = everything translated
            from it (the speaker labels a fine-tuned translator writes inside a turn are ignored)

The translator is either the original Thinker (--inst sd: "cascade, Qwen3-Omni") or the model trained like the
offline system on the same rows but with a zero speaker stream (O-nosd, --inst gstar --zero-sd: "cascade,
same training") -- the pipeline counterpart of the end-to-end model with identical translation training.

    python -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_cascade segment --windows <windows_test_ref.jsonl> --out <dir>
    python -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_cascade assemble --seg <dir>/segments.jsonl \
        --hyp <seg hyp.jsonl> --out <hyp.jsonl>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

FRAME = 0.08


def turns(post: np.ndarray, fill_s: float, min_s: float) -> list[tuple[int, int, int]]:
    """[n, S] posteriors -> (slot, first frame, end frame) runs of activity per slot."""
    out = []
    fill, mn = round(fill_s / FRAME), round(min_s / FRAME)
    for k in range(post.shape[1]):
        on = post[:, k] > 0.5
        idx = np.flatnonzero(on)
        if not len(idx):
            continue
        runs, a, b = [], idx[0], idx[0]
        for i in idx[1:]:
            if i - b - 1 < fill:
                b = i
            else:
                runs.append((a, b + 1))
                a = b = i
        runs.append((a, b + 1))
        out += [(k + 1, a, b) for a, b in runs if b - a >= mn]
    return sorted(out, key=lambda r: (r[1], r[0]))


def segment(args) -> None:
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    (out / "audio").mkdir(parents=True)
    (out / "post").mkdir()
    n_seg = n_empty = 0
    with open(out / "segments.jsonl", "x", encoding="utf-8") as fh:
        for line in open(args.windows, encoding="utf-8"):
            w = json.loads(line)
            post = np.load(w["post"][args.post_key]).astype(np.float32)
            x, sr = sf.read(w["audio"], dtype="float32")
            dur = len(x) / sr
            ts = turns(post, args.fill, args.min_s)
            if not ts:   # nothing above threshold: the whole window, slot of the highest mean activity
                ts = [(int(post.mean(0).argmax()) + 1, 0, len(post))]
                n_empty += 1
            for j, (k, a, b) in enumerate(ts):
                s0, s1 = max(0.0, a * FRAME - args.pad), min(dur, b * FRAME + args.pad)
                if s1 - s0 < 0.2:
                    continue
                sid = f"{w['id']}#{j}"
                wav = out / "audio" / f"{sid}.wav"
                sf.write(str(wav), x[int(s0 * sr): int(s1 * sr)], sr)
                f0, f1 = int(s0 / FRAME), int(np.ceil(s1 / FRAME))
                pp = out / "post" / f"{sid}.npy"
                np.save(pp, post[f0:f1].astype(np.float16))
                fh.write(json.dumps({"id": sid, "window": w["id"], "meeting": w["meeting"], "audio": str(wav.resolve()),
                                     "dur_ms": round((s1 - s0) * 1000), "slot": k, "start_s": round(a * FRAME, 2),
                                     "end_s": round(b * FRAME, 2), "post": {args.post_key: str(pp.resolve())}},
                                    ensure_ascii=False) + "\n")
                n_seg += 1
    print(json.dumps({"segments": n_seg, "windows_without_activity": n_empty, "out": str(out)}))


def assemble(args) -> None:
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    seg = {s["id"]: s for s in map(json.loads, open(args.seg, encoding="utf-8"))}
    hyp = {h["id"]: h for h in map(json.loads, open(args.hyp, encoding="utf-8"))}
    by_w = defaultdict(list)
    for sid, s in seg.items():
        by_w[s["window"]].append(s)
    with open(out, "x", encoding="utf-8") as fh:
        for line in open(args.windows, encoding="utf-8"):
            wid = json.loads(line)["id"]
            segs = []
            for s in sorted(by_w.get(wid, []), key=lambda s: (s["start_s"], s["slot"])):
                text = hyp[s["id"]]["translation"].strip() if s["id"] in hyp else ""
                if text:
                    segs.append({"spk": s["slot"], "start_s": s["start_s"], "end_s": s["end_s"], "text": text})
            fh.write(json.dumps({"id": wid, "translation": "".join(x["text"] for x in segs), "segments": segs},
                                ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(out)}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("segment")
    a.add_argument("--windows", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--post-key", default="offline")
    a.add_argument("--fill", type=float, default=0.4)
    a.add_argument("--min-s", type=float, default=0.24)
    a.add_argument("--pad", type=float, default=0.1)
    a = sub.add_parser("assemble")
    a.add_argument("--seg", required=True)
    a.add_argument("--hyp", required=True)
    a.add_argument("--windows", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args()
    {"segment": segment, "assemble": assemble}[args.cmd](args)


if __name__ == "__main__":
    main()
