"""Cascade variants for paper v6 (review of v5, M04 / M05; prereg_v6 §5). gstar_cascade.py is left as it is.

  oracle-segment   turns from the reference instead of the diarizer: each speaker's non-hesitation utterances of the
                   window, merged when < --fill s apart, cut with --pad s; slot = the speaker's diarizer slot of the
                   session matching (slot_map.json of the windows' builder). Separates segmentation errors from
                   translation errors (reference boundaries AND reference speakers: an upper bound for the cascade)
  context-segment  the diarizer turns of gstar_cascade.segment, each cut with --ctx s of audio on both sides; the
                   translator's output is later kept only where its own timestamps fall in the turn (assemble-v6 --core)
  assemble-v6      translations of pseudo-windows -> one hypothesis per window, like gstar_cascade.assemble, plus
                     --core      keep only output segments whose predicted span overlaps the turn proper
                     --stream    streaming pseudo-windows (decoded by the online decoder): every block keeps its commit
                                 time, shifted to the window clock; the turn's last read is moved to the moment the turn
                                 is known to be over (its last active frame + --fill + the diarizer's 1.04 s look-ahead)

    python -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_cascade_v6 oracle-segment --windows ... --slot-map ... --out ...
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from experiments.uniss_qwen3_omni_v1.evaluation.gstar_cascade import FRAME, turns

LOOKAHEAD_S = 1.04


def write_piece(out, w, sid, x, sr, post, s0, s1, slot, a_s, b_s, key, fh, extra=None):
    wav = out / "audio" / f"{sid}.wav"
    sf.write(str(wav), x[int(s0 * sr): int(s1 * sr)], sr)
    f0, f1 = int(s0 / FRAME), int(np.ceil(s1 / FRAME))
    pp = out / "post" / f"{sid}.npy"
    sl = post[f0:f1]
    if len(sl) < f1 - f0:
        sl = np.concatenate([sl, np.zeros((f1 - f0 - len(sl), post.shape[1]), dtype=sl.dtype)])
    np.save(pp, sl.astype(np.float16))
    fh.write(json.dumps({"id": sid, "window": w["id"], "meeting": w["meeting"], "audio": str(wav.resolve()),
                         "dur_ms": round((s1 - s0) * 1000), "slot": slot, "start_s": round(a_s, 2),
                         "end_s": round(b_s, 2), "offset_s": round(s0, 3), "post": {key: str(pp.resolve())},
                         **(extra or {})}, ensure_ascii=False) + "\n")


def oracle_segment(args) -> None:
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    (out / "audio").mkdir(parents=True)
    (out / "post").mkdir()
    smap = json.load(open(args.slot_map))[args.preset]
    n = 0
    with open(out / "segments.jsonl", "x", encoding="utf-8") as fh:
        for w in map(json.loads, open(args.windows, encoding="utf-8")):
            x, sr = sf.read(w["audio"], dtype="float32")
            dur = len(x) / sr
            post = np.load(w["post"][args.post_key]).astype(np.float32)
            by = defaultdict(list)
            for u in w["utts"]:
                if not u["hesitation"]:
                    by[u["speaker_id"]].append((u["begin_s"], u["end_s"]))
            runs = []
            for spk, us in by.items():
                us.sort()
                a, b = us[0]
                for c, d in us[1:]:
                    if c - b < args.fill:
                        b = max(b, d)
                    else:
                        runs.append((spk, a, b))
                        a, b = c, d
                runs.append((spk, a, b))
            for j, (spk, a, b) in enumerate(sorted(runs, key=lambda r: (r[1], r[0]))):
                s0, s1 = max(0.0, a - args.pad), min(dur, b + args.pad)
                if s1 - s0 < 0.2:
                    continue
                write_piece(out, w, f"{w['id']}#o{j}", x, sr, post, s0, s1, smap[w["meeting"]][spk], a, b,
                            args.post_key, fh)
                n += 1
    print(json.dumps({"segments": n, "out": str(out)}))


def context_segment(args) -> None:
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    (out / "audio").mkdir(parents=True)
    (out / "post").mkdir()
    n = 0
    with open(out / "segments.jsonl", "x", encoding="utf-8") as fh:
        for w in map(json.loads, open(args.windows, encoding="utf-8")):
            post = np.load(w["post"][args.post_key]).astype(np.float32)
            x, sr = sf.read(w["audio"], dtype="float32")
            dur = len(x) / sr
            ts = turns(post, args.fill, args.min_s) or [(int(post.mean(0).argmax()) + 1, 0, len(post))]
            for j, (k, a, b) in enumerate(ts):
                s0, s1 = max(0.0, a * FRAME - args.ctx), min(dur, b * FRAME + args.ctx)
                if s1 - s0 < 0.2:
                    continue
                write_piece(out, w, f"{w['id']}#c{j}", x, sr, post, s0, s1, k, a * FRAME, b * FRAME, args.post_key, fh,
                            {"core_s": [round(a * FRAME - s0, 2), round(b * FRAME - s0, 2)]})
                n += 1
    print(json.dumps({"segments": n, "out": str(out)}))


def assemble_v6(args) -> None:
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    seg = {s["id"]: s for s in map(json.loads, open(args.seg, encoding="utf-8"))}
    hyp = {}
    for f in args.hyp:
        for h in map(json.loads, open(f, encoding="utf-8")):
            hyp[h["id"]] = h
    by_w = defaultdict(list)
    for s in seg.values():
        by_w[s["window"]].append(s)
    sep = " " if args.dir == "zh2en" else ""
    n_drop = n_keep = 0
    with open(out, "x", encoding="utf-8") as fh:
        for line in open(args.windows, encoding="utf-8"):
            wid = json.loads(line)["id"]
            segs, blocks = [], []
            for s in sorted(by_w.get(wid, []), key=lambda s: (s["start_s"], s["slot"])):
                h = hyp.get(s["id"])
                if h is None:
                    continue
                off = s.get("offset_s", max(0.0, s["start_s"] - args.pad))
                if args.core:
                    a, b = s["core_s"]
                    keep = [g for g in h["segments"] if min(b, max(g["end_s"], g["start_s"] + 0.08)) > max(a, g["start_s"])]
                    n_drop += len(h["segments"]) - len(keep)
                    n_keep += len(keep)
                    text = sep.join(g["text"].strip() for g in keep).strip()
                else:
                    text = h["translation"].strip()
                if not text:
                    continue
                seg_rec = {"spk": s["slot"], "start_s": s["start_s"], "end_s": s["end_s"], "text": text}
                if args.stream:
                    known = round(1000 * (s["end_s"] + args.fill + LOOKAHEAD_S))
                    last = max((b["write_ms"] for b in h.get("blocks", [])), default=0)
                    opened = round(1000 * (s["start_s"] + LOOKAHEAD_S))   # the turn is known to have begun
                    for b in h.get("blocks", []):
                        t = max(round(1000 * off) + b["write_ms"], opened)
                        if b["write_ms"] >= last:
                            t = max(t, known)
                        blocks.append({"write_ms": t, "spk": s["slot"], "start_s": s["start_s"], "text": b["text"]})
                    seg_rec["write_ms"] = max([b["write_ms"] for b in blocks if b["spk"] == s["slot"]] or [0])
                segs.append(seg_rec)
            rec = {"id": wid, "translation": sep.join(x["text"] for x in segs), "segments": segs}
            if args.stream:
                rec["blocks"] = sorted(blocks, key=lambda b: b["write_ms"])
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(out), "core_kept": n_keep, "core_dropped": n_drop}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("oracle-segment")
    a.add_argument("--windows", required=True)
    a.add_argument("--slot-map", required=True)
    a.add_argument("--preset", default="offline")
    a.add_argument("--post-key", default="offline")
    a.add_argument("--fill", type=float, default=0.4)
    a.add_argument("--pad", type=float, default=0.1)
    a.add_argument("--out", required=True)
    a = sub.add_parser("context-segment")
    a.add_argument("--windows", required=True)
    a.add_argument("--post-key", default="offline")
    a.add_argument("--fill", type=float, default=0.4)
    a.add_argument("--min-s", type=float, default=0.24)
    a.add_argument("--ctx", type=float, default=2.0)
    a.add_argument("--out", required=True)
    a = sub.add_parser("assemble-v6")
    a.add_argument("--seg", required=True)
    a.add_argument("--hyp", required=True, nargs="+")
    a.add_argument("--windows", required=True)
    a.add_argument("--dir", default="en2zh")
    a.add_argument("--core", action="store_true")
    a.add_argument("--stream", action="store_true")
    a.add_argument("--fill", type=float, default=0.4)
    a.add_argument("--pad", type=float, default=0.1)
    a.add_argument("--out", required=True)
    args = ap.parse_args()
    {"oracle-segment": oracle_segment, "context-segment": context_segment, "assemble-v6": assemble_v6}[args.cmd](args)


if __name__ == "__main__":
    main()
