"""G-STAR-SI streaming rows (docs/multi_speaker_si_plan_v8_streaming.zh-CN.md §3.2), for continuing the offline model.

On the reviewed reference windows of build_ami_gstar_si_rows.py:
  * speaker labels are the Nemotron-3 slots of the 1.04 s streaming preset (its own whole-session Hungarian
    matching of reference speakers to slots; the oracle posteriors follow the same mapping);
  * each translation is cut at its punctuation into chunks, each chunk timed by the AMI words it covers and
    committed --lag s after its last source word (build_ami_ms_rows.utt_chunks); a chunk is the segment
    <|spk k|><|t a|>chunk<|t b|>, segments in commit order;
  * a row per read (every --window s and at the window's end): the audio heard so far, the posteriors the diarizer
    has finalised by then ("post_until_ms" = t - 1040: later frames are zeroed in the loader and at inference),
    the segments committed before as the reply's prefix, the new ones then <|box_end|> (wait) -- or the rest then
    <|im_end|> at the end. Reads with nothing to commit are kept with probability --keep-empty;
  * plus whole-window offline rows (--offline-share of the read rows, from the offline rows file) and single-speaker
    whole-sentence rows (--sd-share), so the offline ability is kept.
Test / validation windows are written with the streaming preset's posteriors and slots for scoring.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.build_ami_gstar_stream_rows --out data/ami_multispk_si_v1/e2e_gstar_stream_v1
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_gstar_si_rows import N_SLOT, q, slot_map
from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_ms_rows import load_words, utt_chunks

LOOKAHEAD_MS = 1040
FRAME = 0.08


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/ami_multispk_si_v1")
    ap.add_argument("--offline", default="data/ami_multispk_si_v1/e2e_gstar_v1")
    ap.add_argument("--manual", default="data/ami_multispk_si_v1/annotations/ami_public_manual_1.6.2.zip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="lowlat")
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--lag", type=float, default=1.0)
    ap.add_argument("--keep-empty", type=float, default=0.5)
    ap.add_argument("--offline-share", type=float, default=0.2)
    ap.add_argument("--sd-share", type=float, default=0.15)
    ap.add_argument("--limit-meetings", type=int, default=0, help="smoke test only")
    ap.add_argument("--seed", type=int, default=20261027)
    args = ap.parse_args()

    data, off, out = Path(args.data), Path(args.offline), Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    rng = random.Random(args.seed)
    win_ms, lag_ms = round(args.window * 1000), round(args.lag * 1000)
    words = load_words(Path(args.manual))
    windows = {sp: [json.loads(l) for l in open(off / f"windows_{sp}_ref.jsonl", encoding="utf-8")]
               for sp in ("train", "validation", "test")}
    meetings = sorted({w["meeting"] for ws in windows.values() for w in ws})
    if args.limit_meetings:
        meetings = meetings[: args.limit_meetings]
    missing = [m for m in meetings if not (data / f"diar/nemotron_{args.preset}/{m}.post.pt").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} sessions have no {args.preset} posteriors yet, e.g. {missing[:3]}")
    smap, posts, oracles = {}, {}, {}
    for m in meetings:   # the streaming preset's own slot matching, and the oracle on its channels
        post = torch.load(data / f"diar/nemotron_{args.preset}/{m}.post.pt", weights_only=False)["post80"].float().numpy()
        z = np.load(off / "oracle" / f"{m}.npz")
        act = {str(s): z["act"][:, i] for i, s in enumerate(z["speakers"])}
        smap[m] = slot_map(act, post)
        o = np.zeros((len(post), N_SLOT), dtype=np.float16)
        for s, k in smap[m].items():
            n = min(len(o), len(act[s]))
            o[:n, k - 1] = np.maximum(o[:n, k - 1], act[s][:n])
        posts[m], oracles[m] = post, o
    (out / "post").mkdir(parents=True)
    (out / "slot_map.json").write_text(json.dumps(smap, indent=0), encoding="utf-8")

    def save(name: str, split: str, w: dict, arr: np.ndarray) -> str:
        f0, n = round(w["start_s"] / FRAME), int(np.ceil(w["dur_ms"] / 1000 / FRAME))
        s = arr[f0: f0 + n]
        if len(s) < n:
            s = np.concatenate([s, np.zeros((n - len(s), arr.shape[1]), dtype=s.dtype)])
        p = out / "post" / name / split / f"{w['id']}.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, s.astype(np.float16))
        return str(p.resolve())

    stats = defaultdict(lambda: defaultdict(int))
    rows = {"train": [], "validation": []}
    for split, ws in windows.items():
        kept = []
        for w in ws:
            m = w["meeting"]
            if m not in smap:
                continue
            w["post"] = {args.preset: save(args.preset, split, w, posts[m]),
                         f"oracle_{args.preset}": save(f"oracle_{args.preset}", split, w, oracles[m])}
            chunks = []
            for u in w["utts"]:
                if u["hesitation"]:
                    continue
                uu = {"begin_time": w["start_s"] + u["begin_s"], "end_time": w["start_s"] + u["end_s"], "zh": u["zh"]}
                for c in utt_chunks(uu, words.get(m, {}).get(u["speaker_id"], []), lag_ms, w["start_s"]):
                    c["commit_ms"] = min(c["commit_ms"], w["dur_ms"])
                    c["piece"] = (f"<|spk{smap[m][u['speaker_id']]}|><|t{q(c['src_start_ms'] / 1000)}|>{c['text']}"
                                  f"<|t{q(c['src_end_ms'] / 1000)}|>")
                    chunks.append(c)
            chunks.sort(key=lambda c: (c["commit_ms"], c["src_start_ms"]))
            w["stream_target"] = {"preset": args.preset, "slot_map": smap[m],
                                  "chunks": [{k: c[k] for k in ("piece", "src_start_ms", "src_end_ms", "commit_ms")}
                                             for c in chunks]}
            w["target"] = {**w.get("target", {}), args.preset: "".join(c["piece"] for c in chunks)}
            kept.append(w)
            stats[split]["windows"] += 1
            if split not in rows:
                continue
            dur = w["dur_ms"]
            reads = list(range(win_ms, dur, win_ms)) + [dur]
            prev = 0
            for t in reads:
                last = t >= dur
                done = [c for c in chunks if c["commit_ms"] <= prev]
                new = [c for c in chunks if c["commit_ms"] > prev and (last or c["commit_ms"] <= t)]
                target = "".join(c["piece"] for c in new)
                if target or last or rng.random() < args.keep_empty:
                    rows[split].append({"id": f"{w['id']}|{args.preset}|{t}", "audio": w["audio"], "dir": "en2zh",
                                        "t_ms": t, "dur_ms": dur, "post": w["post"][args.preset],
                                        "post_until_ms": max(0, t - LOOKAHEAD_MS),
                                        "prefix": "".join(c["piece"] for c in done), "target": target,
                                        "end": "eos" if last else "wait", "kind": "read", "inst": "gstar",
                                        "preset": args.preset})
                    stats[split]["reads"] += 1
                    stats[split]["empty_reads"] += not target
                prev = t
        with open(out / f"windows_{split}_ref.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(w, ensure_ascii=False) + "\n" for w in kept)
    off_rows = [json.loads(l) for l in open(off / "rows_train.jsonl", encoding="utf-8")]
    gstar_off = [r for r in off_rows if r.get("inst") == "gstar"]
    sd_off = [r for r in off_rows if r.get("inst") == "sd"]
    n = len(rows["train"])
    rows["train"] += rng.sample(gstar_off, min(len(gstar_off), round(args.offline_share * n)))
    rows["train"] += rng.sample(sd_off, min(len(sd_off), round(args.sd_share * n)))
    for split, rs in rows.items():
        rng.shuffle(rs)
        with open(out / f"rows_{split}.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rs)
    summary = {sp: dict(v) for sp, v in stats.items()}
    summary["rows"] = {sp: len(rs) for sp, rs in rows.items()}
    summary["rows"]["train_read"] = n
    summary["args"] = vars(args)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
